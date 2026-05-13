"""Benchmark script for OpenAI-compatible TTS /v1/audio/speech endpoint.

Metrics
-------
ttft        Time To First Byte (s): latency from request start until the first
            byte of (streamed) audio data arrives.
audio_ttfp  Time To First Audio Packet (s): time until the first non-empty
            audio chunk is received over the streaming connection.
audio_rtf   Audio Real-Time Factor: total_request_time / audio_duration.
            RTF < 1.0 means the server synthesises faster than real-time.

Usage examples
--------------
# Minimal – 20 requests, concurrency 5
python benchmark.py --num-prompts 20 --max-concurrency 5

# With voice cloning reference
python benchmark.py --ref-audio /path/to/ref.wav --num-prompts 50 \\
    --max-concurrency 10 --request-rate 2

# Against a remote server
python benchmark.py --api-base http://192.168.1.10:8010 \\
    --model VoxCPM2 --num-prompts 100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import struct
import time
from dataclasses import dataclass, field
from statistics import mean, quantiles
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Default test prompts (Chinese – typical call-centre sentences)
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS: list[str] = [
    "您好，欢迎致电合力亿捷，请问有什么可以帮您？",
    "您在深圳市福田区人民法院的（2026）粤0307民初2394号知识产权纠纷案件，已依法向您预留的邮箱送达相关法律文书。",
    "您的身份证号后四位是一三零一，请确认是否正确。",
    "法院位置是朝南出发，经过银行后步行一百米，价格是一千九百八十八元。",
    "您看还有什么问题？欢迎和我沟通，我随时为您服务，再见。",
    "感谢您的来电，祝您生活愉快，再见！",
    "请问您需要办理什么业务？我们提供话费充值、套餐变更、账单查询等服务。",
    "您的套餐余量已不足，建议您尽快充值或升级套餐以确保正常使用。",
    "系统检测到您的账户存在异常登录，为保障安全，请尽快修改密码。",
    "本次通话将被录音，以便我们持续改进服务质量，感谢您的理解与配合。",
]


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def wav_duration_from_bytes(data: bytes) -> Optional[float]:
    """Return the duration (seconds) of a PCM WAV file given its raw bytes.

    Returns None if the data is too short or not a valid WAV file.
    """
    if len(data) < 44:
        return None
    try:
        # RIFF header: "RIFF" + size(4) + "WAVE" + "fmt "(4) + chunk_size(4)
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return None
        # fmt sub-chunk starts at offset 12; skip id(4)+size(4)
        num_channels: int = struct.unpack_from("<H", data, 22)[0]
        sample_rate: int = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample: int = struct.unpack_from("<H", data, 34)[0]
        if num_channels == 0 or sample_rate == 0 or bits_per_sample == 0:
            return None
        bytes_per_sample = bits_per_sample // 8
        # Find "data" sub-chunk (may not always be at offset 36)
        offset = 12
        while offset + 8 <= len(data):
            chunk_id = data[offset:offset + 4]
            chunk_size: int = struct.unpack_from("<I", data, offset + 4)[0]
            if chunk_id == b"data":
                num_frames = chunk_size // (bytes_per_sample * num_channels)
                return num_frames / sample_rate
            offset += 8 + chunk_size
        return None
    except struct.error:
        return None


# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    success: bool
    ttft: float = 0.0          # time to first byte  (s)
    audio_ttfp: float = 0.0    # time to first audio packet (s)
    e2e_latency: float = 0.0   # total end-to-end time (s)
    audio_duration: float = 0.0
    audio_rtf: float = 0.0     # e2e_latency / audio_duration
    error: str = ""


# ---------------------------------------------------------------------------
# Single streaming request
# ---------------------------------------------------------------------------

async def single_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        t_start = time.perf_counter()
        ttft = 0.0
        audio_ttfp = 0.0
        first_byte = False
        chunks: list[bytes] = []

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return RequestResult(
                        success=False,
                        error=f"HTTP {resp.status_code}: {body[:200].decode(errors='replace')}",
                    )

                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    now = time.perf_counter()
                    if not first_byte:
                        ttft = now - t_start
                        audio_ttfp = ttft
                        first_byte = True
                    chunks.append(chunk)

            t_end = time.perf_counter()
            e2e = t_end - t_start

            raw_audio = b"".join(chunks)
            duration = wav_duration_from_bytes(raw_audio)
            if duration is None or duration <= 0:
                # Fallback: estimate from byte size assuming 16-bit mono 22050 Hz
                duration = len(raw_audio) / (22050 * 2)

            rtf = e2e / duration if duration > 0 else 0.0

            return RequestResult(
                success=True,
                ttft=ttft,
                audio_ttfp=audio_ttfp,
                e2e_latency=e2e,
                audio_duration=duration,
                audio_rtf=rtf,
            )

        except Exception as exc:  # noqa: BLE001
            return RequestResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Rate-limited dispatcher
# ---------------------------------------------------------------------------

async def run_benchmark(
    api_base: str,
    api_key: str,
    model: str,
    prompts: list[str],
    max_concurrency: int,
    request_rate: float,        # requests/second; <=0 means unlimited
    ref_audio: Optional[str],
    response_format: str,
    timeout: float,
) -> list[RequestResult]:
    url = f"{api_base.rstrip('/')}/v1/audio/speech"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Build base payload
    base_payload: dict = {
        "model": model,
        "response_format": response_format,
    }
    if ref_audio:
        if ref_audio.startswith(("http://", "https://", "data:")):
            base_payload["ref_audio"] = ref_audio
        else:
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
            ext = ref_audio.lower().rsplit(".", 1)[-1]
            mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
                    "flac": "audio/flac", "ogg": "audio/ogg"}.get(ext, "audio/wav")
            with open(ref_audio, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            base_payload["ref_audio"] = f"data:{mime};base64,{b64}"

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[RequestResult] = []
    tasks: list[asyncio.Task] = []

    interval = (1.0 / request_rate) if request_rate > 0 else 0.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, text in enumerate(prompts):
            payload = {**base_payload, "input": text}
            task = asyncio.create_task(
                single_request(client, url, payload, headers, semaphore)
            )
            tasks.append(task)
            print(f"  dispatched {i + 1}/{len(prompts)}", end="\r", flush=True)
            if interval > 0 and i < len(prompts) - 1:
                await asyncio.sleep(interval)

        print()  # newline after \r progress
        results = list(await asyncio.gather(*tasks))

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentiles(values: list[float], pcts: list[float] = (50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in pcts}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in pcts:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        result[f"p{int(p)}"] = sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
    return result


def print_metric_table(
    name: str,
    values: list[float],
    unit: str = "s",
    multiplier: float = 1.0,
) -> None:
    if not values:
        print(f"  {name}: no data")
        return
    vals = [v * multiplier for v in values]
    pcts = percentiles(vals, [50, 90, 95, 99])
    avg = mean(vals)
    mn, mx = min(vals), max(vals)
    print(
        f"  {name:<18}  mean={avg:7.3f}{unit}  "
        f"p50={pcts['p50']:7.3f}  p90={pcts['p90']:7.3f}  "
        f"p95={pcts['p95']:7.3f}  p99={pcts['p99']:7.3f}  "
        f"min={mn:7.3f}  max={mx:7.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark /v1/audio/speech (OpenAI TTS) – outputs ttft, audio_ttfp, audio_rtf"
    )
    p.add_argument("--api-base", default="http://172.16.52.65:8010",
                   help="Server base URL (default: http://localhost:8010)")
    p.add_argument("--api-key", default="sk-empty")
    p.add_argument("--model", default="VoxCPM2")
    p.add_argument("--num-prompts", type=int, default=30,
                   help="Total number of requests to send (default: 20)")
    p.add_argument("--max-concurrency", type=int, default=3,
                   help="Maximum in-flight requests (default: 5)")
    p.add_argument("--request-rate", type=float, default=0,
                   help="Requests/second dispatch rate; 0 = unlimited (default: 0)")
    p.add_argument("--ref-audio", default=None,
                   help="Optional reference audio for voice cloning (path, URL, or data: URI)")
    p.add_argument("--response-format", default="wav",
                   choices=["wav", "mp3", "flac", "opus", "aac"],
                   help="Audio format returned by the server (default: wav)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--prompts-file", default=None,
                   help="Optional text file with one prompt per line")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Build prompt list
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            file_prompts = [ln.strip() for ln in fh if ln.strip()]
        prompts = file_prompts
    else:
        prompts = DEFAULT_PROMPTS

    # Repeat / trim to reach exactly --num-prompts
    n = args.num_prompts
    prompts = (prompts * ((n // len(prompts)) + 1))[:n]

    print("=" * 72)
    print(f"  TTS Benchmark  –  {args.api_base}/v1/audio/speech")
    print(f"  model={args.model}  prompts={n}  concurrency={args.max_concurrency}"
          f"  rate={'unlimited' if args.request_rate <= 0 else f'{args.request_rate} req/s'}")
    print("=" * 72)

    t0 = time.perf_counter()
    results = await run_benchmark(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        prompts=prompts,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        ref_audio=args.ref_audio,
        response_format=args.response_format,
        timeout=args.timeout,
    )
    wall_time = time.perf_counter() - t0

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n  Completed: {len(ok)}/{n}  failed: {len(failed)}"
          f"  wall-time: {wall_time:.1f}s"
          f"  throughput: {len(ok) / wall_time:.2f} req/s\n")

    if failed:
        print(f"  First error: {failed[0].error}\n")

    if not ok:
        print("  No successful requests – cannot compute metrics.")
        return

    print("  Metrics (lower is better for ttft / audio_ttfp / audio_rtf):\n")
    print_metric_table("ttft", [r.ttft for r in ok], unit="s")
    print_metric_table("audio_ttfp", [r.audio_ttfp for r in ok], unit="s")
    print_metric_table("audio_rtf", [r.audio_rtf for r in ok], unit="")
    print()
    print_metric_table("e2e_latency", [r.e2e_latency for r in ok], unit="s")
    print_metric_table("audio_duration", [r.audio_duration for r in ok], unit="s")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
