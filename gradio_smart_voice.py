"""Smart Voice batch TTS demo for VoxCPM2.

每行文本单独发一个请求，把所有 PCM 拼成一段完整音频后由 Gradio 播放。

Usage:
    # Start the vLLM server first:
    vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000

    # Then launch the demo:
    python gradio_smart_voice.py --api-base http://localhost:8000

    不需要依赖"Secure Context 限制"，在 http 环境下也能运行。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator

import gradio as gr
import httpx
import numpy as np

logger = logging.getLogger(__name__)


# 服务端 /v1/audio/speech 流式 PCM 输出固定为 16 kHz、单声道、int16
# (vllm_omni/entrypoints/openai/serving_speech.py 中 _OUTPUT_SAMPLE_RATE = 16000)
SAMPLE_RATE = 16000
PCM_DTYPE = np.int16
PCM_BYTES_PER_SAMPLE = 2


def split_lines(text: str) -> list[str]:
    """严格按 ``\n`` 分行，去掉前后空白，过滤掉空行。"""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _fetch_pcm_for_line(
    client: httpx.Client,
    api_base: str,
    api_key: str,
    model: str,
    voice: str,
    line: str,
    instructions: str = "",
) -> bytes:
    """对单行文本调用 /v1/audio/speech，把流式 PCM 全部收完后返回。

    这里仍然用 ``stream=True``，是为了让服务端尽早开始返回数据、降低 TTFB；
    但客户端不向 Gradio 边收边播，而是聚合后一次性产出。
    """
    url = api_base.rstrip("/") + "/v1/audio/speech"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict = {
        "model": model,
        "input": line,
        "voice": voice,
        "response_format": "pcm",
        "stream": True,
    }
    if instructions:
        payload["instructions"] = instructions

    buf = bytearray()
    with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code >= 400:
            body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"TTS request failed [{resp.status_code}]: {body[:500]}"
            )
        for chunk in resp.iter_bytes():
            if chunk:
                buf.extend(chunk)

    # int16 PCM 必须 2 字节对齐；尾部不齐通常意味着上游被截断，丢弃即可。
    if len(buf) % PCM_BYTES_PER_SAMPLE:
        drop = len(buf) % PCM_BYTES_PER_SAMPLE
        logger.warning("Discarding %d trailing unaligned PCM byte(s)", drop)
        buf = buf[: len(buf) - drop]
    return bytes(buf)


def synthesize(
    text: str,
    voice: str,
    instructions: str,
    api_base: str,
    api_key: str,
    model: str,
    request_timeout: float,
) -> Iterator[tuple[object, str]]:
    """Gradio 生成器：先逐行合成只更新状态，最后一次性给出完整音频。

    Yields:
        (audio_update, status_text) - 中间步骤 audio_update 用 ``gr.update()``
        表示不更改音频组件；最后一次给 ``(SAMPLE_RATE, np.ndarray)``。
    """
    lines = split_lines(text)
    if not lines:
        gr.Warning("文本为空（按 \\n 分行并过滤空行后没有内容）")
        return
    if not voice.strip():
        gr.Warning("Voice 不能为空")
        return

    voice = voice.strip()
    total = len(lines)
    timeout = httpx.Timeout(connect=10.0, read=request_timeout, write=30.0, pool=10.0)

    # 开始时立即清空旧音频并显示初始状态
    yield None, f"[0/{total}] 准备中…"

    pieces: list[np.ndarray] = []
    with httpx.Client(timeout=timeout) as client:
        for idx, line in enumerate(lines, 1):
            preview = line[:40] + ("…" if len(line) > 40 else "")
            yield gr.update(), f"[{idx}/{total}] 正在合成: {preview}"
            try:
                pcm = _fetch_pcm_for_line(
                    client, api_base, api_key, model, voice, line, instructions
                )
            except (httpx.HTTPError, RuntimeError) as e:
                err = f"[{idx}/{total}] 合成失败: {e}"
                logger.exception(err)
                gr.Error(err)
                return

            if pcm:
                pieces.append(np.frombuffer(pcm, dtype=PCM_DTYPE))
            logger.info(
                "[%d/%d] OK %d samples (%.2fs)",
                idx, total,
                pieces[-1].size if pieces else 0,
                (pieces[-1].size if pieces else 0) / SAMPLE_RATE,
            )

    if not pieces:
        gr.Warning("服务端没有返回任何音频数据")
        return

    audio = np.concatenate(pieces)
    duration = audio.size / SAMPLE_RATE
    yield (SAMPLE_RATE, audio), f"完成，共 {total} 行，{duration:.2f} 秒。"


def build_demo(
    api_base: str,
    api_key: str,
    model: str,
    request_timeout: float,
) -> gr.Blocks:
    with gr.Blocks(title="VoxCPM2 Smart Voice", analytics_enabled=False) as demo:
        gr.Markdown(
            "# VoxCPM2 Smart Voice\n"
            "按 `\\n` 分行，每行单独发一个 TTS 请求，全部合成后拼成完整音频播放。\n\n"
            f"**API:** `{api_base}` &nbsp;&nbsp; **Model:** `{model}`"
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    label="文本（按 \\n 分行，空行自动过滤）",
                    lines=12,
                    placeholder="第一行台词\n第二行台词\n...",
                )
                voice = gr.Textbox(
                    label="Voice（已上传到 /v1/audio/voices 的音色名）",
                    value="chengna",
                )
                instructions_in = gr.Textbox(
                    label="Instructions（朗读风格指令，可留空）",
                    lines=3,
                    placeholder="例如：请用温柔、亲切的语气朗读，语速适中。",
                )
                gr.Examples(
                    examples=[
                        [
                            "那您稍后留意来电啊，一般是两三个小时，我们这边会尽快协调。\n"
                            "您稍后保持电话畅通就行。",
                            "chengna",
                        ],
                        [
                            "电话接听场景下，我们的智能语音机器人能自动处理80%的重复来电。\n"
                            "减少基础客服人力成本，还能24小时自动接电话，提升客户满意度。\n"
                            "您是做什么行业的呀？",
                            "chengna",
                        ],
                        [
                            "那您可以试试我们的智能语音机器人。\n"
                            "它能像真人一样接打电话，自动处理通知、回访、咨询等重复性电话。\n"
                            "把人工解放出来~\n"
                            "您之前有用过类似的产品吗？",
                            "chengna",
                        ],
                    ],
                    inputs=[text_in, voice],
                    label="示例",
                )
                go = gr.Button("生成", variant="primary")
                status = gr.Textbox(label="状态", interactive=False, lines=1)
            with gr.Column(scale=1):
                audio_out = gr.Audio(
                    label="合成结果",
                    type="numpy",
                    autoplay=True,
                )

        # NOTE: Gradio uses ``inspect.isgeneratorfunction`` to detect
        # streaming/yielding handlers. Wrap with an explicit generator
        # function (not a lambda) so progress yields actually drive UI updates.
        def on_click(text: str, voice_name: str, instr: str):
            yield from synthesize(
                text, voice_name, instr, api_base, api_key, model, request_timeout,
            )

        go.click(
            fn=on_click,
            inputs=[text_in, voice, instructions_in],
            outputs=[audio_out, status],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base", default="http://localhost:8010",
                        help="vLLM-Omni server base URL (default: %(default)s)")
    parser.add_argument("--api-key", default="sk-empty",
                        help="API key sent in Authorization header (default: %(default)s)")
    parser.add_argument("--model", default="VoxCPM2",
    # parser.add_argument("--model", default="QwenTTS",
                        help="--served-model-name on the vLLM server (default: %(default)s)")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio bind host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio bind port")
    parser.add_argument("--request-timeout", type=float, default=300.0,
                        help="Per-line read timeout in seconds (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="Enable gradio share tunnel")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    demo = build_demo(args.api_base, args.api_key, args.model, args.request_timeout)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
