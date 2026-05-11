ARG BASE_IMAGE=vllm/vllm-openai:v0.20.0
FROM ${BASE_IMAGE}

ARG COMMON_WORKDIR=/app

WORKDIR ${COMMON_WORKDIR}

# Step 1: Setup - Install system dependencies
RUN apt-get update && \
    apt-get install -y git jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Clone vllm-omni v0.20.0
RUN git clone --depth 1 --branch v0.20.0 https://github.com/vllm-project/vllm-omni.git ${COMMON_WORKDIR}/vllm-omni

# Step 3: Apply patches - override with modified source files (8 kHz output)
COPY vllm_omni/entrypoints/openai/serving_speech.py \
     ${COMMON_WORKDIR}/vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py
COPY vllm_omni/entrypoints/openai/serving_speech_stream.py \
     ${COMMON_WORKDIR}/vllm-omni/vllm_omni/entrypoints/openai/serving_speech_stream.py

# Step 4: Install vllm-omni with patches applied
RUN cd ${COMMON_WORKDIR}/vllm-omni && uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" --no-cache-dir "."
RUN uv sync --no-cache-dir
RUN ln -sf /usr/bin/python3 /usr/bin/python

ENTRYPOINT []