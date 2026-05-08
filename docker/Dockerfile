ARG BASE_IMAGE=vllm/vllm-openai:v0.20.0
FROM ${BASE_IMAGE}

ARG COMMON_WORKDIR=/app

WORKDIR ${COMMON_WORKDIR}

# Step 1: Setup - Install system dependencies
RUN apt-get update && \
    apt-get install -y git jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Clone vllm-omni v0.20.0 and install
RUN git clone --depth 1 --branch v0.20.0 https://github.com/vllm-project/vllm-omni.git ${COMMON_WORKDIR}/vllm-omni && \
    cd ${COMMON_WORKDIR}/vllm-omni && uv pip install --python "$(python3 -c 'import sys; print(sys.executable)')" --no-cache-dir "."

RUN ln -sf /usr/bin/python3 /usr/bin/python

ENTRYPOINT []