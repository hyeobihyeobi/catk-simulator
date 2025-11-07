FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh git curl wget ca-certificates build-essential locales \
    python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# 로케일(Optional)
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

WORKDIR /workspace/LatentDriver
CMD ["zsh"]
