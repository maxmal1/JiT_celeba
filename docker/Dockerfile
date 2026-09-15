FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --break-system-packages -r requirements.txt