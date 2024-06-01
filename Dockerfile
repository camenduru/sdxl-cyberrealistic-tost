FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python==4.9.0.80 imageio==2.34.1 imageio-ffmpeg==0.5.0 ffmpeg-python==0.2.0 av==12.1.0 runpod==1.6.2 \
    pillow==10.3.0 peft==0.11.1 einops==0.8.0 transformers==4.41.2 diffusers==0.28.0 accelerate==0.30.1 xformers==0.0.25

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/scheduler/scheduler_config.json -d /content/model/scheduler -o scheduler_config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/resolve/main/text_encoder/pytorch_model.bin -d /content/model/text_encoder -o pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/text_encoder_2/config.json -d /content/model/text_encoder_2 -o config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/resolve/main/text_encoder_2/pytorch_model.bin -d /content/model/text_encoder_2 -o pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer/merges.txt -d /content/model/tokenizer -o merges.txt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer/vocab.json -d /content/model/tokenizer -o vocab.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer_2/merges.txt -d /content/model/tokenizer_2 -o merges.txt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer_2/special_tokens_map.json -d /content/model/tokenizer_2/ -o pecial_tokens_map.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer_2/tokenizer_config.json -d /content/model/tokenizer_2 -o tokenizer_config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/tokenizer_2/vocab.json -d /content/model/tokenizer_2 -o vocab.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/unet/config.json -d /content/model/unet -o config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/resolve/main/unet/diffusion_pytorch_model.bin -d /content/model/unet -o diffusion_pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/vae/config.json -d /content/model/vae -o config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/resolve/main/vae/diffusion_pytorch_model.bin -d /content/model/vae -o diffusion_pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/misri/cyberrealisticXL_v11VAE/raw/main/model_index.json -d /content/model -o model_index.json

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py