FROM ubuntu:20.04

# Set console only
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y build-essential \
    software-properties-common

# Install nvidia container toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  # Detectar tu distribución && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt update -y && \
    apt install -y nvidia-container-toolkit


# Install system dependencies
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Tesseract OCR
RUN apt update -y && apt install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    wget \
    curl \
    git \
    libsndfile1 \
    libgl1-mesa-glx

# Install MongoDB
RUN apt install -y mongodb && \
    # Clear cache
    rm -rf /var/lib/apt/lists/*


RUN apt update -y && apt install -y \
    python3.9 python3-pip python3.9-venv python3.9-dev python3-distutils && \
    # Create symbolic links
    ln -sf python3.9 /usr/bin/python
# Copy and Install dependencies
COPY python/requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY /python /app/
# Define workdir
WORKDIR /app

# Install AudioClip
RUN cd /app && \
    git clone https://github.com/facebookresearch/ImageBind.git && \
    cd ImageBind && \
    pip install .

# Copy init script
COPY /init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh


# Expose mongo and app ports
EXPOSE 5001 5000

# Init file
ENTRYPOINT ["/usr/local/bin/init.sh"]
