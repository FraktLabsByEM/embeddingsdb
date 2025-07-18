# Usamos Ubuntu como base para mayor compatibilidad
FROM ubuntu:20.04

# Set console only
ENV DEBIAN_FRONTEND=noninteractive

RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  # Detectar tu distribución && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt update && apt install -y \
    nvidia-container-toolkit


# Instalar dependencias básicas
RUN apt update && apt install -y \
    libsndfile1 \
    python3.9 python3.9-distutils python3-pip \
    libgl1-mesa-glx \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Instalar Tesseract ocr
RUN apt update && apt install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils

# Instalar MongoDB
RUN apt install -y mongodb && \
    mkdir -p /data/db

# Copiar e Instalar dependencias
COPY python/requirements.txt /requirements.txt
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
# Definir el directorio de trabajo
WORKDIR /app
COPY /init.sh /usr/local/bin/init.sh
RUN chmod +x /usr/local/bin/init.sh
COPY /python /app/


# Exponer puertos de MongoDB y Flask
EXPOSE 27017 5000

# Comando de inicio
ENTRYPOINT ["/usr/local/bin/init.sh"]
