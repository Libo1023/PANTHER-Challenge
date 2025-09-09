FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Ensures that Python output to stdout/stderr is not buffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"

# Install git for any potential needs
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"
USER user

# Copy requirements first
COPY --chown=user:user requirements.txt /opt/app/

# Install Python dependencies
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Create necessary directories
RUN mkdir -p /opt/algorithm/nnunet/nnUNet_results && \
    mkdir -p /opt/algorithm/nnunet/nnUNet_preprocessed && \
    mkdir -p /opt/algorithm/nnunet/nnUNet_raw && \
    mkdir -p /opt/algorithm/custom_trainers

# Copy nnUNet results (configuration files)
COPY --chown=user:user ./nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

# Copy custom trainer
COPY --chown=user:user ./custom_trainers/ /opt/algorithm/custom_trainers/

# Copy custom trainer to nnUNet installation directory
USER root
RUN cp /opt/algorithm/custom_trainers/nnUNetTrainer_Student.py \
    /usr/local/lib/python*/dist-packages/nnunetv2/training/nnUNetTrainer/ || \
    cp /opt/algorithm/custom_trainers/nnUNetTrainer_Student.py \
    /home/user/.local/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ || \
    echo "Warning: Could not copy trainer to standard location"
USER user

WORKDIR /opt/app

# Copy inference script
COPY --chown=user:user inference.py /opt/app/

# Set environment variables
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT ["python", "inference.py"]