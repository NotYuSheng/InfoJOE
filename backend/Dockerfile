# Stage 1: Extract libz.so.1 and libz.so.1.2.13 from debian:bookworm-slim
FROM debian:bookworm-slim AS zlib-extract

RUN apt-get update && \
    apt-get install --no-install-recommends -y zlib1g && \
    cp /usr/lib/x86_64-linux-gnu/libz.so.1* /tmp/

# Stage 2: Build the application
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Upgrade setuptools to fix CVE-2024-6345
RUN pip install --upgrade setuptools==70.0.0

# Remove unnecessary system packages
RUN apt-get update && \
    # Remove perl to fix CVE-2023-31484
    DEBIAN_FRONTEND=noninteractive apt-get remove --purge --allow-remove-essential -y perl-base && \
    DEBIAN_FRONTEND=noninteractive apt-get remove --purge -y perl-modules-5.36 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Remove libz1g to fix CVE-2023-45853
# WARNING: This will uninstall the package manager, which will prevent us from removing other packages, so this should be done last
RUN dpkg --purge --force-all zlib1g

# Copy extracted libz shared objects from stage 1
COPY --from=zlib-extract /tmp/libz.so.1 /usr/lib/x86_64-linux-gnu/libz.so.1
COPY --from=zlib-extract /tmp/libz.so.1.2.13 /usr/lib/x86_64-linux-gnu/libz.so.1.2.13

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
