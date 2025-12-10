#!/bin/bash
# NVIDIA NIM Setup for WSL2
# Run this in WSL to set up local Llama 3.2 Vision inference

set -e  # Exit on error

echo "======================================================================="
echo "NVIDIA NIM - Local Llama 3.2 Vision Setup"
echo "======================================================================="
echo ""

# NGC API Key (paste your key here)
export NGC_API_KEY="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"

# Local cache directory
export LOCAL_NIM_CACHE="$HOME/.cache/nim"
mkdir -p "$LOCAL_NIM_CACHE"

echo "NGC API Key: ${NGC_API_KEY:0:10}...${NGC_API_KEY: -4}"
echo "Cache directory: $LOCAL_NIM_CACHE"
echo ""

# Step 1: Login to NVIDIA Container Registry
echo "Step 1: Logging into nvcr.io..."
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

if [ $? -eq 0 ]; then
    echo "[OK] Docker login successful"
else
    echo "[ERROR] Docker login failed"
    exit 1
fi

echo ""
echo "Step 2: Pulling Llama 3.2 Vision NIM container..."
echo "This may take 10-30 minutes (large model download)"
echo ""

# Step 2: Pull the container (optional, will auto-pull on run)
# Uncomment to pre-download:
# docker pull nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest

# Step 3: Run NVIDIA NIM
echo "Step 3: Starting NVIDIA NIM server..."
echo "Server will be available at: http://localhost:8000"
echo ""
echo "IMPORTANT:"
echo "- First run downloads ~25GB model (be patient!)"
echo "- Server takes 2-5 minutes to start"
echo "- Check logs for 'Application startup complete'"
echo ""
echo "Press Ctrl+C to stop the server when done"
echo ""

docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest

echo ""
echo "NIM server stopped"
