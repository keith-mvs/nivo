# Running NVIDIA NIM Locally in WSL2

## Why Local NIM?

✓ **No API limits** - Unlimited requests
✓ **No 403 errors** - Full model access
✓ **Privacy** - Runs 100% locally
✓ **Speed** - No network latency
✓ **Free** - After initial ~25GB download

---

## Prerequisites

### 1. WSL2 with Ubuntu
```powershell
# In PowerShell (Windows), check WSL version
wsl --list --verbose

# Should show WSL 2, not WSL 1
# If not, upgrade: wsl --set-version Ubuntu 2
```

### 2. Docker in WSL
```bash
# In WSL (Ubuntu), check Docker
docker --version

# If not installed:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Logout and login to WSL again
```

### 3. NVIDIA Container Toolkit
```bash
# In WSL, install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/os-release.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 4. Verify GPU Access
```bash
# In WSL, test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Should show your RTX 2080 Ti
```

---

## Quick Start

### Step 1: Copy Script to WSL

**From Windows:**
```powershell
# The script is at: C:\Users\kjfle\.projects\nivo\setup_nvidia_nim_wsl.sh

# Copy to WSL home directory:
wsl cp /mnt/c/Users/kjfle/.projects/nivo/setup_nvidia_nim_wsl.sh ~/
```

**OR manually in WSL:**
```bash
cd ~
nano setup_nvidia_nim_wsl.sh
# Paste the script contents
# Ctrl+O to save, Ctrl+X to exit
chmod +x setup_nvidia_nim_wsl.sh
```

### Step 2: Run the Setup Script

```bash
# In WSL
cd ~
bash setup_nvidia_nim_wsl.sh
```

**What happens:**
1. Logs into NVIDIA Container Registry
2. Pulls Llama 3.2 Vision container (~25GB - **takes 10-30 minutes**)
3. Starts NIM server on http://localhost:8000
4. Downloads model weights (~12GB more - **first run only**)

**Wait for this message:**
```
Application startup complete
```

### Step 3: Test from Windows

**Open new PowerShell/terminal (keep NIM server running in WSL):**

```bash
# In Windows (new terminal)
cd C:\Users\kjfle\.projects\nivo
python test_local_nim.py "C:/Users/kjfle/Pictures/jpeg/your_image.jpeg"
```

---

## Manual WSL Commands (Alternative)

If you prefer running commands manually:

```bash
# In WSL

# 1. Set API key
export NGC_API_KEY="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"

# 2. Create cache directory
export LOCAL_NIM_CACHE="$HOME/.cache/nim"
mkdir -p "$LOCAL_NIM_CACHE"

# 3. Login to NVIDIA registry
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# 4. Run NIM server
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest
```

**Note:** Remove `<>` brackets around API key - that was causing your syntax errors!

---

## Testing the Server

### From Windows Python

```bash
python test_local_nim.py "C:/path/to/image.jpeg"
```

### From WSL curl

```bash
# Health check
curl http://localhost:8000/v1/health/ready

# Should return: {"status": "ready"}
```

### From Python Script

```python
import requests

# Check if server is running
response = requests.get("http://localhost:8000/v1/health/ready")
print(response.json())  # {"status": "ready"}
```

---

## Performance Expectations

**RTX 2080 Ti (11GB VRAM):**
- First inference: ~10-15 seconds (model loading)
- Subsequent inferences: ~1-3 seconds per image
- Batch processing: ~2-5 images/second

**Memory Usage:**
- Model: ~8-10GB VRAM
- Overhead: ~1-2GB VRAM
- Total: ~9-12GB (fits on 2080 Ti)

---

## Troubleshooting

### "Docker: command not found"
```bash
# Install Docker in WSL
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Logout and login again
```

### "could not select device driver"
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### "GPU not accessible"
```bash
# Verify GPU in WSL
nvidia-smi

# If not working, update WSL:
# In PowerShell: wsl --update
```

### "Out of memory"
- Llama 3.2 11B needs ~10GB VRAM
- Close other GPU applications
- Use smaller model if needed (7B variant)

### "Connection refused" from Windows
```bash
# Make sure NIM server is running in WSL
# Check with: curl http://localhost:8000/v1/health/ready
```

### "Syntax error" in bash script
- Don't include `$` prompt symbol
- Remove `<>` brackets around API key
- Use Unix line endings (LF, not CRLF)

---

## Integration with Image Engine

Once NIM is running, update the client to use local endpoint:

**Edit `.env`:**
```bash
# Use local NIM instead of cloud API
NVIDIA_API_BASE_URL=http://localhost:8000
```

**Or in code:**
```python
from src.analyzers.nvidia_build import VisionLanguageModel

# Point to local NIM server
vlm = VisionLanguageModel()
vlm.BASE_URL = "http://localhost:8000/v1"

# Now all requests go to local server (no rate limits!)
description = vlm.describe_image("image.jpg")
```

---

## Stopping the Server

```bash
# In WSL terminal where NIM is running
Ctrl+C

# Or kill the container
docker ps  # Find container ID
docker stop <container_id>
```

---

## Persistence

**Model is cached after first download:**
- Location: `~/.cache/nim/` in WSL
- Size: ~25-30GB
- Reused on subsequent runs (instant startup)

**To preserve between WSL sessions:**
```bash
# Models persist in WSL filesystem
# No need to re-download unless you delete cache
```

---

## Next Steps

1. **Start NIM in WSL:**
   ```bash
   bash setup_nvidia_nim_wsl.sh
   ```

2. **Wait for "Application startup complete"** (2-5 minutes first time)

3. **Test from Windows:**
   ```bash
   python test_local_nim.py "your_image.jpeg"
   ```

4. **Integrate with video analyzer** - no more 403 errors!

---

## Comparison: Cloud API vs Local NIM

| Feature | Cloud API | Local NIM |
|---------|-----------|-----------|
| Setup | API key only | Docker + download |
| Speed | Network latency | Local (faster) |
| Rate Limits | 100 req/min | Unlimited |
| Privacy | Cloud processing | 100% local |
| Cost | Free tier limits | Free (after download) |
| Model Access | May require activation | Full access |
| Storage | None | ~30GB disk |
| GPU Required | No | Yes (RTX 2080 Ti) |

**Recommendation:** Use local NIM for development and heavy workloads.

---

*Last Updated: 2025-11-30*
*Your NGC API Key: nvapi-Q1QGT4...hJK7 (configured)*
