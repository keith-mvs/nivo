# NVIDIA Build API Setup Guide

## Current Status

✓ API Key configured in `.env` file
✓ Base client implementation complete
✓ OpenAI-compatible endpoint integration ready
⚠ **Model Access Required**: Llama 3.2 Vision needs activation

---

## Step 1: Activate Llama 3.2 Vision Model

Your API key is configured, but you need to enable access to the vision model.

### Visit the Model Page

1. Go to: https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct
2. Sign in with your NVIDIA account
3. Look for one of these buttons:
   - **"Enable Model"**
   - **"Request Access"**
   - **"Get API Key"** (may need to accept terms)
   - **"Try API"** (this might auto-enable)

### What to Look For

**On the model page, check for:**
- Access restrictions or tier requirements
- "Free" tier availability
- Credit requirements
- Rate limits (usually 100 requests/min for free tier)

**Common Issues:**
- Model might be in preview/beta and require approval
- Free tier might not include vision models
- Some models require credit card on file (even if free)

---

## Step 2: Update Your API Key (If Needed)

If the model requires a different API key or activation:

### Option A: Same Key, Just Enable Model

If you just need to enable the model with your existing key:
1. Click "Enable" on the model page
2. Accept any terms/conditions
3. Wait for confirmation
4. **No changes needed** - your current `.env` file will work

### Option B: Model-Specific Key Required

If Llama 3.2 Vision requires a separate key:

**Edit `.env` file:**
```bash
# General NVIDIA Build API Key (current)
NVIDIA_API_KEY=nvapi-your-current-key-here

# Llama 3.2 Vision Model Key (if different)
LLAMA_VISION_API_KEY=nvapi-paste-vision-specific-key-here
```

**Then update** `src/analyzers/nvidia_build/vision_language.py`:
```python
# In __init__ method, change:
self.api_key = api_key or os.getenv("LLAMA_VISION_API_KEY") or os.getenv("NVIDIA_API_KEY")
```

---

## Step 3: Verify Access

After enabling the model, test the API:

```bash
python test_nvidia_api_live.py "C:/Users/kjfle/Pictures/jpeg/your_test_image.jpeg"
```

### Expected Output (Success)

```
======================================================================
NVIDIA BUILD API - LIVE TEST
======================================================================

Test image: your_test_image.jpeg

----------------------------------------------------------------------
Testing Retail Object Detection
----------------------------------------------------------------------
[OK] API call successful

Results:
  Objects: laptop, coffee mug, notebook, pen
  Object count: 4
  Description: laptop, coffee mug, notebook, pen

----------------------------------------------------------------------
Testing Vision-Language Model (Image Description)
----------------------------------------------------------------------
[OK] API call successful

Generated description:
  "A laptop computer on a desk with a coffee mug and office supplies."

======================================================================
TEST COMPLETE
======================================================================
```

### Expected Errors

**403 Forbidden** = Model not enabled (follow Step 1)
**401 Unauthorized** = API key invalid (check `.env` file)
**404 Not Found** = Wrong endpoint (already fixed in code)
**429 Too Many Requests** = Rate limit exceeded (wait 1 minute)

---

## Step 4: Alternative Models (If Llama Vision Unavailable)

If Llama 3.2 Vision requires paid tier, try these free alternatives:

### Option 1: Nemotron Nano VLM

Edit `test_nvidia_api_live.py`:
```python
vlm = VisionLanguageModel(model="nemotron")  # Instead of default "llama-vision"
```

Check if available at: https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl

### Option 2: Use TensorRT Locally (No API Required)

Skip NVIDIA Build API entirely and use local TensorRT:
```bash
# Already built and ready:
ls models/*.trt

# Run local inference (no API calls):
python scripts/benchmark_ml_performance.py
```

TensorRT models run 100% locally with no API keys or rate limits.

---

## Current .env File

Your `.env` file currently has:

```
# NVIDIA Build API Configuration
NVIDIA_API_KEY=nvapi-9x...j0Rv  # (masked for security)
```

**Action Items:**

1. Visit https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct
2. Enable the model (look for button/link)
3. Verify access granted (usually instant)
4. Run test script: `python test_nvidia_api_live.py <image_path>`
5. If still 403, check for credit requirements or tier restrictions

---

## Testing Checklist

After enabling access:

- [ ] Visit Llama 3.2 Vision model page
- [ ] Click "Enable Model" or "Try API"
- [ ] Accept any terms/conditions
- [ ] Wait for confirmation (usually immediate)
- [ ] Run: `python test_nvidia_api_live.py <image_path>`
- [ ] Verify both object detection and description work
- [ ] Check for 200 OK status (not 403/404)

---

## Support Resources

**NVIDIA Build Documentation:**
- API Docs: https://docs.api.nvidia.com
- Model Catalog: https://build.nvidia.com/models
- Getting Started: https://docs.api.nvidia.com/nim/docs/api-quickstart

**Common Questions:**
- **Q: Is Llama Vision free?**
  - A: Check model page for tier requirements. Many models have free tier with rate limits.

- **Q: Do I need a credit card?**
  - A: Some models require it on file even if free tier.

- **Q: What's the rate limit?**
  - A: Usually 100 requests/minute for free tier.

- **Q: Can I use multiple models?**
  - A: Yes, same API key works for all enabled models.

---

## Troubleshooting

### 403 Forbidden Persists

1. Check https://build.nvidia.com dashboard
2. Look for "My Models" or "Enabled Models"
3. Verify Llama 3.2 Vision is listed
4. Check for pending approvals or restrictions
5. Try enabling from model page directly

### Model Not Available in Free Tier

**Fallback Options:**
1. Use TensorRT locally (already set up)
2. Try alternative vision models (Nemotron, Phi-3 Vision)
3. Use baseline CLIP + DETR (no API needed)

### Need Different Model

Edit model selection in code:
```python
# In vision_language.py, line 33:
self.model_name = "meta/llama-3.2-90b-vision-instruct"  # Larger model
# or
self.model_name = "nvidia/nemotron-nano-12b-v2-vl"  # NVIDIA model
```

---

## Next Steps After Setup

Once API access is working:

1. **Run Benchmarks**: Compare API vs local TensorRT
2. **Integrate with Videos**: Analyze video library with vision descriptions
3. **Test Retail Detection**: Use for product/object identification
4. **Optimize Prompts**: Fine-tune prompts for better results

---

*Last Updated: 2025-11-30*
*Status: Waiting for model activation*
