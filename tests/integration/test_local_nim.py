"""Test local NVIDIA NIM server running in WSL."""

import requests
import base64
import sys
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_local_nim(image_path: str, base_url: str = "http://localhost:8000"):
    """
    Test local NVIDIA NIM server.

    Args:
        image_path: Path to test image
        base_url: NIM server URL (default: http://localhost:8000)
    """
    print("=" * 70)
    print("LOCAL NVIDIA NIM - TEST")
    print("=" * 70)
    print(f"\nServer: {base_url}")
    print(f"Image: {Path(image_path).name}")
    print()

    # Encode image
    image_b64 = encode_image(image_path)

    # Test 1: Health check
    print("-" * 70)
    print("Test 1: Server Health Check")
    print("-" * 70)
    try:
        health_response = requests.get(f"{base_url}/v1/health/ready", timeout=5)
        if health_response.status_code == 200:
            print("[OK] NIM server is ready")
        else:
            print(f"[WARNING] Server returned {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to NIM server")
        print("Make sure the Docker container is running in WSL:")
        print("  bash setup_nvidia_nim_wsl.sh")
        return
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return

    print()

    # Test 2: Image Description
    print("-" * 70)
    print("Test 2: Image Description (Vision-Language)")
    print("-" * 70)

    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one clear sentence."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3,
        "stream": False
    }

    try:
        print("Sending request to NIM server...")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            print("[OK] Request successful\n")
            print(f"Description:")
            print(f'  "{description}"')
            print()

            # Print stats
            if "usage" in result:
                usage = result["usage"]
                print(f"Token usage:")
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 3: Object Detection
    print("-" * 70)
    print("Test 3: Object Detection")
    print("-" * 70)

    payload_objects = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List all objects visible in this image as a comma-separated list."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.2,
        "stream": False
    }

    try:
        print("Detecting objects...")
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload_objects,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            objects = result["choices"][0]["message"]["content"]
            print("[OK] Detection successful\n")
            print(f"Objects detected:")
            print(f'  {objects}')
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")

    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("Local NIM Advantages:")
    print("  ✓ No API rate limits")
    print("  ✓ No cloud dependencies")
    print("  ✓ Full privacy (runs locally)")
    print("  ✓ Faster (no network latency)")
    print("  ✓ Free (after initial download)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Find a test image
        import glob
        images = glob.glob("C:/Users/kjfle/Pictures/jpeg/*.jpeg")
        if images:
            test_image = images[0]
        else:
            print("ERROR: No test images found")
            print("Usage: python test_local_nim.py <image_path>")
            sys.exit(1)

    test_local_nim(test_image)
