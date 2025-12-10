"""Object detection using NVIDIA Build API vision models."""

from typing import List, Dict, Any
from .client import NVIDIABuildClient


class RetailObjectDetector(NVIDIABuildClient):
    """
    Detect objects and products using NVIDIA Build API VLM.

    Uses Llama 3.2 Vision model with specialized prompts for object detection.
    """

    # OpenAI-compatible chat completions endpoint
    ENDPOINT = "chat/completions"
    MODEL = "meta/llama-3.2-11b-vision-instruct"

    def detect_products(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        max_detections: int = 100
    ) -> Dict[str, Any]:
        """
        Detect objects and products in image using VLM.

        Args:
            image_path: Path to image file
            confidence_threshold: Not used (kept for API compatibility)
            max_detections: Not used (kept for API compatibility)

        Returns:
            Dictionary containing:
            {
                "objects": ["bottle", "laptop", "person"],
                "object_count": 3,
                "product_types": ["bottle"],
                "dominant_product": "bottle",
                "has_products": True,
                "description": "A person using a laptop with a water bottle"
            }

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        # Encode image
        image_b64 = self._encode_image(image_path)

        # Prepare OpenAI-compatible payload with detection prompt
        payload = {
            "model": self.MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List all objects and items visible in this image. Provide a comma-separated list."
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
            "temperature": 0.2
        }

        # Make API request
        response = self._post(self.ENDPOINT, payload)

        # Extract text from response
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = ""

        # Parse objects from response
        objects = [obj.strip().lower() for obj in text.split(",") if obj.strip()]
        objects = objects[:max_detections] if objects else []

        # Identify products (simple heuristic: common retail items)
        product_keywords = ["bottle", "can", "box", "package", "container", "bag", "jar", "product"]
        product_types = [obj for obj in objects if any(kw in obj for kw in product_keywords)]

        # Aggregate results
        dominant_product = product_types[0] if product_types else None

        return {
            "objects": objects,
            "object_count": len(objects),
            "product_types": product_types,
            "dominant_product": dominant_product,
            "has_products": len(product_types) > 0,
            "description": text
        }

    def analyze_video_frame(self, frame_path: str) -> Dict[str, Any]:
        """
        Analyze video frame for retail content.

        Convenience method specifically for video analysis.

        Args:
            frame_path: Path to video frame image

        Returns:
            Simplified results for video tagging:
            {
                "is_product_video": bool,
                "product_tags": ["bottle", "package"],
                "product_count": 3
            }
        """
        results = self.detect_products(frame_path, confidence_threshold=0.6)

        return {
            "is_product_video": results["has_products"],
            "product_tags": results["product_types"][:5],  # Top 5 product types
            "product_count": results["product_count"]
        }
