"""Vision-language model integration using NVIDIA Build API."""

from typing import Optional
from .client import NVIDIABuildClient


class VisionLanguageModel(NVIDIABuildClient):
    """
    Generate natural language descriptions of images using NVIDIA Build API.

    Uses Llama 3.2 Vision or Nemotron Nano VLM models via OpenAI-compatible
    chat completions endpoint.
    """

    # OpenAI-compatible chat completions endpoint
    ENDPOINT = "chat/completions"

    # Available vision models
    MODELS = {
        "llama-vision": "meta/llama-3.2-11b-vision-instruct",
        "nemotron": "nvidia/nemotron-nano-12b-v2-vl",
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-vision"):
        """
        Initialize vision-language model client.

        Args:
            api_key: NVIDIA API key (uses env var if None)
            model: Model to use ("llama-vision" or "nemotron")
        """
        super().__init__(api_key)
        self.model_name = self.MODELS.get(model, self.MODELS["llama-vision"])

    def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> str:
        """
        Generate natural language description of image.

        Args:
            image_path: Path to image file
            prompt: Optional custom prompt (uses default if None)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)

        Returns:
            Natural language description string

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If temperature not in [0, 1]
        """
        # Validate inputs
        if not 0 <= temperature <= 1:
            raise ValueError(f"temperature must be in [0, 1], got {temperature}")

        # Default prompt for general description
        if prompt is None:
            prompt = "Describe what is happening in this image in one clear sentence."

        # Encode image to base64
        image_b64 = self._encode_image(image_path)

        # Prepare OpenAI-compatible payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Make API request
        response = self._post(self.ENDPOINT, payload)

        # Extract text from OpenAI-format response
        try:
            description = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            description = ""

        return description.strip()

    def analyze_video_frame(
        self,
        frame_path: str,
        context: str = "video"
    ) -> str:
        """
        Generate description optimized for video analysis.

        Args:
            frame_path: Path to video frame image
            context: Context for description ("video", "action", "scene")

        Returns:
            Frame description string
        """
        # Context-specific prompts
        prompts = {
            "video": "Describe the activity happening in this video frame",
            "action": "What action is being performed in this image?",
            "scene": "Describe the scene, setting, and environment in this image"
        }

        prompt = prompts.get(context, prompts["video"])

        return self.describe_image(
            frame_path,
            prompt=prompt,
            max_tokens=50,
            temperature=0.2  # More deterministic for tagging
        )

    def generate_searchable_tags(self, image_path: str) -> list:
        """
        Generate searchable tags from image description.

        Args:
            image_path: Path to image file

        Returns:
            List of tag strings extracted from description
        """
        description = self.describe_image(
            image_path,
            prompt="List 3-5 key objects, activities, or concepts visible in this image",
            max_tokens=30,
            temperature=0.1
        )

        # Simple extraction (split by commas/spaces)
        # More sophisticated NLP could be added here
        tags = []
        for part in description.lower().replace(",", " ").split():
            if len(part) > 2 and part.isalpha():
                tags.append(part)

        return list(set(tags))[:5]  # Return up to 5 unique tags
