"""Base client for NVIDIA Build API."""

import os
import base64
import requests
import time
from typing import Dict, Any, Optional
from pathlib import Path


class NVIDIABuildClient:
    """
    Base client for NVIDIA Build API.

    Provides common functionality for API authentication, request handling,
    and response parsing.

    Attributes:
        BASE_URL: NVIDIA Build API base URL
        api_key: API key for authentication
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NVIDIA Build API client.

        Args:
            api_key: NVIDIA API key (if None, reads from NVIDIA_API_KEY env var)

        Raises:
            ValueError: If API key not found
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY not found. "
                "Set environment variable or pass api_key parameter.\n"
                "Get your API key at: https://build.nvidia.com"
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Rate limiting
        self.requests_per_minute = 100
        self.request_times = []

    def _wait_for_rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        # Check if we've hit the limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until oldest request expires
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_times = []

        self.request_times.append(now)

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image file to base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Make POST request to NVIDIA Build API.

        Args:
            endpoint: API endpoint path
            data: Request payload
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails after all retries
        """
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(retry_attempts):
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Make request
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )

                # Check for errors
                response.raise_for_status()

                return response.json()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    print(f"Rate limit hit (attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                elif response.status_code >= 500:  # Server error
                    print(f"Server error (attempt {attempt + 1}/{retry_attempts})")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # Client error - don't retry
                    raise

            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError(f"Failed after {retry_attempts} attempts")

    def test_connection(self) -> bool:
        """
        Test API connection and authentication.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test request (adjust based on actual API)
            response = requests.get(
                f"{self.BASE_URL}/models",  # Example endpoint
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
