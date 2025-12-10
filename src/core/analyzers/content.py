"""Content-based image analysis (perceptual hashing, quality, blur detection).

Uses OpenCV and imagehash for CPU-optimized image analysis.
"""

import cv2
import numpy as np
from PIL import Image
import imagehash
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class ContentAnalyzer:
    """Analyze image content for quality and similarity."""

    def __init__(self, num_workers: int = None):
        """
        Initialize content analyzer.

        Args:
            num_workers: Number of worker threads (defaults to CPU count)
        """
        self.num_workers = num_workers or multiprocessing.cpu_count()

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results dictionary
        """
        results = {
            "image_path": image_path,
        }

        try:
            # Load image
            img_pil = Image.open(image_path)

            # Use imdecode to handle Unicode paths on Windows
            try:
                # Read file as binary and decode with OpenCV (handles Unicode paths)
                with open(image_path, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except:
                img_cv = None

            if img_cv is None:
                # Fallback: load with PIL and convert
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Parallel analysis using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, self.num_workers)) as executor:
                # Submit all analysis tasks
                hash_future = executor.submit(self._compute_hashes, img_pil)
                blur_future = executor.submit(self._detect_blur, img_cv)
                quality_future = executor.submit(self._assess_quality, img_cv)
                color_future = executor.submit(self._analyze_colors, img_cv)

                # Collect results
                results.update(hash_future.result())
                results.update(blur_future.result())
                results.update(quality_future.result())
                results.update(color_future.result())

        except Exception as e:
            results["error"] = f"Content analysis failed: {e}"

        return results

    def _compute_hashes(self, img: Image.Image) -> Dict[str, str]:
        """Compute perceptual hashes for similarity detection."""
        try:
            return {
                "phash": str(imagehash.phash(img)),
                "average_hash": str(imagehash.average_hash(img)),
                "dhash": str(imagehash.dhash(img)),
                "whash": str(imagehash.whash(img)),
            }
        except Exception as e:
            return {"hash_error": str(e)}

    def _detect_blur(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Detect blur using Laplacian variance.

        Higher variance = sharper image.
        Typical threshold: < 100 is blurry, > 500 is sharp.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            return {
                "sharpness_score": round(laplacian_var, 2),
                "is_blurry": laplacian_var < 100,
                "sharpness_level": self._categorize_sharpness(laplacian_var),
            }
        except Exception as e:
            return {"blur_error": str(e)}

    def _categorize_sharpness(self, variance: float) -> str:
        """Categorize sharpness level."""
        if variance < 100:
            return "very_blurry"
        elif variance < 300:
            return "slightly_blurry"
        elif variance < 500:
            return "acceptable"
        elif variance < 1000:
            return "sharp"
        else:
            return "very_sharp"

    def _assess_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Assess overall image quality.

        Checks for noise, dynamic range, and exposure.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Noise estimation (using standard deviation of Laplacian)
            noise_level = cv2.Laplacian(gray, cv2.CV_64F).std()

            # Dynamic range (difference between brightest and darkest)
            dynamic_range = gray.max() - gray.min()

            # Exposure (mean brightness)
            brightness = gray.mean()
            exposure_quality = self._categorize_exposure(brightness)

            # Contrast (standard deviation of pixel values)
            contrast = gray.std()

            # Overall quality score (0-100)
            quality_score = self._calculate_quality_score(
                noise_level, dynamic_range, brightness, contrast
            )

            return {
                "quality_score": round(quality_score, 2),
                "noise_level": round(noise_level, 2),
                "dynamic_range": int(dynamic_range),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "exposure": exposure_quality,
            }
        except Exception as e:
            return {"quality_error": str(e)}

    def _categorize_exposure(self, brightness: float) -> str:
        """Categorize exposure level."""
        if brightness < 50:
            return "underexposed"
        elif brightness < 85:
            return "slightly_dark"
        elif brightness < 170:
            return "well_exposed"
        elif brightness < 205:
            return "slightly_bright"
        else:
            return "overexposed"

    def _calculate_quality_score(
        self, noise: float, dynamic_range: float, brightness: float, contrast: float
    ) -> float:
        """Calculate overall quality score (0-100)."""
        # Normalize components
        noise_score = max(0, 100 - (noise * 2))  # Lower noise is better
        range_score = min(100, (dynamic_range / 255) * 100)  # Higher range is better
        exposure_score = 100 - abs(127.5 - brightness) / 1.275  # Closer to middle is better
        contrast_score = min(100, (contrast / 70) * 100)  # Higher contrast is better

        # Weighted average
        quality = (
            noise_score * 0.3 +
            range_score * 0.2 +
            exposure_score * 0.3 +
            contrast_score * 0.2
        )

        return max(0, min(100, quality))

    def _analyze_colors(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze color composition."""
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Dominant colors using k-means
            pixels = img_rgb.reshape(-1, 3).astype(np.float32)

            # Sample if too many pixels (for performance)
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 5
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )

            # Count pixels in each cluster
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)

            dominant_colors = []
            for idx in sorted_indices[:3]:  # Top 3 colors
                color = centers[idx].astype(int)
                percentage = (counts[idx] / len(labels)) * 100
                dominant_colors.append({
                    "rgb": color.tolist(),
                    "percentage": round(percentage, 1),
                })

            # Color temperature (warm vs cool)
            avg_color = img_rgb.mean(axis=(0, 1))
            warmth = (avg_color[0] - avg_color[2]) / 255  # R - B normalized

            return {
                "dominant_colors": dominant_colors,
                "average_color": avg_color.astype(int).tolist(),
                "color_temperature": "warm" if warmth > 0.1 else "cool" if warmth < -0.1 else "neutral",
                "warmth_score": round(warmth, 3),
            }
        except Exception as e:
            return {"color_error": str(e)}

    def compare_images(self, hash1: str, hash2: str) -> int:
        """
        Compare two perceptual hashes.

        Args:
            hash1: First image hash (hex string)
            hash2: Second image hash (hex string)

        Returns:
            Hamming distance (0 = identical, lower = more similar)
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2
        except:
            return 999  # Invalid comparison
