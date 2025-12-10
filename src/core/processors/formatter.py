"""Format converter for images - converts to most compatible formats."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import piexif

from ..utils.image_io import load_image, save_image_with_exif, has_transparency


class ImageFormatter:
    """Convert images to standardized, compatible formats."""

    def __init__(
        self,
        photo_format: str = "jpg",
        graphic_format: str = "png",
        jpeg_quality: int = 95,
        png_compression: int = 6,
        preserve_exif: bool = True,
        safe_conversion: bool = True,
    ):
        """
        Initialize formatter.

        Args:
            photo_format: Target format for photos (jpg recommended)
            graphic_format: Target format for graphics with transparency (png)
            jpeg_quality: JPEG quality (1-100, 95 recommended)
            png_compression: PNG compression level (0-9, 6 is good balance)
            preserve_exif: Preserve EXIF data during conversion
            safe_conversion: Keep original if conversion fails
        """
        self.photo_format = photo_format.lower()
        self.graphic_format = graphic_format.lower()
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.preserve_exif = preserve_exif
        self.safe_conversion = safe_conversion

    def convert_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        force_format: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert single image file.

        Args:
            input_path: Path to input image
            output_path: Optional output path (auto-generated if None)
            force_format: Force specific format (overrides auto-detection)

        Returns:
            Path to converted file or None if failed
        """
        try:
            # Load image
            img = load_image(input_path)
            if img is None:
                print(f"Failed to load image: {input_path}")
                return None

            # Determine target format
            if force_format:
                target_format = force_format.lower()
            else:
                target_format = self._determine_format(img)

            # Check if conversion needed
            current_ext = Path(input_path).suffix.lower().lstrip('.')
            if current_ext == target_format:
                print(f"Already in target format: {input_path}")
                return input_path

            # Generate output path
            if output_path is None:
                output_path = self._generate_output_path(input_path, target_format)

            # Load EXIF if preserving
            exif_dict = None
            if self.preserve_exif:
                try:
                    exif_dict = piexif.load(input_path)
                except:
                    pass  # No EXIF or couldn't read

            # Convert and save
            success = save_image_with_exif(
                img,
                output_path,
                exif_dict=exif_dict,
                quality=self.jpeg_quality,
            )

            if success:
                # Verify conversion
                if os.path.exists(output_path):
                    return output_path
                else:
                    print(f"Conversion succeeded but output not found: {output_path}")
                    return None
            else:
                if self.safe_conversion:
                    print(f"Conversion failed, keeping original: {input_path}")
                    return input_path
                return None

        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            if self.safe_conversion:
                return input_path
            return None

    def convert_batch(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """
        Convert multiple images.

        Args:
            input_paths: List of input image paths
            output_dir: Optional output directory
            show_progress: Show progress

        Returns:
            Dictionary mapping input_path -> output_path
        """
        results = {}

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        total = len(input_paths)
        for i, input_path in enumerate(input_paths, 1):
            if show_progress:
                print(f"Converting {i}/{total}: {Path(input_path).name}")

            # Generate output path if using output_dir
            output_path = None
            if output_dir:
                img = load_image(input_path)
                if img:
                    target_format = self._determine_format(img)
                    filename = Path(input_path).stem + f".{target_format}"
                    output_path = str(Path(output_dir) / filename)

            # Convert
            converted_path = self.convert_file(input_path, output_path)
            if converted_path:
                results[input_path] = converted_path

        print(f"\nConverted {len(results)}/{total} files successfully")
        return results

    def _determine_format(self, img: Image.Image) -> str:
        """
        Determine optimal output format for image.

        Args:
            img: PIL Image object

        Returns:
            Format string (jpg or png)
        """
        # Use PNG for images with transparency
        if has_transparency(img):
            return self.graphic_format

        # Use PNG for small graphics, screenshots
        width, height = img.size
        if width < 1000 and height < 1000 and img.mode in ('P', 'L'):
            return self.graphic_format

        # Default to JPEG for photos
        return self.photo_format

    def _generate_output_path(self, input_path: str, target_format: str) -> str:
        """Generate output path with new extension."""
        input_pathobj = Path(input_path)
        return str(input_pathobj.parent / f"{input_pathobj.stem}.{target_format}")

    def get_format_stats(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Analyze format distribution in file list.

        Args:
            file_paths: List of image file paths

        Returns:
            Dictionary with format statistics
        """
        stats = {
            "total": len(file_paths),
            "by_format": {},
            "needs_conversion": 0,
        }

        for path in file_paths:
            ext = Path(path).suffix.lower().lstrip('.')
            stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1

            # Check if needs conversion
            try:
                img = load_image(path)
                if img:
                    target = self._determine_format(img)
                    if ext != target:
                        stats["needs_conversion"] += 1
            except:
                pass

        return stats

    def estimate_space_impact(self, file_paths: List[str]) -> Dict[str, float]:
        """
        Estimate storage impact of format conversion.

        Args:
            file_paths: List of image file paths

        Returns:
            Dictionary with size estimates
        """
        current_size = sum(Path(p).stat().st_size for p in file_paths)

        # Rough estimates based on format
        estimated_size = current_size  # Start with current

        for path in file_paths:
            ext = Path(path).suffix.lower().lstrip('.')
            file_size = Path(path).stat().st_size

            try:
                img = load_image(path)
                if img:
                    target = self._determine_format(img)

                    # Estimate size change
                    if ext in ['png', 'bmp', 'tiff'] and target == 'jpg':
                        # PNG/BMP to JPEG usually reduces size significantly
                        estimated_size -= file_size * 0.6  # ~60% reduction
                    elif ext in ['heic', 'heif'] and target == 'jpg':
                        # HEIC to JPEG is variable
                        estimated_size -= file_size * 0.2  # ~20% reduction
                    elif ext == 'jpg' and target == 'png':
                        # JPEG to PNG usually increases size
                        estimated_size += file_size * 0.5  # ~50% increase

            except:
                pass

        return {
            "current_mb": current_size / 1_000_000,
            "estimated_mb": estimated_size / 1_000_000,
            "savings_mb": (current_size - estimated_size) / 1_000_000,
            "savings_percent": ((current_size - estimated_size) / current_size * 100) if current_size > 0 else 0,
        }
