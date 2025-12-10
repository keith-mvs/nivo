"""Image I/O utilities with format support."""

import io
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import piexif

# Supported image extensions
SUPPORTED_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
    '.webp', '.heic', '.heif'
}

RAW_FORMATS = {
    '.raw', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2'
}


def is_supported_image(file_path: str) -> bool:
    """
    Check if file is a supported image format.

    Args:
        file_path: Path to file

    Returns:
        True if supported
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS or ext in RAW_FORMATS


def load_image(file_path: str) -> Optional[Image.Image]:
    """
    Load image from file with format support.

    Args:
        file_path: Path to image file

    Returns:
        PIL Image object or None if failed
    """
    try:
        ext = Path(file_path).suffix.lower()

        # Handle HEIC/HEIF
        if ext in {'.heic', '.heif'}:
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                print(f"Warning: pillow-heif not installed, cannot open {file_path}")
                return None

        # Handle RAW formats
        if ext in RAW_FORMATS:
            try:
                import rawpy
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess()
                return Image.fromarray(rgb)
            except ImportError:
                print(f"Warning: rawpy not installed, cannot open RAW file {file_path}")
                return None
            except Exception as e:
                print(f"Error processing RAW file {file_path}: {e}")
                return None

        # Standard formats
        img = Image.open(file_path)
        img.load()  # Force load to catch issues early
        return img

    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def save_image_with_exif(
    image: Image.Image,
    output_path: str,
    exif_dict: Optional[dict] = None,
    quality: int = 95,
    optimize: bool = True
) -> bool:
    """
    Save image with EXIF data preservation.

    Args:
        image: PIL Image object
        output_path: Output file path
        exif_dict: EXIF dictionary (piexif format)
        quality: JPEG quality (1-100)
        optimize: Enable optimization

    Returns:
        True if successful
    """
    try:
        ext = Path(output_path).suffix.lower()
        save_kwargs = {}

        # JPEG settings
        if ext in {'.jpg', '.jpeg'}:
            save_kwargs.update({
                'quality': quality,
                'optimize': optimize,
                'progressive': True,
            })

            # Add EXIF if provided
            if exif_dict:
                try:
                    exif_bytes = piexif.dump(exif_dict)
                    save_kwargs['exif'] = exif_bytes
                except Exception as e:
                    print(f"Warning: Could not embed EXIF: {e}")

        # PNG settings
        elif ext == '.png':
            save_kwargs.update({
                'compress_level': 6,
                'optimize': optimize,
            })

        # Convert RGBA to RGB for JPEG
        if ext in {'.jpg', '.jpeg'} and image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save image
        image.save(output_path, **save_kwargs)
        return True

    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def get_image_dimensions(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions without loading full image.

    Args:
        file_path: Path to image

    Returns:
        (width, height) tuple or None
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        return None


def has_transparency(image: Image.Image) -> bool:
    """
    Check if image has transparency channel.

    Args:
        image: PIL Image object

    Returns:
        True if image has alpha channel
    """
    return image.mode in ('RGBA', 'LA', 'PA') or (
        image.mode == 'P' and 'transparency' in image.info
    )
