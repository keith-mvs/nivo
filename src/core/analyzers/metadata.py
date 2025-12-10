"""Metadata extraction from images (EXIF, GPS, camera info)."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from PIL import Image
import piexif
from PIL.ExifTags import TAGS, GPSTAGS


class MetadataExtractor:
    """Extract metadata from images."""

    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all available metadata from image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "file_name": Path(image_path).name,
            "file_path": str(image_path),
            "file_size": Path(image_path).stat().st_size,
        }

        try:
            with Image.open(image_path) as img:
                # Basic image info
                metadata.update({
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "megapixels": round((img.width * img.height) / 1_000_000, 2),
                })

                # Extract EXIF
                exif_data = self._extract_exif(img)
                metadata.update(exif_data)

        except Exception as e:
            metadata["error"] = f"Failed to extract metadata: {e}"

        # File dates as fallback
        stat = Path(image_path).stat()
        if "datetime_original" not in metadata:
            metadata["datetime_original"] = datetime.fromtimestamp(stat.st_mtime)
        if "datetime_modified" not in metadata:
            metadata["datetime_modified"] = datetime.fromtimestamp(stat.st_mtime)

        return metadata

    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from PIL Image."""
        exif_data = {}

        try:
            exif = img._getexif()
            if not exif:
                return exif_data

            # Parse standard EXIF tags
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)

                # Handle GPS info separately
                if tag == "GPSInfo":
                    gps_data = self._parse_gps(value)
                    exif_data.update(gps_data)
                    continue

                # Parse datetime fields
                if "DateTime" in str(tag) and isinstance(value, str):
                    try:
                        exif_data[self._snake_case(tag)] = datetime.strptime(
                            value, "%Y:%m:%d %H:%M:%S"
                        )
                    except ValueError:
                        exif_data[self._snake_case(tag)] = value
                else:
                    # Store other tags
                    key = self._snake_case(tag)
                    exif_data[key] = self._clean_value(value)

        except (AttributeError, KeyError, IndexError) as e:
            exif_data["exif_error"] = str(e)

        return exif_data

    def _parse_gps(self, gps_info: dict) -> Dict[str, Any]:
        """Parse GPS EXIF data to lat/lon."""
        gps_data = {}

        try:
            # Decode GPS tags
            gps_decoded = {}
            for tag_id, value in gps_info.items():
                tag = GPSTAGS.get(tag_id, tag_id)
                gps_decoded[tag] = value

            # Convert to decimal degrees
            if "GPSLatitude" in gps_decoded and "GPSLongitude" in gps_decoded:
                lat = self._convert_to_degrees(gps_decoded["GPSLatitude"])
                lon = self._convert_to_degrees(gps_decoded["GPSLongitude"])

                # Apply direction
                if gps_decoded.get("GPSLatitudeRef") == "S":
                    lat = -lat
                if gps_decoded.get("GPSLongitudeRef") == "W":
                    lon = -lon

                gps_data["latitude"] = lat
                gps_data["longitude"] = lon
                gps_data["gps_coordinates"] = f"{lat}, {lon}"

            # Altitude
            if "GPSAltitude" in gps_decoded:
                altitude = float(gps_decoded["GPSAltitude"])
                if gps_decoded.get("GPSAltitudeRef") == 1:
                    altitude = -altitude
                gps_data["altitude"] = altitude

        except (KeyError, ValueError, TypeError) as e:
            gps_data["gps_error"] = str(e)

        return gps_data

    def _convert_to_degrees(self, value) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0

    def _snake_case(self, text: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(text))
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', text).lower()

    def _clean_value(self, value) -> Any:
        """Clean EXIF values for JSON serialization."""
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        elif isinstance(value, (tuple, list)):
            return [self._clean_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        return value

    def get_camera_info(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Extract camera-specific information."""
        camera_info = {}

        if "make" in metadata:
            camera_info["make"] = metadata["make"]
        if "model" in metadata:
            camera_info["model"] = metadata["model"]
        if "lens_model" in metadata:
            camera_info["lens"] = metadata["lens_model"]

        # Settings
        settings = []
        if "f_number" in metadata:
            settings.append(f"f/{metadata['f_number']}")
        if "exposure_time" in metadata:
            exp = metadata['exposure_time']
            if isinstance(exp, tuple):
                settings.append(f"{exp[0]}/{exp[1]}s")
            else:
                settings.append(f"{exp}s")
        if "iso_speed_ratings" in metadata:
            settings.append(f"ISO {metadata['iso_speed_ratings']}")
        if "focal_length" in metadata:
            fl = metadata['focal_length']
            if isinstance(fl, tuple):
                settings.append(f"{fl[0]/fl[1]}mm")
            else:
                settings.append(f"{fl}mm")

        if settings:
            camera_info["settings"] = ", ".join(settings)

        return camera_info
