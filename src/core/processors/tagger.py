"""Tag embedding system - writes metadata directly to image files."""

import piexif
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Any
from ..utils.logging_config import get_logger
try:
    from iptcinfo3 import IPTCInfo
    IPTC_AVAILABLE = True
except ImportError:
    IPTC_AVAILABLE = False



logger = get_logger(__name__)
class MetadataTagger:
    """Embed tags and metadata into image files."""

    def __init__(
        self,
        embed_tags: bool = True,
        embed_caption: bool = True,
        embed_keywords: bool = True,
    ):
        """
        Initialize tagger.

        Args:
            embed_tags: Write tags to IPTC keywords
            embed_caption: Generate and write caption
            embed_keywords: Write keywords to EXIF
        """
        self.embed_tags = embed_tags
        self.embed_caption = embed_caption
        self.embed_keywords = embed_keywords

        if not IPTC_AVAILABLE:
            logger.warning("Warning: iptcinfo3 not available, IPTC tagging disabled")

    def embed_metadata(
        self,
        image_path: str,
        analysis_data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Embed analysis results as metadata in image file.

        Args:
            image_path: Path to image file
            analysis_data: Combined analysis data (metadata + content + ML)
            output_path: Optional output path (modifies in-place if None)

        Returns:
            True if successful
        """
        try:
            output_path = output_path or image_path

            # Load existing EXIF
            try:
                exif_dict = piexif.load(image_path)
            except:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

            # Embed tags in EXIF UserComment
            if self.embed_tags and "tags" in analysis_data:
                tags = analysis_data["tags"]
                if isinstance(tags, list):
                    tag_string = ", ".join(tags)
                else:
                    tag_string = str(tags)

                # Write to UserComment in EXIF
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = tag_string.encode("utf-8")

            # Embed quality scores
            if "quality_score" in analysis_data:
                # Store in ImageDescription
                quality_info = f"Quality: {analysis_data['quality_score']}/100"
                if "sharpness_level" in analysis_data:
                    quality_info += f", Sharpness: {analysis_data['sharpness_level']}"

                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = quality_info.encode("utf-8")

            # Save EXIF back to image
            img = Image.open(image_path)

            # Convert RGBA to RGB if saving as JPEG
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                        img = background

            exif_bytes = piexif.dump(exif_dict)
            img.save(output_path, exif=exif_bytes, quality=95)

            # Embed IPTC if available
            if IPTC_AVAILABLE and self.embed_keywords:
                self._embed_iptc(output_path, analysis_data)

            return True

        except Exception as e:
            logger.error(f"Error embedding metadata in {image_path}: {e}")
            return False

    def _embed_iptc(self, image_path: str, analysis_data: Dict[str, Any]) -> bool:
        """Embed IPTC metadata."""
        try:
            info = IPTCInfo(image_path, force=True)

            # Keywords
            if "tags" in analysis_data:
                tags = analysis_data["tags"]
                if isinstance(tags, list):
                    info["keywords"] = [tag.encode("utf-8") for tag in tags]

            # Caption
            if self.embed_caption:
                caption = self._generate_caption(analysis_data)
                if caption:
                    info["caption/abstract"] = caption.encode("utf-8")

            # Scene
            if "primary_scene" in analysis_data:
                info["object name"] = analysis_data["primary_scene"].encode("utf-8")

            # Save
            info.save()
            return True

        except Exception as e:
            logger.error(f"Error embedding IPTC in {image_path}: {e}")
            return False

    def _generate_caption(self, analysis_data: Dict[str, Any]) -> str:
        """Generate descriptive caption from analysis data."""
        parts = []

        # Scene
        if "primary_scene" in analysis_data:
            parts.append(analysis_data["primary_scene"])

        # Objects
        if "unique_objects" in analysis_data:
            objects = analysis_data["unique_objects"][:5]
            if objects:
                parts.append(f"containing {', '.join(objects)}")

        # Quality
        if "quality_score" in analysis_data:
            score = analysis_data["quality_score"]
            if score >= 80:
                parts.append("high quality")
            elif score >= 60:
                parts.append("good quality")

        # Date
        if "datetime_original" in analysis_data:
            dt = analysis_data["datetime_original"]
            if hasattr(dt, "strftime"):
                parts.append(f"taken on {dt.strftime('%Y-%m-%d')}")

        # Location
        if "gps_coordinates" in analysis_data:
            parts.append(f"at {analysis_data['gps_coordinates']}")

        caption = ". ".join(parts).capitalize()
        return caption if caption else "Photo"

    def batch_embed(
        self,
        files_and_data: List[tuple],
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Embed metadata in multiple files.

        Args:
            files_and_data: List of (image_path, analysis_data) tuples
            show_progress: Show progress

        Returns:
            Statistics dict
        """
        stats = {"success": 0, "failed": 0}

        for image_path, analysis_data in files_and_data:
            try:
                success = self.embed_metadata(image_path, analysis_data)
                if success:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                stats["failed"] += 1

        logger.info(f"Embedded metadata in {stats['success']} files")
        if stats["failed"] > 0:
            logger.error(f"Failed: {stats['failed']} files")

        return stats

    def read_embedded_tags(self, image_path: str) -> Dict[str, Any]:
        """Read embedded tags from image."""
        tags_data = {}

        try:
            # Read EXIF
            exif_dict = piexif.load(image_path)

            if piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
                tags_data["exif_tags"] = user_comment.decode("utf-8", errors="ignore")

            if piexif.ImageIFD.ImageDescription in exif_dict["0th"]:
                desc = exif_dict["0th"][piexif.ImageIFD.ImageDescription]
                tags_data["description"] = desc.decode("utf-8", errors="ignore")

            # Read IPTC
            if IPTC_AVAILABLE:
                info = IPTCInfo(image_path)
                if info["keywords"]:
                    tags_data["iptc_keywords"] = [
                        kw.decode("utf-8", errors="ignore")
                        for kw in info["keywords"]
                    ]
                if info["caption/abstract"]:
                    tags_data["iptc_caption"] = info["caption/abstract"].decode("utf-8", errors="ignore")

        except Exception as e:
            tags_data["error"] = str(e)

        return tags_data
