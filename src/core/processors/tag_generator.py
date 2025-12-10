"""Comprehensive tag generator with 10 hierarchical categories."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter


class TagGenerator:
    """Generate rich, hierarchical tags from image analysis results."""

    # Scene hierarchy mapping
    SCENE_HIERARCHY = {
        "outdoor": ["nature", "landscape", "beach", "mountain", "forest", "city", "street", "park", "garden"],
        "indoor": ["home", "office", "restaurant", "store", "gym", "kitchen", "bedroom", "living_room"],
        "event": ["party", "concert", "sports", "wedding", "birthday", "graduation", "ceremony"],
        "subject": ["portrait", "selfie", "group", "pet", "food", "product"],
        "travel": ["landmark", "architecture", "sunset", "sunrise", "skyline", "monument"],
        "vehicle": ["car", "truck", "motorcycle", "bicycle", "airplane", "boat", "train"],
    }

    # Object semantic groups
    OBJECT_GROUPS = {
        "people": ["person", "man", "woman", "child", "baby", "crowd", "face"],
        "vehicles": ["car", "truck", "motorcycle", "bicycle", "bus", "train", "airplane", "boat"],
        "animals": ["dog", "cat", "bird", "horse", "cow", "elephant", "bear", "fish", "sheep"],
        "electronics": ["laptop", "phone", "tv", "computer", "keyboard", "mouse", "monitor"],
        "furniture": ["chair", "couch", "bed", "table", "desk", "bench", "cabinet"],
        "food": ["pizza", "cake", "sandwich", "fruit", "vegetable", "drink", "bottle"],
        "sports": ["ball", "racket", "skateboard", "surfboard", "ski", "snowboard", "bat"],
        "nature": ["tree", "flower", "plant", "grass", "water", "sky", "cloud", "mountain"],
    }

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "excellent": 90,
        "good": 75,
        "fair": 60,
        "poor": 0,
    }

    # Color temperature mapping
    COLOR_TEMPERATURE = {
        "warm": ["red", "orange", "yellow", "brown"],
        "cool": ["blue", "cyan", "teal", "purple"],
        "neutral": ["gray", "white", "black", "beige"],
        "vibrant": ["magenta", "lime", "pink", "turquoise"],
    }

    def __init__(self, max_tags: int = 30):
        """
        Initialize tag generator.

        Args:
            max_tags: Maximum tags per image
        """
        self.max_tags = max_tags

    def generate_tags(self, analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate comprehensive tags from analysis data.

        Args:
            analysis_data: Dictionary with analysis results

        Returns:
            Dictionary with categorized tags
        """
        tags = {
            "scene": self._generate_scene_tags(analysis_data),
            "objects": self._generate_object_tags(analysis_data),
            "quality": self._generate_quality_tags(analysis_data),
            "color": self._generate_color_tags(analysis_data),
            "temporal": self._generate_temporal_tags(analysis_data),
            "technical": self._generate_technical_tags(analysis_data),
            "format": self._generate_format_tags(analysis_data),
            "people": self._generate_people_tags(analysis_data),
            "location": self._generate_location_tags(analysis_data),
            "mood": self._generate_mood_tags(analysis_data),
        }

        # Flatten and limit total tags
        all_tags = []
        for category, tag_list in tags.items():
            all_tags.extend(tag_list)

        if len(all_tags) > self.max_tags:
            # Prioritize by category importance
            priority_order = ["scene", "objects", "quality", "people", "temporal",
                           "location", "color", "mood", "technical", "format"]
            limited_tags = {}
            count = 0
            for cat in priority_order:
                if count >= self.max_tags:
                    limited_tags[cat] = []
                else:
                    available = self.max_tags - count
                    limited_tags[cat] = tags[cat][:available]
                    count += len(limited_tags[cat])
            tags = limited_tags

        return tags

    def _generate_scene_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate scene-related tags."""
        tags = []

        primary_scene = data.get("primary_scene", "").lower()
        if primary_scene and primary_scene != "unknown":
            tags.append(primary_scene)

            # Add parent category
            for parent, children in self.SCENE_HIERARCHY.items():
                if primary_scene in children or primary_scene == parent:
                    if parent not in tags:
                        tags.append(parent)
                    break

        # Secondary scenes if available
        if "scene_scores" in data:
            for scene, score in data["scene_scores"].items():
                if score > 0.3 and scene.lower() not in tags:
                    tags.append(scene.lower())

        return tags[:5]

    def _generate_object_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate object-related tags."""
        tags = []

        # Get detected objects
        objects = data.get("unique_objects", []) or data.get("detected_objects", [])

        for obj in objects[:10]:
            obj_lower = obj.lower()
            tags.append(obj_lower)

            # Add semantic group
            for group, members in self.OBJECT_GROUPS.items():
                if obj_lower in members and group not in tags:
                    tags.append(f"has_{group}")
                    break

        return tags[:10]

    def _generate_quality_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate quality-related tags."""
        tags = []

        # Quality score
        quality = data.get("quality_score", 0)
        if quality >= self.QUALITY_THRESHOLDS["excellent"]:
            tags.append("excellent_quality")
        elif quality >= self.QUALITY_THRESHOLDS["good"]:
            tags.append("good_quality")
        elif quality >= self.QUALITY_THRESHOLDS["fair"]:
            tags.append("fair_quality")
        else:
            tags.append("poor_quality")

        # Sharpness
        sharpness = data.get("sharpness_level", "").lower()
        if "very_sharp" in sharpness or sharpness == "sharp":
            tags.append("sharp")
        elif "very_blurry" in sharpness:
            tags.append("very_blurry")
        elif "blurry" in sharpness:
            tags.append("blurry")

        # Blur score
        blur_score = data.get("blur_score", 0)
        if blur_score > 500:
            tags.append("crisp")
        elif blur_score < 100:
            tags.append("soft_focus")

        return tags

    def _generate_color_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate color-related tags."""
        tags = []

        # Dominant colors
        dominant_colors = data.get("dominant_colors", [])
        for color in dominant_colors[:3]:
            if isinstance(color, dict):
                color_name = color.get("name", "").lower()
            else:
                color_name = str(color).lower()

            if color_name:
                tags.append(color_name)

                # Add temperature
                for temp, colors in self.COLOR_TEMPERATURE.items():
                    if color_name in colors:
                        if temp not in tags:
                            tags.append(f"{temp}_tones")
                        break

        # Brightness
        brightness = data.get("brightness", 0)
        if brightness > 180:
            tags.append("bright")
        elif brightness < 80:
            tags.append("dark")

        # Contrast
        contrast = data.get("contrast", 0)
        if contrast > 60:
            tags.append("high_contrast")
        elif contrast < 30:
            tags.append("low_contrast")

        return tags[:6]

    def _generate_temporal_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate time-related tags."""
        tags = []

        # Date taken
        date_taken = data.get("date_taken") or data.get("datetime_original")
        if date_taken:
            try:
                if isinstance(date_taken, str):
                    # Handle various date formats
                    for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            dt = datetime.strptime(date_taken, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        dt = None
                else:
                    dt = date_taken

                if dt:
                    # Year
                    tags.append(f"year_{dt.year}")

                    # Season
                    month = dt.month
                    if month in [12, 1, 2]:
                        tags.append("winter")
                    elif month in [3, 4, 5]:
                        tags.append("spring")
                    elif month in [6, 7, 8]:
                        tags.append("summer")
                    else:
                        tags.append("fall")

                    # Time of day
                    hour = dt.hour
                    if 5 <= hour < 12:
                        tags.append("morning")
                    elif 12 <= hour < 17:
                        tags.append("afternoon")
                    elif 17 <= hour < 21:
                        tags.append("evening")
                    else:
                        tags.append("night")

                    # Weekend
                    if dt.weekday() >= 5:
                        tags.append("weekend")

            except Exception:
                pass

        return tags

    def _generate_technical_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate technical camera tags."""
        tags = []

        # Exposure
        exposure = data.get("exposure_time")
        if exposure:
            if isinstance(exposure, (int, float)):
                if exposure < 1/500:
                    tags.append("fast_shutter")
                elif exposure > 1:
                    tags.append("long_exposure")

        # ISO
        iso = data.get("iso")
        if iso:
            try:
                iso_val = int(iso) if isinstance(iso, str) else iso
                if iso_val <= 400:
                    tags.append("low_iso")
                elif iso_val >= 3200:
                    tags.append("high_iso")
            except (ValueError, TypeError):
                pass

        # Aperture
        aperture = data.get("f_number") or data.get("aperture")
        if aperture:
            try:
                aperture_val = float(aperture) if isinstance(aperture, str) else aperture
                if aperture_val <= 2.8:
                    tags.append("wide_aperture")
                elif aperture_val >= 11:
                    tags.append("narrow_aperture")
            except (ValueError, TypeError):
                pass

        # Flash
        if data.get("flash_fired"):
            tags.append("flash_used")

        # Camera
        camera = data.get("camera_model") or data.get("make")
        if camera:
            camera_lower = camera.lower()
            if "iphone" in camera_lower:
                tags.append("iphone")
            elif "samsung" in camera_lower:
                tags.append("samsung")
            elif "canon" in camera_lower:
                tags.append("canon")
            elif "nikon" in camera_lower:
                tags.append("nikon")
            elif "sony" in camera_lower:
                tags.append("sony")

        return tags

    def _generate_format_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate format-related tags."""
        tags = []

        # Resolution
        width = data.get("width", 0)
        height = data.get("height", 0)

        if width and height:
            megapixels = (width * height) / 1_000_000
            if megapixels >= 20:
                tags.append("high_resolution")
            elif megapixels >= 8:
                tags.append("medium_resolution")
            else:
                tags.append("low_resolution")

            # Aspect ratio
            ratio = width / height if height > 0 else 1
            if 0.9 <= ratio <= 1.1:
                tags.append("square")
            elif ratio > 1.7:
                tags.append("panoramic")
            elif ratio > 1.3:
                tags.append("landscape")
            elif ratio < 0.8:
                tags.append("portrait")

            # 4K equivalent
            if width >= 3840 or height >= 2160:
                tags.append("4k_plus")

        # Orientation
        orientation = data.get("orientation")
        if orientation:
            if orientation in [6, 8]:
                tags.append("rotated")

        return tags

    def _generate_people_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate people-related tags."""
        tags = []

        objects = data.get("unique_objects", []) or data.get("detected_objects", [])
        object_counts = data.get("object_counts", {})

        # Count people
        person_count = 0
        if "person" in object_counts:
            person_count = object_counts["person"]
        elif "person" in objects:
            person_count = objects.count("person") if isinstance(objects, list) else 1

        if person_count == 0:
            tags.append("no_people")
        elif person_count == 1:
            tags.append("single_person")
        elif person_count == 2:
            tags.append("couple")
        elif person_count <= 5:
            tags.append("small_group")
        else:
            tags.append("crowd")

        # Face detection
        faces = data.get("face_count", 0)
        if faces > 0:
            tags.append("has_faces")

        return tags

    def _generate_location_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate location-related tags."""
        tags = []

        # GPS data
        gps = data.get("gps") or data.get("gps_coordinates")
        if gps:
            tags.append("geotagged")

            # Handle different GPS formats
            lat = 0
            if isinstance(gps, dict):
                lat = gps.get("latitude", 0)
            elif isinstance(gps, str) and "," in gps:
                try:
                    lat = float(gps.split(",")[0])
                except (ValueError, IndexError):
                    pass

            # Rough hemisphere
            if lat > 0:
                tags.append("northern_hemisphere")
            elif lat < 0:
                tags.append("southern_hemisphere")

        # Indoor/outdoor from scene
        scene = data.get("primary_scene", "").lower()
        if scene in self.SCENE_HIERARCHY.get("indoor", []) or scene == "indoor":
            tags.append("indoor")
        elif scene in self.SCENE_HIERARCHY.get("outdoor", []) or scene == "outdoor":
            tags.append("outdoor")

        # Urban indicators
        urban_scenes = ["city", "street", "architecture", "building", "skyline"]
        if any(s in scene for s in urban_scenes):
            tags.append("urban")

        return tags

    def _generate_mood_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate mood/atmosphere tags."""
        tags = []

        # Infer from colors and brightness
        brightness = data.get("brightness", 128)
        colors = [c.get("name", "").lower() if isinstance(c, dict) else str(c).lower()
                  for c in data.get("dominant_colors", [])]
        scene = data.get("primary_scene", "").lower()

        # Bright, warm colors -> cheerful
        if brightness > 150 and any(c in ["yellow", "orange", "pink"] for c in colors):
            tags.append("cheerful")

        # Dark, cool colors -> moody
        if brightness < 100 and any(c in ["blue", "purple", "gray"] for c in colors):
            tags.append("moody")

        # Sunset/sunrise scenes -> romantic
        if any(s in scene for s in ["sunset", "sunrise", "beach"]):
            tags.append("romantic")

        # Nature scenes -> peaceful
        if any(s in scene for s in ["nature", "forest", "garden", "park", "landscape"]):
            tags.append("peaceful")

        # Sports/action scenes -> energetic
        if any(s in scene for s in ["sports", "action", "concert"]):
            tags.append("energetic")

        return tags

    def get_flat_tags(self, categorized_tags: Dict[str, List[str]]) -> List[str]:
        """
        Flatten categorized tags into single list.

        Args:
            categorized_tags: Dictionary of tag categories

        Returns:
            Flat list of unique tags
        """
        all_tags = []
        for tag_list in categorized_tags.values():
            all_tags.extend(tag_list)
        return list(dict.fromkeys(all_tags))  # Preserve order, remove duplicates

    def get_tag_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for tagged results.

        Args:
            results: List of analysis results with tags

        Returns:
            Summary statistics
        """
        all_tags = Counter()
        category_counts = {cat: Counter() for cat in [
            "scene", "objects", "quality", "color", "temporal",
            "technical", "format", "people", "location", "mood"
        ]}

        for result in results:
            tags = result.get("tags", {})
            if isinstance(tags, dict):
                for category, tag_list in tags.items():
                    for tag in tag_list:
                        all_tags[tag] += 1
                        if category in category_counts:
                            category_counts[category][tag] += 1
            elif isinstance(tags, list):
                for tag in tags:
                    all_tags[tag] += 1

        return {
            "total_images": len(results),
            "unique_tags": len(all_tags),
            "top_tags": all_tags.most_common(20),
            "category_breakdown": {
                cat: counts.most_common(10)
                for cat, counts in category_counts.items()
                if counts
            },
        }
