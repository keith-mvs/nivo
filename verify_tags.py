"""Verify tags were embedded in test images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.processors.tagger import MetadataTagger

# Test files that were tagged
test_files = [
    "D:\\Pictures\\Batch_1\\1636_44_66_file_Uiemeu2wftDEeXvrtcswkx_61CC6F20_1BD1_4F67_BF9.jpeg",
    "D:\\Pictures\\Batch_1\\2017_05_17_file_KeSM8Ug6Eh3m57Xddx6RuJ_WhatsApp_Image_2017_05.jpeg",
    "D:\\Pictures\\Batch_1\\20240728_213530607_iOS.heic",
]

tagger = MetadataTagger()

print("=== VERIFYING EMBEDDED TAGS ===\n")

for file_path in test_files:
    filename = Path(file_path).name
    print(f"File: {filename}")

    tags_data = tagger.read_embedded_tags(file_path)

    if "error" in tags_data:
        print(f"  ERROR: {tags_data['error']}")
    else:
        if "exif_tags" in tags_data:
            tags = tags_data["exif_tags"]
            print(f"  EXIF Tags: {tags[:100]}{'...' if len(tags) > 100 else ''}")

        if "iptc_keywords" in tags_data:
            keywords = tags_data["iptc_keywords"]
            print(f"  IPTC Keywords ({len(keywords)}): {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

        if "description" in tags_data:
            print(f"  Description: {tags_data['description']}")

        if not any(k in tags_data for k in ["exif_tags", "iptc_keywords", "description"]):
            print("  No tags found!")

    print()

print("=== VERIFICATION COMPLETE ===")
