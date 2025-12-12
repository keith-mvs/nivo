"""Main orchestration engine for photo processing pipeline."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

from .utils.config import Config
from .utils.image_io import is_supported_image
from .factories.analyzer_factory import AnalyzerFactory
from .pipeline.analysis_pipeline import AnalysisPipeline
from .processors.deduplicator import Deduplicator
from .processors.renamer import ImageRenamer
from .processors.tagger import MetadataTagger
from .processors.formatter import ImageFormatter


class ImageEngine:
    """Main engine orchestrating the photo processing pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize image engine.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)

        # Initialize components using factory pattern
        self._init_analyzers()
        self._init_processors()

    def _init_analyzers(self):
        """Initialize analysis components using factory pattern."""
        factory = AnalyzerFactory(self.config)
        analyzers = factory.create_all_analyzers()

        # Store analyzers for backwards compatibility
        self.metadata_extractor = analyzers['metadata']
        self.content_analyzer = analyzers['content']
        self.ml_analyzer = analyzers['ml']

        # Create analysis pipeline
        self.analysis_pipeline = AnalysisPipeline(
            metadata_extractor=self.metadata_extractor,
            content_analyzer=self.content_analyzer,
            ml_analyzer=self.ml_analyzer,
            config=self.config,
        )

    def _init_processors(self):
        """Initialize processing components."""
        # Deduplicator
        dedup_config = self.config.get("deduplication", {})
        if not isinstance(dedup_config, dict):
            dedup_config = {}

        self.deduplicator = Deduplicator(
            hash_algorithm=dedup_config.get("hash_algorithm", "sha256"),
            use_quick_hash=True,
            max_workers=self.config.get("processing.max_workers", 4),
        )

        # Renamer
        rename_config = self.config.get("renaming", {})
        if not isinstance(rename_config, dict):
            rename_config = {}

        self.renamer = ImageRenamer(
            pattern=rename_config.get("pattern", "{datetime}"),
            date_format=rename_config.get("date_format", "%Y-%m-%d"),
            time_format=rename_config.get("time_format", "%H%M%S"),
            datetime_format=rename_config.get("datetime_format", "%Y-%m-%d_%H%M%S"),
            collision_suffix=rename_config.get("collision_suffix", "_{seq:03d}"),
            preserve_original=rename_config.get("preserve_original", True),
        )

        # Tagger
        tag_config = self.config.get("tagging", {})
        if not isinstance(tag_config, dict):
            tag_config = {}

        categories = tag_config.get("categories", {})
        if not isinstance(categories, dict):
            categories = {}

        iptc_fields = tag_config.get("iptc_fields", {})
        if not isinstance(iptc_fields, dict):
            iptc_fields = {}

        self.tagger = MetadataTagger(
            embed_tags=tag_config.get("embed_tags", True),
            embed_caption=categories.get("caption", True),
            embed_keywords=iptc_fields.get("keywords", True),
        )

        # Formatter
        format_config = self.config.get("formatting", {})
        if not isinstance(format_config, dict):
            format_config = {}

        self.formatter = ImageFormatter(
            photo_format=format_config.get("photo_format", "jpg"),
            graphic_format=format_config.get("graphic_format", "png"),
            jpeg_quality=format_config.get("jpeg_quality", 95),
            png_compression=format_config.get("png_compression", 6),
            preserve_exif=True,
            safe_conversion=format_config.get("safe_conversion", True),
        )

    def scan_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> List[str]:
        """
        Scan directory for supported images.

        Args:
            directory: Directory path to scan
            recursive: Scan subdirectories

        Returns:
            List of image file paths
        """
        image_files = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_supported_image(file_path):
                        image_files.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and is_supported_image(file_path):
                    image_files.append(file_path)

        print(f"Found {len(image_files)} images in {directory}")
        return sorted(image_files)

    def analyze_images(
        self,
        image_paths: List[str],
        use_batch: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Analyze images with all enabled analyzers.

        Args:
            image_paths: List of image file paths
            use_batch: Use batch processing for ML analysis (more efficient)

        Returns:
            List of combined analysis results
        """
        return self.analysis_pipeline.run(
            image_paths=image_paths,
            use_batch=use_batch,
            show_progress=True,
        )

    def process_pipeline(
        self,
        directory: str,
        output_dir: Optional[str] = None,
        analyze: bool = True,
        dedupe: bool = False,
        rename: bool = False,
        embed_tags: bool = False,
        convert_format: bool = False,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete processing pipeline.

        Args:
            directory: Input directory
            output_dir: Optional output directory
            analyze: Run analysis
            dedupe: Run deduplication
            rename: Rename files
            embed_tags: Embed tags in files
            convert_format: Convert formats
            dry_run: Preview changes without executing

        Returns:
            Pipeline results dictionary
        """
        print(f"\n{'='*60}")
        print(f"IMAGE ENGINE - Processing Pipeline")
        print(f"{'='*60}\n")

        pipeline_results = {
            "directory": directory,
            "dry_run": dry_run,
        }

        # Step 1: Scan directory
        image_paths = self.scan_directory(directory)
        pipeline_results["total_images"] = len(image_paths)

        if len(image_paths) == 0:
            print("No images found. Exiting.")
            return pipeline_results

        # Step 2: Analyze (if enabled)
        analysis_results = None
        if analyze:
            analysis_results = self.analyze_images(image_paths)
            pipeline_results["analysis_results"] = analysis_results

            # Save analysis report
            if self.config.get("output.generate_report"):
                report_path = "image_engine_analysis.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
                print(f"Analysis report saved: {report_path}")

        # Step 3: Deduplication (if enabled)
        if dedupe:
            print("\n=== Deduplication ===\n")
            duplicates = self.deduplicator.find_duplicates(image_paths)
            pipeline_results["duplicates"] = duplicates

            if duplicates:
                strategy = self.config.get("deduplication.keep_strategy", "highest_quality")
                dedup_stats = self.deduplicator.remove_duplicates(
                    duplicates,
                    strategy=strategy,
                    dry_run=dry_run,
                )
                pipeline_results["dedup_stats"] = dedup_stats

        # Step 4: Format conversion (if enabled)
        if convert_format:
            print("\n=== Format Conversion ===\n")
            converted = self.formatter.convert_batch(
                image_paths,
                output_dir=output_dir,
                show_progress=True,
            )
            pipeline_results["converted_files"] = len(converted)

        # Step 5: Rename files (if enabled)
        if rename and analysis_results:
            print("\n=== Renaming Files ===\n")
            rename_map = self.renamer.rename_files(
                analysis_results,
                output_dir=output_dir,
                dry_run=dry_run,
            )
            pipeline_results["renamed_files"] = len(rename_map)

        # Step 6: Embed tags (if enabled)
        if embed_tags and analysis_results:
            print("\n=== Embedding Tags ===\n")
            files_and_data = [(r["file_path"], r) for r in analysis_results if "file_path" in r]
            tag_stats = self.tagger.batch_embed(files_and_data)
            pipeline_results["tagged_files"] = tag_stats["success"]

        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}\n")

        return pipeline_results

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and GPU status."""
        info = {
            "config_path": self.config.config_path,
            "analyzers": {
                "metadata": True,
                "content": self.content_analyzer is not None,
                "ml": self.ml_analyzer is not None,
            },
        }

        if self.ml_analyzer:
            info["ml_info"] = {
                "device": str(self.ml_analyzer.device),
                "batch_size": self.ml_analyzer.batch_size,
            }

            # GPU memory info
            memory = self.ml_analyzer.get_memory_usage()
            if memory:
                info["gpu_memory"] = memory

        return info
