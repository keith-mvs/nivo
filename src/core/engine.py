"""Main orchestration engine for photo processing pipeline."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm

from .utils.config import Config
from .utils.image_io import is_supported_image
from .utils.gpu_monitor import get_monitor
from .analyzers.metadata import MetadataExtractor
from .analyzers.content import ContentAnalyzer
from .analyzers.ml_vision import MLVisionAnalyzer
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

        # Initialize components
        self._init_analyzers()
        self._init_processors()

    def _init_analyzers(self):
        """Initialize analysis components."""
        # Metadata extractor (always enabled)
        self.metadata_extractor = MetadataExtractor()

        # Content analyzer
        if self.config.get("analysis.content_analysis"):
            num_workers = self.config.get("processing.max_workers", 4)
            self.content_analyzer = ContentAnalyzer(num_workers=num_workers)
        else:
            self.content_analyzer = None

        # ML analyzer
        if self.config.get("analysis.ml_analysis"):
            ml_config = self.config.get("analysis.ml_models", {})
            # Ensure ml_config is a dict
            if not isinstance(ml_config, dict):
                ml_config = {}

            # Check if YOLO optimization is enabled (highest priority)
            use_yolo = ml_config.get("use_yolo", False)

            if use_yolo:
                # Use YOLOv8-optimized analyzer (3-5x faster object detection)
                from .analyzers.ml_vision_yolo import YOLOVisionAnalyzer
                self.ml_analyzer = YOLOVisionAnalyzer(
                    use_gpu=ml_config.get("use_gpu", True),
                    batch_size=ml_config.get("batch_size", 16),
                    scene_model=ml_config.get("scene_detection", "openai/clip-vit-base-patch32"),
                    yolo_model=ml_config.get("yolo_model", "yolov8n.pt"),
                    precision=ml_config.get("tensorrt_precision", "fp16"),
                    min_confidence=self.config.get("tagging.min_confidence", 0.6),
                )
            else:
                # Check if TensorRT optimization is enabled
                use_tensorrt = ml_config.get("use_tensorrt", False)

                if use_tensorrt:
                    # Use TensorRT-optimized analyzer
                    from .analyzers.ml_vision_tensorrt import TensorRTVisionAnalyzer
                    self.ml_analyzer = TensorRTVisionAnalyzer(
                        use_gpu=ml_config.get("use_gpu", True),
                        batch_size=ml_config.get("batch_size", 16),
                        scene_model=ml_config.get("scene_detection", "openai/clip-vit-base-patch32"),
                        object_model=ml_config.get("object_detection", "facebook/detr-resnet-50"),
                        use_tensorrt=True,
                        precision=ml_config.get("tensorrt_precision", "fp16"),
                        min_confidence=self.config.get("tagging.min_confidence", 0.6),
                    )
                else:
                    # Use standard PyTorch analyzer
                    self.ml_analyzer = MLVisionAnalyzer(
                        use_gpu=ml_config.get("use_gpu", True),
                        batch_size=ml_config.get("batch_size", 8),
                        scene_model=ml_config.get("scene_detection", "openai/clip-vit-base-patch32"),
                        object_model=ml_config.get("object_detection", "facebook/detr-resnet-50"),
                        min_confidence=self.config.get("tagging.min_confidence", 0.6),
                    )
        else:
            self.ml_analyzer = None

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
        print(f"\n=== Analyzing {len(image_paths)} images ===")

        # Show GPU status if ML analysis is enabled
        if self.ml_analyzer and self.ml_analyzer.device.type == "cuda":
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()

        results = []

        # Phase 1: Metadata extraction (CPU-bound, parallel)
        print("Phase 1/3: Extracting metadata...")
        metadata_results = []
        with ThreadPoolExecutor(max_workers=self.config.get("processing.max_workers", 4)) as executor:
            futures = {executor.submit(self.metadata_extractor.extract, path): path for path in image_paths}

            with tqdm(total=len(image_paths), desc="Metadata", unit="img", ncols=100) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        metadata_results.append(result)
                        # Show current file being processed
                        if "file_name" in result:
                            pbar.set_postfix_str(f"Current: {result['file_name'][:40]}")
                    except Exception as e:
                        metadata_results.append({"error": str(e)})
                    pbar.update(1)

        # Phase 2: Content analysis (CPU-bound, parallel)
        if self.content_analyzer:
            print("\nPhase 2/3: Analyzing content (quality, blur, colors)...")
            content_results = []
            with ThreadPoolExecutor(max_workers=self.config.get("processing.max_workers", 4)) as executor:
                futures = {executor.submit(self.content_analyzer.analyze, path): path for path in image_paths}

                with tqdm(total=len(image_paths), desc="Content", unit="img", ncols=100) as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            content_results.append(result)
                            # Show quality score if available
                            if "quality_score" in result:
                                pbar.set_postfix_str(f"Quality: {result['quality_score']:.1f}/100")
                        except Exception as e:
                            content_results.append({"error": str(e)})
                        pbar.update(1)
        else:
            content_results = [{}] * len(image_paths)

        # Phase 3: ML analysis (GPU-accelerated batch processing)
        if self.ml_analyzer:
            device_type = "GPU" if self.ml_analyzer.device.type == "cuda" else "CPU"
            print(f"\nPhase 3/3: ML Analysis ({device_type}-accelerated scene/object detection)...")

            # Start GPU monitoring for non-batch mode
            monitor = get_monitor()
            if not use_batch and self.ml_analyzer.device.type == "cuda":
                monitor.start()

            if use_batch:
                ml_results = self.ml_analyzer.analyze_batch(image_paths, show_progress=True)
            else:
                ml_results = []
                with tqdm(total=len(image_paths), desc="ML Analysis", unit="img", ncols=120,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                    for path in image_paths:
                        result = self.ml_analyzer.analyze(path)
                        ml_results.append(result)

                        # Build status string
                        status_parts = []
                        if "primary_scene" in result:
                            status_parts.append(f"Scene: {result['primary_scene']}")
                        if self.ml_analyzer.device.type == "cuda":
                            status_parts.append(monitor.get_status_string())

                        pbar.set_postfix_str(" | ".join(status_parts))
                        pbar.update(1)

                # Show final GPU stats
                if self.ml_analyzer.device.type == "cuda":
                    print()
                    monitor.print_stats()
                    monitor.stop()

            # Clear GPU cache
            self.ml_analyzer.clear_cache()
        else:
            ml_results = [{}] * len(image_paths)

        # Combine results
        print("\nCombining analysis results...")
        for metadata, content, ml in zip(metadata_results, content_results, ml_results):
            combined = {**metadata, **content, **ml}
            results.append(combined)

        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"✓ Analysis Complete: {len(results)} images processed")

        # Calculate statistics
        if results:
            errors = sum(1 for r in results if "error" in r)
            if errors:
                print(f"  ⚠ Errors: {errors} images")

            # Quality stats
            if content_results and any("quality_score" in r for r in content_results):
                quality_scores = [r["quality_score"] for r in content_results if "quality_score" in r]
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"  📊 Average quality: {avg_quality:.1f}/100")

            # Scene stats
            if ml_results and any("primary_scene" in r for r in ml_results):
                scenes = {}
                for r in ml_results:
                    if "primary_scene" in r:
                        scene = r["primary_scene"]
                        scenes[scene] = scenes.get(scene, 0) + 1
                top_scenes = sorted(scenes.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  🎬 Top scenes: {', '.join(f'{s} ({c})' for s, c in top_scenes)}")

        print(f"{'='*60}\n")
        return results

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
