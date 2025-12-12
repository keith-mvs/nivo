"""Analysis pipeline orchestrating metadata, content, and ML analysis phases."""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tqdm import tqdm

from ..utils.config import Config
from ..utils.gpu_monitor import get_monitor
from ..utils.progress_reporter import ProgressReporter, format_analysis_stats
from ..analyzers.metadata import MetadataExtractor
from ..analyzers.content import ContentAnalyzer


class AnalysisPipeline:
    """
    Orchestrates 3-phase image analysis pipeline:

    Phase 1: Metadata extraction (CPU-parallel)
    Phase 2: Content analysis (CPU-parallel)
    Phase 3: ML vision analysis (GPU-batched)
    """

    def __init__(
        self,
        metadata_extractor: MetadataExtractor,
        content_analyzer: Optional[ContentAnalyzer],
        ml_analyzer: Optional[Any],  # MLAnalyzer type
        config: Config,
    ):
        """
        Initialize analysis pipeline.

        Args:
            metadata_extractor: Metadata extraction analyzer
            content_analyzer: Content analysis analyzer (or None if disabled)
            ml_analyzer: ML vision analyzer (or None if disabled)
            config: Configuration object
        """
        self.metadata_extractor = metadata_extractor
        self.content_analyzer = content_analyzer
        self.ml_analyzer = ml_analyzer
        self.config = config

    def run(
        self,
        image_paths: List[str],
        use_batch: bool = True,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute 3-phase analysis pipeline.

        Args:
            image_paths: List of image file paths
            use_batch: Use batch processing for ML analysis
            show_progress: Show progress bars

        Returns:
            List of combined analysis results
        """
        if show_progress:
            print(f"\n=== Analyzing {len(image_paths)} images ===")
            self._print_gpu_info()
            print()

        # Execute analysis phases
        metadata_results = self._phase1_metadata(image_paths, show_progress)
        content_results = self._phase2_content(image_paths, show_progress)
        ml_results = self._phase3_ml(image_paths, use_batch, show_progress)

        # Combine results
        if show_progress:
            print("\nCombining analysis results...")

        results = []
        for metadata, content, ml in zip(metadata_results, content_results, ml_results):
            combined = {**metadata, **content, **ml}
            results.append(combined)

        # Print summary
        if show_progress:
            self._print_summary(results, content_results, ml_results)

        return results

    def _print_gpu_info(self):
        """Print GPU information if ML analysis is enabled."""
        if self.ml_analyzer and hasattr(self.ml_analyzer, 'device'):
            if self.ml_analyzer.device.type == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def _phase1_metadata(
        self,
        image_paths: List[str],
        show_progress: bool
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Extract metadata (CPU-parallel).

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of metadata dictionaries
        """
        if show_progress:
            print("Phase 1/3: Extracting metadata...")

        results = []
        max_workers = self.config.get("processing.max_workers", 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.metadata_extractor.extract, path): path
                for path in image_paths
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(image_paths),
                    desc="Metadata",
                    unit="img",
                    ncols=100
                )

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)

                    # Show current file being processed
                    if show_progress and "file_name" in result:
                        if hasattr(iterator, 'set_postfix_str'):
                            iterator.set_postfix_str(f"Current: {result['file_name'][:40]}")
                except Exception as e:
                    results.append({"error": str(e)})

        return results

    def _phase2_content(
        self,
        image_paths: List[str],
        show_progress: bool
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Analyze content (CPU-parallel).

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of content analysis dictionaries
        """
        if not self.content_analyzer:
            return [{}] * len(image_paths)

        if show_progress:
            print("\nPhase 2/3: Analyzing content (quality, blur, colors)...")

        results = []
        max_workers = self.config.get("processing.max_workers", 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.content_analyzer.analyze, path): path
                for path in image_paths
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(image_paths),
                    desc="Content",
                    unit="img",
                    ncols=100
                )

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)

                    # Show quality score if available
                    if show_progress and "quality_score" in result:
                        if hasattr(iterator, 'set_postfix_str'):
                            iterator.set_postfix_str(f"Quality: {result['quality_score']:.1f}/100")
                except Exception as e:
                    results.append({"error": str(e)})

        return results

    def _phase3_ml(
        self,
        image_paths: List[str],
        use_batch: bool,
        show_progress: bool
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: ML vision analysis (GPU-batched).

        Args:
            image_paths: List of image file paths
            use_batch: Use batch processing
            show_progress: Show progress bar

        Returns:
            List of ML analysis dictionaries
        """
        if not self.ml_analyzer:
            return [{}] * len(image_paths)

        device_type = "GPU" if self.ml_analyzer.device.type == "cuda" else "CPU"
        if show_progress:
            print(f"\nPhase 3/3: ML Analysis ({device_type}-accelerated scene/object detection)...")

        # Batch mode (recommended)
        if use_batch:
            ml_results = self.ml_analyzer.analyze_batch(image_paths, show_progress=show_progress)
        else:
            # Single-image mode with progress
            ml_results = self._ml_single_mode(image_paths, show_progress)

        # Clear GPU cache
        if hasattr(self.ml_analyzer, 'clear_cache'):
            self.ml_analyzer.clear_cache()

        return ml_results

    def _ml_single_mode(
        self,
        image_paths: List[str],
        show_progress: bool
    ) -> List[Dict[str, Any]]:
        """
        Run ML analysis in single-image mode (slower, for debugging).

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            List of ML analysis results
        """
        results = []
        monitor = get_monitor()

        # Start GPU monitoring
        if self.ml_analyzer.device.type == "cuda":
            monitor.start()

        iterator = image_paths
        if show_progress:
            iterator = tqdm(
                image_paths,
                desc="ML Analysis",
                unit="img",
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )

        for path in iterator:
            result = self.ml_analyzer.analyze(path)
            results.append(result)

            # Build status string
            if show_progress and hasattr(iterator, 'set_postfix_str'):
                status_parts = []
                if "primary_scene" in result:
                    status_parts.append(f"Scene: {result['primary_scene']}")
                if self.ml_analyzer.device.type == "cuda":
                    status_parts.append(monitor.get_status_string())

                iterator.set_postfix_str(" | ".join(status_parts))

        # Show final GPU stats
        if show_progress and self.ml_analyzer.device.type == "cuda":
            print()
            monitor.print_stats()
            monitor.stop()

        return results

    def _print_summary(
        self,
        results: List[Dict[str, Any]],
        content_results: List[Dict[str, Any]],
        ml_results: List[Dict[str, Any]]
    ):
        """
        Print summary statistics.

        Args:
            results: Combined results
            content_results: Content analysis results
            ml_results: ML analysis results
        """
        print(f"\n{'='*60}")
        print(f"✓ Analysis Complete: {len(results)} images processed")

        # Error statistics
        errors = sum(1 for r in results if "error" in r)
        if errors:
            print(f"  ⚠ Errors: {errors} images")

        # Quality statistics
        if content_results and any("quality_score" in r for r in content_results):
            quality_scores = [r["quality_score"] for r in content_results if "quality_score" in r]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"  📊 Average quality: {avg_quality:.1f}/100")

        # Scene statistics
        if ml_results and any("primary_scene" in r for r in ml_results):
            scenes = {}
            for r in ml_results:
                if "primary_scene" in r:
                    scene = r["primary_scene"]
                    scenes[scene] = scenes.get(scene, 0) + 1

            top_scenes = sorted(scenes.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  🎬 Top scenes: {', '.join(f'{s} ({c})' for s, c in top_scenes)}")

        print(f"{'='*60}\n")
