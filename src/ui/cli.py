"""Command-line interface for Image Engine."""

import click
import sys
from pathlib import Path

from ..core.engine import ImageEngine
from ..core.utils.config import Config


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Image Engine - Intelligent photo management system.

    Analyze, rename, deduplicate, and reformat photos with ML-powered insights.
    """
    pass


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--analyze/--no-analyze', default=True, help='Run analysis')
@click.option('--dedupe/--no-dedupe', default=False, help='Remove duplicates')
@click.option('--rename/--no-rename', default=False, help='Rename files')
@click.option('--embed-tags/--no-embed-tags', default=False, help='Embed tags in files')
@click.option('--convert/--no-convert', default=False, help='Convert formats')
@click.option('--dry-run/--execute', default=True, help='Preview changes (default)')
@click.option('--config', type=click.Path(), help='Configuration file path')
def process(directory, output, analyze, dedupe, rename, embed_tags, convert, dry_run, config):
    """
    Process photos with full pipeline.

    Examples:

    \b
    # Analyze photos and preview all operations
    python -m src.cli process ./photos --analyze --rename --convert --dry-run

    \b
    # Execute full pipeline (analyze, dedupe, rename, tag, convert)
    python -m src.cli process ./photos --analyze --dedupe --rename --embed-tags --convert --execute

    \b
    # Just analyze and generate report
    python -m src.cli process ./photos --analyze

    \b
    # Deduplicate and convert formats
    python -m src.cli process ./photos --dedupe --convert --execute -o ./processed
    """
    click.echo(click.style("\n=== Image Engine ===\n", fg="cyan", bold=True))

    try:
        # Initialize engine
        engine = ImageEngine(config_path=config)

        # Show system info
        if analyze:
            info = engine.get_system_info()
            click.echo(f"Device: {info.get('ml_info', {}).get('device', 'CPU only')}")
            if 'gpu_memory' in info:
                gpu_mem = info['gpu_memory']
                click.echo(f"GPU Memory: {gpu_mem.get('gpu_allocated_gb', 0):.2f} GB allocated")
            click.echo()

        # Run pipeline
        results = engine.process_pipeline(
            directory=directory,
            output_dir=output,
            analyze=analyze,
            dedupe=dedupe,
            rename=rename,
            embed_tags=embed_tags,
            convert_format=convert,
            dry_run=dry_run,
        )

        # Summary
        click.echo(click.style("\n=== Summary ===\n", fg="green", bold=True))
        click.echo(f"Total images: {results.get('total_images', 0)}")

        if 'analysis_results' in results:
            click.echo(f"Analyzed: {len(results['analysis_results'])} images")

        if 'duplicates' in results and results['duplicates']:
            click.echo(f"Duplicate sets: {len(results['duplicates'])}")

        if 'converted_files' in results:
            click.echo(f"Converted: {results['converted_files']} files")

        if 'renamed_files' in results:
            click.echo(f"Renamed: {results['renamed_files']} files")

        if 'tagged_files' in results:
            click.echo(f"Tagged: {results['tagged_files']} files")

        if dry_run:
            click.echo(click.style("\n[DRY RUN] No changes were made.", fg="yellow", bold=True))
            click.echo("Run with --execute to apply changes.")

        click.echo()

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        if '--debug' in sys.argv:
            raise
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file')
@click.option('--ml/--no-ml', default=True, help='Include ML analysis')
@click.option('--config', type=click.Path(), help='Configuration file path')
def analyze(directory, output, ml, config):
    """
    Analyze photos and generate report.

    Examples:

    \b
    # Analyze with ML
    python -m src.cli analyze ./photos

    \b
    # Analyze without ML (faster)
    python -m src.cli analyze ./photos --no-ml

    \b
    # Save report to custom location
    python -m src.cli analyze ./photos -o analysis_report.json
    """
    click.echo(click.style("\n=== Analyzing Photos ===\n", fg="cyan", bold=True))

    try:
        # Temporarily disable ML if requested
        if not ml and config is None:
            # Create temporary config
            temp_config = Config()
            temp_config.set('analysis.ml_analysis', False)
            engine = ImageEngine()
            engine.config = temp_config
            engine._init_analyzers()
        else:
            engine = ImageEngine(config_path=config)

        # Scan and analyze
        image_paths = engine.scan_directory(directory)
        results = engine.analyze_images(image_paths)

        # Save report
        output_path = output or "analysis_report.json"
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        click.echo(click.style(f"\n✓ Analysis complete!", fg="green", bold=True))
        click.echo(f"Report saved: {output_path}")
        click.echo(f"Images analyzed: {len(results)}")

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--dry-run/--execute', default=True, help='Preview duplicates')
@click.option('--strategy', type=click.Choice(['highest_quality', 'oldest', 'newest', 'largest']),
              default='highest_quality', help='Which duplicate to keep')
@click.option('--config', type=click.Path(), help='Configuration file path')
def dedupe(directory, dry_run, strategy, config):
    """
    Find and remove duplicate photos.

    Examples:

    \b
    # Find duplicates (preview only)
    python -m src.cli dedupe ./photos

    \b
    # Remove duplicates, keeping highest quality
    python -m src.cli dedupe ./photos --execute --strategy highest_quality

    \b
    # Remove duplicates, keeping oldest file
    python -m src.cli dedupe ./photos --execute --strategy oldest
    """
    click.echo(click.style("\n=== Deduplication ===\n", fg="cyan", bold=True))

    try:
        engine = ImageEngine(config_path=config)

        # Scan
        image_paths = engine.scan_directory(directory)

        # Find duplicates
        duplicates = engine.deduplicator.find_duplicates(image_paths)

        if not duplicates:
            click.echo(click.style("✓ No duplicates found!", fg="green", bold=True))
            return

        # Remove duplicates
        stats = engine.deduplicator.remove_duplicates(
            duplicates,
            strategy=strategy,
            dry_run=dry_run,
        )

        if dry_run:
            click.echo(click.style("\n[DRY RUN] Preview of duplicates:", fg="yellow"))
            click.echo(f"Files to delete: {len(stats['files_to_delete'])}")
            click.echo(f"Space to save: {stats['space_saved'] / 1_000_000:.2f} MB")
            click.echo("\nRun with --execute to delete duplicates.")
        else:
            click.echo(click.style("\n✓ Duplicates removed!", fg="green", bold=True))
            click.echo(f"Space saved: {stats['space_saved'] / 1_000_000:.2f} MB")

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--pattern', default='{datetime}', help='Naming pattern')
@click.option('--dry-run/--execute', default=True, help='Preview renames')
@click.option('--config', type=click.Path(), help='Configuration file path')
def rename(directory, output, pattern, dry_run, config):
    """
    Rename photos with date-based patterns.

    Available pattern variables:
    - {date}: Date only (YYYY-MM-DD)
    - {time}: Time only (HHMMSS)
    - {datetime}: Date and time (YYYY-MM-DD_HHMMSS)
    - {camera}: Camera model
    - {tags}: ML-detected tags
    - {seq}: Sequence number

    Examples:

    \b
    # Preview rename with date-time pattern
    python -m src.cli rename ./photos --pattern "{datetime}"

    \b
    # Rename with camera and date
    python -m src.cli rename ./photos --pattern "{date}_{camera}" --execute

    \b
    # Rename and move to output directory
    python -m src.cli rename ./photos -o ./renamed --execute
    """
    click.echo(click.style("\n=== Renaming Files ===\n", fg="cyan", bold=True))

    try:
        engine = ImageEngine(config_path=config)

        # Analyze to get metadata for renaming
        image_paths = engine.scan_directory(directory)
        analysis_results = engine.analyze_images(image_paths, use_batch=False)

        # Update pattern
        engine.renamer.pattern = pattern

        # Rename
        rename_map = engine.renamer.rename_files(
            analysis_results,
            output_dir=output,
            dry_run=dry_run,
        )

        if not dry_run:
            click.echo(click.style(f"\n✓ Renamed {len(rename_map)} files!", fg="green", bold=True))

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--format', type=click.Choice(['jpg', 'png', 'auto']), default='auto',
              help='Target format (auto chooses best)')
@click.option('--quality', type=int, default=95, help='JPEG quality (1-100)')
@click.option('--config', type=click.Path(), help='Configuration file path')
def convert(directory, output, format, quality, config):
    """
    Convert images to standardized formats.

    Examples:

    \b
    # Convert to auto-detected optimal formats
    python -m src.cli convert ./photos -o ./converted

    \b
    # Force convert all to JPEG
    python -m src.cli convert ./photos --format jpg --quality 95

    \b
    # Convert to PNG
    python -m src.cli convert ./photos --format png
    """
    click.echo(click.style("\n=== Format Conversion ===\n", fg="cyan", bold=True))

    try:
        engine = ImageEngine(config_path=config)
        engine.formatter.jpeg_quality = quality

        # Scan
        image_paths = engine.scan_directory(directory)

        # Show statistics
        stats = engine.formatter.get_format_stats(image_paths)
        click.echo("Current format distribution:")
        for fmt, count in sorted(stats["by_format"].items()):
            click.echo(f"  {fmt}: {count} files")
        click.echo(f"\nFiles needing conversion: {stats['needs_conversion']}")

        # Estimate impact
        impact = engine.formatter.estimate_space_impact(image_paths)
        click.echo(f"\nEstimated space impact:")
        click.echo(f"  Current: {impact['current_mb']:.2f} MB")
        click.echo(f"  After conversion: {impact['estimated_mb']:.2f} MB")
        click.echo(f"  Savings: {impact['savings_mb']:.2f} MB ({impact['savings_percent']:.1f}%)")

        # Convert
        click.echo()
        force_format = None if format == 'auto' else format
        converted = engine.formatter.convert_batch(
            image_paths,
            output_dir=output,
            show_progress=True,
        )

        click.echo(click.style(f"\n✓ Converted {len(converted)} files!", fg="green", bold=True))

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', type=click.Path(), help='Configuration file path')
def info(config):
    """Show system information and GPU status."""
    try:
        engine = ImageEngine(config_path=config)
        info = engine.get_system_info()

        click.echo(click.style("\n=== System Information ===\n", fg="cyan", bold=True))
        click.echo(f"Config: {info['config_path']}")
        click.echo(f"\nEnabled Analyzers:")
        click.echo(f"  Metadata: {info['analyzers']['metadata']}")
        click.echo(f"  Content: {info['analyzers']['content']}")
        click.echo(f"  ML/AI: {info['analyzers']['ml']}")

        if 'ml_info' in info:
            click.echo(f"\nML Configuration:")
            click.echo(f"  Device: {info['ml_info']['device']}")
            click.echo(f"  Batch size: {info['ml_info']['batch_size']}")

        if 'gpu_memory' in info:
            gpu = info['gpu_memory']
            click.echo(f"\nGPU Memory:")
            click.echo(f"  Allocated: {gpu.get('gpu_allocated_gb', 0):.2f} GB")
            click.echo(f"  Reserved: {gpu.get('gpu_reserved_gb', 0):.2f} GB")
            click.echo(f"  Peak: {gpu.get('gpu_max_allocated_gb', 0):.2f} GB")

        click.echo()

    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
