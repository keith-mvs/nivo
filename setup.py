"""Setup script for Image Engine."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="nivo",
    version="1.0.0",
    description="Intelligent photo management system with ML-powered analysis, deduplication, and auto-tagging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Image Engine Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "Pillow>=10.0.0",
        "piexif>=1.1.3",
        "opencv-python>=4.8.0",
        "imagehash>=4.3.1",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "iptcinfo3>=2.1.4",
        "click>=8.1.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
        "pillow-heif>=0.13.0",
        "rawpy>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nivo=src.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
