"""Image analysis modules."""

# Register HEIF/HEIC opener for PIL globally
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed
