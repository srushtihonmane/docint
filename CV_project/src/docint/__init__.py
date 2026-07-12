"""docint — Document Intelligence Pipeline.

Turns phone photos of documents (skewed, shadowed, low-light, printed or
handwritten, English / Marathi / Hindi) into clean structured output:
JSON regions + parsed questions + searchable full text, plus a deskewed
image.

Stages (one module each, independently callable):

    preprocess -> detect -> recognize -> layout -> pipeline (orchestration)

This package intentionally imports nothing at top level so that
``import docint`` stays cheap and dependency-free; import the stage modules
directly (e.g. ``from docint.pipeline import Pipeline``).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
