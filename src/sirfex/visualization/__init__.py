"""Headless evidence queries and transformations for the optional Studio UI.

This package intentionally has no Streamlit or Plotly imports.  It reads the
same immutable artifacts used by the CLI and is safe to use in core-only
installations.
"""

from sirfex.visualization.readers import EvidenceReader, EvidenceReadError

__all__ = ["EvidenceReader", "EvidenceReadError"]
