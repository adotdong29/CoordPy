"""Out-of-tree CoordPy plugin — a markdown ReportSink exemplar."""

from .sink import MarkdownReportSink, register

__all__ = ["MarkdownReportSink", "register"]
__version__ = "0.1.0"
