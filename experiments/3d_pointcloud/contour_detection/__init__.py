"""Contour detection methods for GNG-DT Robot.

This module provides two contour detection algorithms:
- TodaContourDetector: Angle gap threshold method (Toda et al., 2021)
- FurutaContourDetector: CCW traversal method (Furuta et al., FSS2022)
"""

from .toda_method import TodaContourDetector
from .furuta_method import FurutaContourDetector, ContourResult

__all__ = ["TodaContourDetector", "FurutaContourDetector", "ContourResult"]
