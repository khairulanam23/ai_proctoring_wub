"""Physical paper and answer-sheet recognition module.

Performs geometric quadrilateral contour analysis, aspect ratio validation (A4/Letter),
desk contrast extraction, multi-sheet detection, and temporal tracking of paper manipulation
or lifting during handwritten examinations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class PaperState(str, Enum):
    """Observable physical paper status on candidate desk."""

    ABSENT = "paper_absent"
    PRESENT = "paper_present"
    MULTIPLE_SHEETS = "multiple_sheets"
    LARGE_MOVEMENT = "large_paper_movement"
    LIFTED_MANIPULATED = "paper_lifted_manipulated"


@dataclass
class PaperSheet:
    """Individual recognized sheet of paper or answer booklet."""

    quadrilateral: list[tuple[int, int]]  # 4 corner coordinates
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[int, int]
    area: float
    aspect_ratio: float
    confidence: float
    angle_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadrilateral": [list(pt) for pt in self.quadrilateral],
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "area": round(self.area, 1),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "confidence": round(self.confidence, 4),
            "angle_deg": round(self.angle_deg, 2),
        }


@dataclass
class PaperAnalysisResult:
    """Per-frame physical paper analysis outcome."""

    paper_present: bool = False
    paper_count: int = 0
    state: PaperState = PaperState.ABSENT
    sheets: list[PaperSheet] = field(default_factory=list)
    displacement_px: float = 0.0
    manipulation_detected: bool = False
    inference_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_present": self.paper_present,
            "paper_count": self.paper_count,
            "state": self.state.value,
            "sheets": [s.to_dict() for s in self.sheets],
            "displacement_px": round(self.displacement_px, 2),
            "manipulation_detected": self.manipulation_detected,
            "inference_ms": round(self.inference_ms, 2),
        }


class PaperDetector:
    """Detects and tracks paper sheets in the workspace / desk region."""

    def __init__(
        self,
        desk_top_ratio: float = 0.40,
        min_area_ratio: float = 0.03,
        max_area_ratio: float = 0.70,
        min_aspect_ratio: float = 1.15,
        max_aspect_ratio: float = 1.75,
    ) -> None:
        self.desk_top_ratio = float(desk_top_ratio)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

        self._last_centroid: tuple[int, int] | None = None
        self._last_angle: float | None = None
        self._last_area: float | None = None

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> PaperAnalysisResult:
        """Analyze frame for paper sheets and evaluate temporal stability."""
        t0 = time.perf_counter()
        result = PaperAnalysisResult()

        if frame is None or frame.size == 0:
            return result

        height, width = frame.shape[:2]
        desk_y_start = int(height * self.desk_top_ratio)
        desk_roi = frame[desk_y_start:, :]
        roi_h, roi_w = desk_roi.shape[:2]
        total_frame_area = float(width * height)

        # 1. Grayscale and edge detection
        gray = cv2.cvtColor(desk_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding and Canny to capture contrasting paper boundaries
        edges = cv2.Canny(blurred, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_sheets: list[PaperSheet] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / total_frame_area

            # Area filter
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue

            # Approximate polygonal curve
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

            # Must be a convex 4-vertex quadrilateral
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                # Map back to full frame coordinates
                full_pts = [(int(x), int(y + desk_y_start)) for x, y in pts]

                # Compute bounding box and aspect ratio
                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), angle = rect
                if rw == 0 or rh == 0:
                    continue
                long_side = max(rw, rh)
                short_side = min(rw, rh)
                aspect_ratio = long_side / short_side

                # Standard A4 is ~1.414, Letter is ~1.294
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    x1 = min(p[0] for p in full_pts)
                    y1 = min(p[1] for p in full_pts)
                    x2 = max(p[0] for p in full_pts)
                    y2 = max(p[1] for p in full_pts)

                    cx = int((x1 + x2) / 2.0)
                    cy = int((y1 + y2) / 2.0)

                    # Contrast check against local desk patch
                    # Papers are typically brighter than the surrounding desk
                    sheet_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(sheet_mask, [pts], -1, 255, -1)
                    mean_val = cv2.mean(gray, mask=sheet_mask)[0]
                    confidence = min(1.0, 0.5 + (mean_val / 510.0))

                    found_sheets.append(
                        PaperSheet(
                            quadrilateral=full_pts,
                            bbox=(x1, y1, x2, y2),
                            centroid=(cx, cy),
                            area=float(area),
                            aspect_ratio=aspect_ratio,
                            confidence=confidence,
                            angle_deg=float(angle),
                        )
                    )

        result.sheets = found_sheets
        result.paper_count = len(found_sheets)
        result.paper_present = len(found_sheets) > 0

        # Temporal analysis: check for manipulation or lifting
        if found_sheets:
            primary = found_sheets[0]
            cx, cy = primary.centroid
            angle = primary.angle_deg
            area = primary.area

            if self._last_centroid is not None:
                disp = float(np.hypot(cx - self._last_centroid[0], cy - self._last_centroid[1]))
                result.displacement_px = disp

                # Large horizontal movement (> 25% of frame width)
                if disp > (0.25 * width):
                    result.state = PaperState.LARGE_MOVEMENT
                    result.manipulation_detected = True
                # Lifted / tilted: significant vertical rise towards top of frame or angle flip
                elif (self._last_centroid[1] - cy) > (0.15 * height) or (
                    self._last_angle is not None and abs(angle - self._last_angle) > 35.0
                ):
                    result.state = PaperState.LIFTED_MANIPULATED
                    result.manipulation_detected = True
                elif len(found_sheets) > 1:
                    result.state = PaperState.MULTIPLE_SHEETS
                else:
                    result.state = PaperState.PRESENT
            else:
                result.state = (
                    PaperState.MULTIPLE_SHEETS if len(found_sheets) > 1 else PaperState.PRESENT
                )

            self._last_centroid = (cx, cy)
            self._last_angle = angle
            self._last_area = area
        else:
            result.state = PaperState.ABSENT
            self._last_centroid = None
            self._last_angle = None
            self._last_area = None

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def reset(self) -> None:
        """Reset temporal state between exam sessions."""
        self._last_centroid = None
        self._last_angle = None
        self._last_area = None
