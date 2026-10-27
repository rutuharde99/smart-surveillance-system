"""
Display Module for SafeVision.

Handles:
- Live camera display
- Object detection bounding boxes
- Motion regions
- Alerts
- FPS
- Detection count
- People count
- Crowd status
"""

import cv2
import numpy as np
from typing import List, Optional

from .models import ProcessedFrame, Alert


class DisplayModule:
    """
    Displays the processed camera frame with SafeVision overlays.
    """

    # BGR colors
    COLOR_OBJECT = (0, 255, 0)       # Green
    COLOR_MOTION = (255, 0, 0)       # Blue
    COLOR_ALERT = (0, 0, 255)        # Red
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_STATUS_NORMAL = (0, 200, 0)
    COLOR_STATUS_CROWD = (0, 165, 255)
    COLOR_STATUS_HIGH = (0, 0, 255)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, window_name: str = "SafeVision"):
        """
        Initialize display module.
        """
        self.window_name = window_name
        self._window_created = False

    def render(
        self,
        processed_frame: ProcessedFrame,
        fps: float,
        alerts: Optional[List[Alert]] = None
    ) -> None:
        """
        Render one processed camera frame.
        """

        # ---------------------------------------------------------
        # Validate frame
        # ---------------------------------------------------------

        if processed_frame is None:
            return

        if processed_frame.original_frame is None:
            return

        # Make a copy so the original camera frame is not modified
        display_frame = processed_frame.original_frame.copy()

        # ---------------------------------------------------------
        # Draw motion regions
        # ---------------------------------------------------------

        for motion_region in processed_frame.motion_regions:

            self._draw_bounding_box(
                display_frame,
                motion_region.bbox,
                self.COLOR_MOTION,
                f"Motion ({motion_region.area}px)"
            )

        # ---------------------------------------------------------
        # Draw object detections
        # ---------------------------------------------------------

        for detection in processed_frame.object_detections:

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            self._draw_bounding_box(
                display_frame,
                detection.bbox,
                self.COLOR_OBJECT,
                label
            )

        # ---------------------------------------------------------
        # Draw alerts
        # ---------------------------------------------------------

        if alerts:
            self._draw_alerts_banner(
                display_frame,
                alerts
            )

        # ---------------------------------------------------------
        # Draw SafeVision information
        # ---------------------------------------------------------

        self._draw_information_panel(
            display_frame,
            fps=fps,
            detection_count=len(processed_frame.object_detections),
            person_count=processed_frame.person_count,
            crowd_status=processed_frame.crowd_status
        )

        # ---------------------------------------------------------
        # Create window
        # ---------------------------------------------------------

        if not self._window_created:

            cv2.namedWindow(
                self.window_name,
                cv2.WINDOW_NORMAL
            )

            # Give the window a reasonable starting size
            cv2.resizeWindow(
                self.window_name,
                960,
                540
            )

            self._window_created = True

        # ---------------------------------------------------------
        # Show camera frame
        # ---------------------------------------------------------

        cv2.imshow(
            self.window_name,
            display_frame
        )

    # =============================================================
    # BOUNDING BOX
    # =============================================================

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox,
        color: tuple,
        label: Optional[str] = None
    ) -> None:
        """
        Draw bounding box and optional label.
        """

        x1 = int(bbox.x)
        y1 = int(bbox.y)

        x2 = int(bbox.x + bbox.width)
        y2 = int(bbox.y + bbox.height)

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        if not label:
            return

        # Text settings
        font_scale = 0.5
        thickness = 1

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            self.FONT,
            font_scale,
            thickness
        )

        # Put label above box if possible
        label_y = y1 - 8

        if label_y < text_height + 5:
            label_y = y1 + text_height + 8

        # Label background
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 5),
            (
                x1 + text_width + 8,
                label_y + baseline
            ),
            color,
            -1
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 2),
            self.FONT,
            font_scale,
            self.COLOR_WHITE,
            thickness,
            cv2.LINE_AA
        )

    # =============================================================
    # ALERT BANNER
    # =============================================================

    def _draw_alerts_banner(
        self,
        frame: np.ndarray,
        alerts: List[Alert]
    ) -> None:
        """
        Draw alert banner at the top of the camera frame.
        """

        if not alerts:
            return

        height, width = frame.shape[:2]

        banner_height = 45

        # Red banner
        cv2.rectangle(
            frame,
            (0, 0),
            (width, banner_height),
            self.COLOR_ALERT,
            -1
        )

        # Alert messages
        alert_text = " | ".join(
            alert.message
            for alert in alerts
        )

        text = f"ALERT: {alert_text}"

        cv2.putText(
            frame,
            text,
            (10, 30),
            self.FONT,
            0.6,
            self.COLOR_WHITE,
            2,
            cv2.LINE_AA
        )

    # =============================================================
    # INFORMATION PANEL
    # =============================================================

    def _draw_information_panel(
        self,
        frame: np.ndarray,
        fps: float,
        detection_count: int,
        person_count: int,
        crowd_status: str
    ) -> None:
        """
        Draw FPS, detections, people count and status.
        """

        # ---------------------------------------------------------
        # Information
        # ---------------------------------------------------------

        status = str(crowd_status).upper()

        lines = [
            f"FPS: {fps:.1f}",
            f"Detections: {detection_count}",
            f"People: {person_count}",
            f"Status: {status}"
        ]

        # ---------------------------------------------------------
        # Panel size
        # ---------------------------------------------------------

        x = 10
        y = 25

        line_height = 30

        panel_width = 220
        panel_height = 145

        # Semi-transparent black panel
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (5, 5),
            (panel_width, panel_height),
            self.COLOR_BLACK,
            -1
        )

        cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0,
            frame
        )

        # ---------------------------------------------------------
        # Draw information
        # ---------------------------------------------------------

        for index, line in enumerate(lines):

            current_y = y + (index * line_height)

            # Status color
            if line.startswith("Status:"):

                if status == "HIGH":
                    text_color = self.COLOR_STATUS_HIGH

                elif status in ("CROWD", "WARNING"):
                    text_color = self.COLOR_STATUS_CROWD

                else:
                    text_color = self.COLOR_STATUS_NORMAL

            else:
                text_color = self.COLOR_WHITE

            cv2.putText(
                frame,
                line,
                (x, current_y),
                self.FONT,
                0.6,
                text_color,
                2,
                cv2.LINE_AA
            )

    # =============================================================
    # CLOSE
    # =============================================================

    def close(self) -> None:
        """
        Close the OpenCV display window.
        """

        if self._window_created:

            cv2.destroyWindow(
                self.window_name
            )

            self._window_created = False

    # =============================================================
    # CONTEXT MANAGER
    # =============================================================

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        self.close()
        return False