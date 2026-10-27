"""
Video capture module for the Smart Surveillance System.

This module provides the VideoCapture class that interfaces with the webcam
hardware and provides frames to the processing pipeline.
"""

import cv2
import logging
from datetime import datetime
from typing import Optional, Tuple
import numpy as np

from .models import FrameMetadata
from .error_handler import ErrorHandler


class VideoCapture:
    """
    Interface with webcam hardware and provide frames to the processing pipeline.

    Uses DirectShow on Windows because it provides reliable webcam access
    on systems where the default OpenCV backend may produce black frames.
    """

    def __init__(
        self,
        camera_index: int = 0,
        logger: Optional[logging.Logger] = None
    ):
        self.camera_index = camera_index
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_number = 0
        self._is_opened = False

        # Initialize webcam
        self._initialize_capture()

    def _initialize_capture(self) -> None:
        """Initialize the webcam using DirectShow."""

        try:
            # IMPORTANT:
            # CAP_DSHOW fixes the black-frame problem on many Windows webcams.
            self._capture = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW
            )

            if self._capture is None or not self._capture.isOpened():
                self._is_opened = False

                self.logger.error(
                    f"Failed to open webcam at index {self.camera_index}"
                )
                return

            # Request a standard resolution.
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Keep only a small buffer so the application doesn't process
            # old frames.
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify that we can actually receive a frame.
            success, test_frame = self._capture.read()

            if not success or test_frame is None:
                self.logger.error(
                    "Webcam opened, but no frame could be received."
                )

                self._capture.release()
                self._capture = None
                self._is_opened = False
                return

            if not self._validate_frame(test_frame):
                self.logger.error(
                    "Webcam returned an invalid frame."
                )

                self._capture.release()
                self._capture = None
                self._is_opened = False
                return

            self._is_opened = True

            width = int(
                self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            )
            height = int(
                self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            fps = self._capture.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30.0

            self.logger.info(
                f"Webcam initialized successfully: "
                f"{width}x{height} @ {fps:.1f} FPS "
                f"(DirectShow)"
            )

        except Exception as e:
            self._is_opened = False
            self.logger.error(
                f"Exception during webcam initialization: {e}"
            )

    def is_opened(self) -> bool:
        """
        Check if webcam is open and ready.

        Returns:
            True if webcam is available.
        """

        return (
            self._is_opened
            and self._capture is not None
            and self._capture.isOpened()
        )

    def read_frame(
        self
    ) -> Tuple[
        bool,
        Optional[np.ndarray],
        Optional[FrameMetadata]
    ]:
        """
        Read one frame from the webcam.

        Returns:
            success, frame, metadata
        """

        if not self.is_opened():
            self.logger.warning(
                "Attempted to read frame from closed webcam"
            )
            return False, None, None

        try:
            success, frame = self._capture.read()

            if not success or frame is None:
                self.logger.warning(
                    f"Failed to read frame "
                    f"{self._frame_number + 1}"
                )

                # Try to reconnect.
                if self._attempt_reconnection():
                    success, frame = self._capture.read()

                if not success or frame is None:
                    return False, None, None

            # Validate frame.
            if not self._validate_frame(frame):
                self.logger.error(
                    f"Invalid frame format at frame "
                    f"{self._frame_number + 1}"
                )
                return False, None, None

            # Increment frame number.
            self._frame_number += 1

            # Create metadata.
            timestamp = datetime.now()

            height, width = frame.shape[:2]

            metadata = FrameMetadata(
                frame_number=self._frame_number,
                timestamp=timestamp,
                width=width,
                height=height
            )

            return True, frame, metadata

        except Exception as e:
            self.logger.error(
                f"Exception while reading frame: {e}"
            )
            return False, None, None

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate webcam frame.
        """

        if frame is None:
            return False

        if not isinstance(frame, np.ndarray):
            return False

        if len(frame.shape) != 3:
            return False

        if frame.shape[2] != 3:
            return False

        if frame.dtype != np.uint8:
            return False

        # Make sure frame isn't empty.
        if frame.size == 0:
            return False

        return True

    def _attempt_reconnection(self) -> bool:
        """
        Attempt to reconnect to webcam.
        """

        self.logger.info(
            "Attempting to reconnect to webcam..."
        )

        def reconnect_operation():

            # Release old camera.
            if self._capture is not None:
                self._capture.release()

            self._capture = None
            self._is_opened = False

            # Reopen using DirectShow.
            self._initialize_capture()

            if not self.is_opened():
                raise RuntimeError(
                    "Failed to reopen webcam"
                )

            return True

        result = self.error_handler.retry_with_backoff(
            reconnect_operation,
            "Webcam reconnection",
            max_retries=3
        )

        return result is not None

    def release(self) -> None:
        """
        Release webcam resource.

        This is important because it allows another application
        to access the webcam after SafeVision closes.
        """

        if self._capture is not None:

            try:
                self._capture.release()
            except Exception as e:
                self.logger.warning(
                    f"Error releasing webcam: {e}"
                )

            self._capture = None

        self._is_opened = False

        self.logger.info(
            "Webcam released"
        )

    def get_fps(self) -> float:
        """
        Get webcam FPS.
        """

        if not self.is_opened():
            return 0.0

        try:
            fps = self._capture.get(
                cv2.CAP_PROP_FPS
            )

            return fps if fps > 0 else 30.0

        except Exception as e:

            self.logger.warning(
                f"Failed to get FPS: {e}"
            )

            return 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get webcam resolution.

        Returns:
            (width, height)
        """

        if not self.is_opened():
            return (0, 0)

        try:

            width = int(
                self._capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                self._capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            return (width, height)

        except Exception as e:

            self.logger.warning(
                f"Failed to get resolution: {e}"
            )

            return (0, 0)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        """Ensure webcam is released."""
        self.release()
        return False