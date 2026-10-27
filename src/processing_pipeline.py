"""
Processing Pipeline module for the Smart Surveillance System.

This module orchestrates the frame processing workflow, coordinating
between video capture, object detection, motion detection, crowd detection,
and output systems.
"""

import time
import logging
from typing import Optional

import numpy as np
import cv2

from .models import ProcessedFrame, FrameMetadata, Configuration
from .object_detector import ObjectDetector
from .motion_detector import MotionDetector
from .crowd_detector import CrowdDetector
from .error_handler import ErrorHandler


class PipelineStatistics:
    """Statistics for pipeline processing."""

    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.total_motion_regions = 0
        self.total_processing_time_ms = 0.0
        self.failed_frames = 0

    def get_average_processing_time(self) -> float:
        """Get average processing time per frame in milliseconds."""
        if self.total_frames == 0:
            return 0.0

        return self.total_processing_time_ms / self.total_frames


class ProcessingPipeline:
    """
    Orchestrates the frame processing workflow.

    The pipeline processes frames through sequential stages:
    1. Preprocessing
    2. Object detection using YOLO
    3. Motion detection
    4. Crowd detection
    5. Post-processing
    """

    def __init__(
        self,
        config: Configuration,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration object with all system settings.
            logger: Logger instance for error reporting.
        """

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        self.statistics = PipelineStatistics()

        # ---------------------------------------------------------
        # Initialize Object Detector
        # ---------------------------------------------------------
        try:
            self.object_detector = ObjectDetector(
                model_path=config.model_path,
                confidence_threshold=config.detection_threshold,
                target_classes=config.target_classes
            )

            self.logger.info(
                f"Object detector initialized with model: "
                f"{config.model_path}"
            )

        except Exception as e:
            self.error_handler.handle_initialization_error(
                e,
                "ObjectDetector"
            )

        # ---------------------------------------------------------
        # Initialize Motion Detector
        # ---------------------------------------------------------
        try:
            self.motion_detector = MotionDetector(
                sensitivity=config.motion_sensitivity,
                min_area=config.motion_min_area
            )

            self.logger.info(
                f"Motion detector initialized with sensitivity: "
                f"{config.motion_sensitivity}"
            )

        except Exception as e:
            self.error_handler.handle_initialization_error(
                e,
                "MotionDetector"
            )

        # ---------------------------------------------------------
        # Initialize Crowd Detector
        # ---------------------------------------------------------
        try:
            self.crowd_detector = CrowdDetector(
                crowd_threshold=5,
                high_crowd_threshold=10
            )

            self.logger.info(
                "Crowd detector initialized successfully"
            )

        except Exception as e:
            self.error_handler.handle_initialization_error(
                e,
                "CrowdDetector"
            )

    def process_frame(
        self,
        frame: np.ndarray,
        metadata: FrameMetadata
    ) -> Optional[ProcessedFrame]:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame from webcam.
            metadata: Frame timestamp and number.

        Returns:
            ProcessedFrame containing detection, motion,
            and crowd analysis results.
        """

        start_time = time.time()

        try:

            # -----------------------------------------------------
            # 1. Preprocess Frame
            # -----------------------------------------------------
            processed_frame = self._preprocess_frame(frame)

            # -----------------------------------------------------
            # 2. Object Detection
            # -----------------------------------------------------
            try:

                object_detections = self.object_detector.detect(
                    processed_frame
                )

            except TimeoutError as e:

                self.logger.warning(
                    f"Object detection timeout: {e}"
                )

                object_detections = []

            except Exception as e:

                self.error_handler.handle_processing_error(
                    e,
                    metadata.frame_number
                )

                object_detections = []

            # -----------------------------------------------------
            # 3. Motion Detection
            # -----------------------------------------------------
            try:

                motion_regions = self.motion_detector.detect_motion(
                    processed_frame
                )

            except Exception as e:

                self.error_handler.handle_processing_error(
                    e,
                    metadata.frame_number
                )

                motion_regions = []

            # -----------------------------------------------------
            # 4. Crowd Detection
            # -----------------------------------------------------
            try:

                crowd_analysis = self.crowd_detector.analyze(
                    object_detections
                )

            except Exception as e:

                self.error_handler.handle_processing_error(
                    e,
                    metadata.frame_number
                )

                crowd_analysis = {
                    "person_count": 0,
                    "status": "NORMAL",
                    "crowd_detected": False,
                    "high_crowd": False
                }

            # -----------------------------------------------------
            # 5. Calculate Processing Time
            # -----------------------------------------------------
            processing_time_ms = (
                time.time() - start_time
            ) * 1000

            # -----------------------------------------------------
            # 6. Update Statistics
            # -----------------------------------------------------
            self.statistics.total_frames += 1

            self.statistics.total_detections += (
                len(object_detections)
            )

            self.statistics.total_motion_regions += (
                len(motion_regions)
            )

            self.statistics.total_processing_time_ms += (
                processing_time_ms
            )

            # -----------------------------------------------------
            # 7. Create ProcessedFrame
            # -----------------------------------------------------
            result = ProcessedFrame(
                original_frame=frame,
                metadata=metadata,
                object_detections=object_detections,
                motion_regions=motion_regions,
                processing_time_ms=processing_time_ms,

                # SafeVision crowd information
                person_count=crowd_analysis["person_count"],
                crowd_status=crowd_analysis["status"],
                crowd_detected=crowd_analysis["crowd_detected"],
                high_crowd=crowd_analysis["high_crowd"]
            )

            return result

        except Exception as e:

            self.error_handler.handle_processing_error(
                e,
                metadata.frame_number
            )

            self.statistics.failed_frames += 1

            return None

    def _preprocess_frame(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Apply optional preprocessing to the frame.

        Args:
            frame: Input frame in BGR format.

        Returns:
            Preprocessed frame.
        """

        if not self.config.preprocessing_enabled:
            return frame

        processed = frame.copy()

        # Optional resize for performance
        if self.config.resize_width is not None:

            height, width = processed.shape[:2]

            aspect_ratio = height / width

            new_width = self.config.resize_width

            new_height = int(
                new_width * aspect_ratio
            )

            processed = cv2.resize(
                processed,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )

            self.logger.debug(
                f"Resized frame from "
                f"{width}x{height} to "
                f"{new_width}x{new_height}"
            )

        return processed

    def get_statistics(self) -> PipelineStatistics:
        """
        Get processing statistics.

        Returns:
            PipelineStatistics object with cumulative metrics.
        """

        return self.statistics

    def reset_statistics(self):
        """Reset all statistics counters."""

        self.statistics = PipelineStatistics()

        self.logger.info(
            "Pipeline statistics reset"
        )