"""
Main Surveillance System module for SafeVision.

This module coordinates:
- Webcam capture
- YOLO object detection
- Motion detection
- Crowd detection
- Alerts
- Snapshots
- Event logging
- Performance monitoring
- Display
"""

import cv2
import logging
import time
from typing import Optional

from .models import Configuration, Alert, Detection, BoundingBox
from .video_capture import VideoCapture
from .processing_pipeline import ProcessingPipeline
from .display_module import DisplayModule
from .event_logger import EventLogger
from .snapshot_saver import SnapshotSaver
from .alert_generator import AlertGenerator
from .performance_monitor import PerformanceMonitor
from .error_handler import ErrorHandler


class SurveillanceSystem:
    """
    Main SafeVision surveillance system.
    """

    # Keyboard controls
    KEY_QUIT = ord("q")
    KEY_PAUSE = ord("p")
    KEY_RESUME = ord("r")
    KEY_SNAPSHOT = ord("s")

    # Performance logging interval
    METRICS_LOG_INTERVAL = 60

    def __init__(
        self,
        config: Configuration,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SafeVision.
        """

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)

        self._running = False
        self._paused = False
        self._last_metrics_log_time = 0.0

        self.logger.info(
            "Initializing SafeVision..."
        )

        try:

            # -------------------------------------------------
            # Video Capture
            # -------------------------------------------------
            self.video_capture = VideoCapture(
                camera_index=0,
                logger=self.logger
            )

            if not self.video_capture.is_opened():
                raise RuntimeError(
                    "Failed to open webcam"
                )

            # -------------------------------------------------
            # Processing Pipeline
            # -------------------------------------------------
            self.pipeline = ProcessingPipeline(
                config,
                logger=self.logger
            )

            # -------------------------------------------------
            # Display
            # -------------------------------------------------
            if config.display_enabled:

                self.display = DisplayModule(
                    window_name=config.window_name
                )

            else:

                self.display = None

            # -------------------------------------------------
            # Event Logger
            # -------------------------------------------------
            self.event_logger = EventLogger(
                log_directory=config.log_directory,
                max_file_size_mb=config.log_max_size_mb,
                logger=self.logger
            )

            # -------------------------------------------------
            # Snapshot Saver
            # -------------------------------------------------
            self.snapshot_saver = SnapshotSaver(
                output_directory=config.snapshot_directory,
                enabled=config.snapshot_enabled,
                rate_limit_seconds=config.snapshot_rate_limit,
                logger=self.logger
            )

            # -------------------------------------------------
            # Alert Generator
            # -------------------------------------------------
            self.alert_generator = AlertGenerator(
                alert_classes=config.alert_classes,
                cooldown_seconds=config.alert_cooldown
            )

            # -------------------------------------------------
            # Performance Monitor
            # -------------------------------------------------
            self.performance_monitor = PerformanceMonitor(
                logger=self.logger
            )

            self.logger.info(
                "SafeVision initialized successfully"
            )

        except Exception as e:

            self.error_handler.handle_initialization_error(
                e,
                "SafeVision"
            )

    def run(self):
        """
        Run the SafeVision processing loop.
        """

        self._running = True

        self.logger.info(
            "Starting SafeVision..."
        )

        self.logger.info(
            "Controls: Q=Quit, P=Pause, R=Resume, S=Manual Snapshot"
        )

        try:

            while self._running:

                # -------------------------------------------------
                # Pause handling
                # -------------------------------------------------
                if self._paused:

                    self._handle_paused_state()

                    continue

                # -------------------------------------------------
                # Capture frame
                # -------------------------------------------------
                success, frame, metadata = (
                    self.video_capture.read_frame()
                )

                if not success or frame is None:

                    self.logger.warning(
                        "Failed to capture frame, retrying..."
                    )

                    time.sleep(0.1)

                    continue

                # -------------------------------------------------
                # Process frame
                # -------------------------------------------------
                processed_frame = (
                    self.pipeline.process_frame(
                        frame,
                        metadata
                    )
                )

                if processed_frame is None:

                    self.logger.warning(
                        f"Failed to process frame "
                        f"{metadata.frame_number}"
                    )

                    continue

                # -------------------------------------------------
                # Normal object alerts
                # -------------------------------------------------
                alerts = self.alert_generator.check_alerts(
                    processed_frame.object_detections
                )

                # -------------------------------------------------
                # SafeVision Crowd Alert
                # -------------------------------------------------
                if processed_frame.crowd_detected:

                    crowd_alert = self._create_crowd_alert(
                        processed_frame.person_count,
                        processed_frame.crowd_status
                    )

                    # Avoid adding repeated crowd alerts
                    # during the alert cooldown period.
                    if self._can_create_crowd_alert():

                        alerts.append(crowd_alert)

                        self.logger.warning(
                            f"CROWD ALERT: "
                            f"{processed_frame.person_count} "
                            f"people detected - "
                            f"{processed_frame.crowd_status}"
                        )

                # -------------------------------------------------
                # Log events
                # -------------------------------------------------
                self._log_events(
                    processed_frame,
                    alerts
                )

                # -------------------------------------------------
                # Save snapshot
                # -------------------------------------------------
                if (
                    len(processed_frame.object_detections) > 0
                    or processed_frame.crowd_detected
                ):

                    self.snapshot_saver.save_snapshot(
                        processed_frame
                    )

                # -------------------------------------------------
                # Performance monitoring
                # -------------------------------------------------
                self.performance_monitor.record_frame(
                    processed_frame.processing_time_ms,
                    len(processed_frame.object_detections)
                )

                # -------------------------------------------------
                # Display
                # -------------------------------------------------
                if self.display is not None:

                    current_fps = (
                        self.performance_monitor
                        .get_current_fps()
                    )

                    self.display.render(
                        processed_frame,
                        current_fps,
                        alerts
                    )

                # -------------------------------------------------
                # Periodic metrics
                # -------------------------------------------------
                self._log_metrics_if_needed()

                # -------------------------------------------------
                # Keyboard input
                # -------------------------------------------------
                key = cv2.waitKey(1) & 0xFF

                if not self._handle_keyboard_input(
                    key,
                    processed_frame
                ):

                    break

        except KeyboardInterrupt:

            self.logger.info(
                "Received keyboard interrupt (Ctrl+C)"
            )

        except Exception as e:

            self.logger.error(
                f"Unexpected error in main loop: {e}",
                exc_info=True
            )

        finally:

            self._shutdown()

    def _create_crowd_alert(
        self,
        person_count: int,
        crowd_status: str
    ) -> Alert:
        """
        Create an alert for crowd buildup.
        """

        # Create a simple detection object so the existing
        # Alert structure can be reused.

        bbox = BoundingBox(
            x=0,
            y=0,
            width=0,
            height=0
        )

        detection = Detection(
            class_name="crowd",
            confidence=1.0,
            bbox=bbox,
            timestamp=time_to_datetime()
        )

        message = (
            f"{crowd_status}: "
            f"{person_count} people detected!"
        )

        return Alert(
            message=message,
            class_name="crowd",
            timestamp=time_to_datetime(),
            detection=detection
        )

    def _can_create_crowd_alert(self) -> bool:
        """
        Check whether enough time has passed since
        the last crowd alert.
        """

        current_time = time.time()

        last_alert = getattr(
            self,
            "_last_crowd_alert_time",
            0.0
        )

        if (
            current_time - last_alert
            < self.config.alert_cooldown
        ):

            return False

        self._last_crowd_alert_time = current_time

        return True

    def _handle_paused_state(self):
        """
        Handle paused state.
        """

        key = cv2.waitKey(100) & 0xFF

        if key == self.KEY_RESUME:

            self._paused = False

            self.logger.info(
                "Surveillance resumed"
            )

        elif key == self.KEY_QUIT:

            self._running = False

    def _handle_keyboard_input(
        self,
        key: int,
        processed_frame
    ) -> bool:
        """
        Handle keyboard input.

        Returns:
            True to continue, False to quit.
        """

        if key == self.KEY_QUIT:

            self.logger.info(
                "Quit command received"
            )

            return False

        elif key == self.KEY_PAUSE:

            self._paused = True

            self.logger.info(
                "Surveillance paused "
                "(press 'r' to resume)"
            )

        elif key == self.KEY_SNAPSHOT:

            filepath = (
                self.snapshot_saver
                .save_snapshot(processed_frame)
            )

            if filepath:

                self.logger.info(
                    f"Manual snapshot saved: {filepath}"
                )

            else:

                self.logger.warning(
                    "Failed to save manual snapshot"
                )

        return True

    def _log_events(
        self,
        processed_frame,
        alerts
    ):
        """
        Log detections, motion, and alerts.
        """

        # Object detections
        for detection in processed_frame.object_detections:

            self.event_logger.log_object_detection(
                detection
            )

        # Motion detections
        for motion_region in processed_frame.motion_regions:

            self.event_logger.log_motion_detection(
                motion_region
            )

        # Crowd information
        if processed_frame.crowd_detected:

            self.logger.info(
                f"SafeVision Crowd Detection: "
                f"{processed_frame.person_count} people - "
                f"{processed_frame.crowd_status}"
            )

        # Alerts
        for alert in alerts:

            self.logger.warning(
                f"ALERT: {alert.message}"
            )

    def _log_metrics_if_needed(self):
        """
        Log performance metrics every 60 seconds.
        """

        current_time = time.time()

        if (
            current_time - self._last_metrics_log_time
            >= self.METRICS_LOG_INTERVAL
        ):

            metrics = (
                self.performance_monitor
                .get_metrics()
            )

            self.event_logger.log_performance_metrics(
                metrics
            )

            self.logger.info(
                f"Performance: "
                f"{metrics.current_fps:.1f} FPS, "
                f"{metrics.total_frames} frames, "
                f"{metrics.total_detections} detections"
            )

            self._last_metrics_log_time = current_time

    def _shutdown(self):
        """
        Gracefully shutdown all components.
        """

        self.logger.info(
            "Shutting down SafeVision..."
        )

        try:

            # Final metrics
            metrics = (
                self.performance_monitor
                .get_metrics()
            )

            self.event_logger.log_performance_metrics(
                metrics
            )

            self.logger.info(
                self.performance_monitor.get_summary()
            )

            # Close display
            if self.display is not None:

                self.display.close()

            # Close logger
            self.event_logger.close()

            # Release webcam
            self.video_capture.release()

            # Destroy OpenCV windows
            cv2.destroyAllWindows()

            self.logger.info(
                "SafeVision shutdown complete"
            )

        except Exception as e:

            self.logger.error(
                f"Error during shutdown: {e}"
            )

    def stop(self):
        """
        Stop the surveillance system.
        """

        self._running = False


def time_to_datetime():
    """
    Return the current datetime.

    This helper keeps datetime creation in one place.
    """

    from datetime import datetime

    return datetime.now()