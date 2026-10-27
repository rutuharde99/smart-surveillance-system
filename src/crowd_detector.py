
"""
Crowd Detector module for SafeVision.

This module counts detected people and determines
whether a crowd is present in the camera frame.
"""

from typing import List

from .models import Detection


class CrowdDetector:
    """
    Detects crowd buildup based on the number of
    people detected by the YOLO object detector.
    """

    def __init__(
        self,
        crowd_threshold: int = 5,
        high_crowd_threshold: int = 10
    ):
        """
        Initialize the crowd detector.

        Args:
            crowd_threshold: Minimum number of people for a crowd alert.
            high_crowd_threshold: Number of people for a high crowd alert.
        """
        self.crowd_threshold = crowd_threshold
        self.high_crowd_threshold = high_crowd_threshold

    def count_people(self, detections: List[Detection]) -> int:
        """
        Count the number of people detected in the current frame.

        Args:
            detections: List of YOLO Detection objects.

        Returns:
            Number of people detected.
        """
        person_count = 0

        for detection in detections:
            if detection.class_name == "person":
                person_count += 1

        return person_count

    def get_crowd_status(self, person_count: int) -> str:
        """
        Determine the crowd status based on person count.

        Args:
            person_count: Number of people detected.

        Returns:
            Crowd status.
        """
        if person_count >= self.high_crowd_threshold:
            return "HIGH CROWD"

        elif person_count >= self.crowd_threshold:
            return "CROWD DETECTED"

        else:
            return "NORMAL"

    def analyze(self, detections: List[Detection]) -> dict:
        """
        Analyze the current detections for crowd buildup.

        Args:
            detections: List of YOLO Detection objects.

        Returns:
            Dictionary containing crowd analysis results.
        """
        person_count = self.count_people(detections)
        status = self.get_crowd_status(person_count)

        return {
            "person_count": person_count,
            "status": status,
            "crowd_detected": person_count >= self.crowd_threshold,
            "high_crowd": person_count >= self.high_crowd_threshold
        }