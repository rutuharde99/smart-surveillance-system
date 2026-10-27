"""
Data models for the Smart Surveillance System.

This module defines all core data structures using Python dataclasses
for type safety and clarity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata associated with a captured frame."""
    frame_number: int
    timestamp: datetime
    width: int
    height: int


@dataclass
class BoundingBox:
    """Rectangular region in a frame."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    
    def area(self) -> int:
        """Calculate the area of the bounding box in pixels."""
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        """Calculate the center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Detection:
    """Object detection result."""
    class_name: str
    confidence: float  # 0.0 to 1.0
    bbox: BoundingBox
    timestamp: datetime


@dataclass
class MotionRegion:
    """Motion detection result."""
    bbox: BoundingBox
    area: int  # Pixel count
    timestamp: datetime


@dataclass
class ProcessedFrame:
    """Complete frame processing result."""
    original_frame: np.ndarray
    metadata: FrameMetadata
    object_detections: List[Detection]
    motion_regions: List[MotionRegion]
    processing_time_ms: float

    # SafeVision crowd analysis
    person_count: int = 0
    crowd_status: str = "NORMAL"
    crowd_detected: bool = False
    high_crowd: bool = False

@dataclass
class Alert:
    """Alert notification."""
    message: str
    class_name: str
    timestamp: datetime
    detection: Detection


@dataclass
class PerformanceMetrics:
    """System performance statistics."""
    current_fps: float
    average_fps: float
    total_frames: int
    total_detections: int
    average_processing_time_ms: float
    timestamp: datetime


@dataclass
class Configuration:
    """System configuration settings."""
    
    # Object Detection
    model_path: str = "yolov8n.pt"
    detection_threshold: float = 0.5
    target_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "dog", "cat", "bicycle", "motorcycle"
    ])
    
    # Motion Detection
    motion_sensitivity: str = "medium"
    motion_min_area: int = 100
    
    # Display
    display_enabled: bool = True
    window_name: str = "Smart Surveillance"
    
    # Snapshot
    snapshot_enabled: bool = True
    snapshot_directory: str = "snapshots"
    snapshot_rate_limit: float = 1.0
    
    # Logging
    log_directory: str = "logs"
    log_max_size_mb: int = 10
    
    # Alerts
    alert_classes: List[str] = field(default_factory=lambda: ["person"])
    alert_cooldown: float = 5.0
    
    # Performance
    target_fps: int = 10
    preprocessing_enabled: bool = False
    resize_width: Optional[int] = None
