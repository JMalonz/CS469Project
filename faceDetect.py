import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create face detection instance
base_options = python.BaseOptions(model_asset_path = 'blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options = base_options)
detector = vision.FaceDetector.create_from_options(options)