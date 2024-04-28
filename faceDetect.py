import numpy as np
import mediapipe as mp
from PIL import Image
from PIL import ImageFilter
from typing import Tuple, Union
import math
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # green

def _normalized_to_pixel_coordinates(
    normalized_x: float, 
    normalized_y: float, 
    image_width: int,
    image_height: int
) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def face_detection_filtering(
    file_path
) -> Image:
  # Create face detection instance
  base_options = python.BaseOptions(model_asset_path = 'blaze_face_short_range.tflite')
  options = vision.FaceDetectorOptions(
      base_options = base_options,
      min_detection_confidence = 0.5,
      min_suppression_threshold = 0.3
  )
  detector = vision.FaceDetector.create_from_options(options)

  # Load image
  img = mp.Image.create_from_file(file_path)

  # Detect faces
  detection_result = detector.detect(img)
  print("Number of faces detected: ", len(detection_result.detections))

  # Visualize faces detected (if any)
  img_copy = np.copy(img.numpy_view())
  annotated_img = visualize(img_copy, detection_result)
  rbg_annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
  resized_img = cv2.resize(rbg_annotated_img, (1920, 1080))
  cv2.imshow("test", resized_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Open image using PIL to use image processing
  img = Image.open(file_path)

  # Apply filter to faces detected
  for d in detection_result.detections:
      # Get coordinates of bounding box for current face
      bbox = d.bounding_box
      x = bbox.origin_x
      y = bbox.origin_y
      width = bbox.width
      height = bbox.height
      box = (x, y, x + width, y + height)

      # Crop
      cropped = img.crop(box)
      # cropped.show("test")

      # Apply a filter to the cropped face
      # options include:
      # SMOOTH
      # SMOOTH_MORE
      # SHARPEN
      # UNSHARP MASK
      # MEDIAN BLUR
      # GAUSSIAN BLUR
      filtered_img = cropped.filter(ImageFilter.UnsharpMask)
      # apply a filter multiple times
      # for i in range(3):
      #     filtered_img = filtered_img.filter(ImageFilter.SHARPEN)

      # Compare
      # res = Image.new('RGB', (width * 2, height))
      # res.paste(cropped, (0,0))
      # res.paste(filtered_img, (width, 0))
      # res.show("test")

      # Place back into original
      img.paste(filtered_img, box)

  # Results
  return img