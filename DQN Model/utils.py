import cv2
import numpy as np

def preprocess_image(image):
    image = cv2.resize(image, (240, 240))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def draw_bounding_box(image, position, grid_size=60):
    x, y = position
    cv2.rectangle(image, (y, x), (y + grid_size, x + grid_size), (0, 255, 0), 2)
    return image
