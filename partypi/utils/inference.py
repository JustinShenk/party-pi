import os
import cv2
import numpy as np
from keras.preprocessing import image

utils_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(utils_dir)


def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)


def load_detection_model(model_path=None):
    local_cascade_path = os.path.join(base_dir, 'face.xml')
    if not os.path.exists(local_cascade_path):
        raise NameError('File not found:', local_cascade_path)
    model_path = local_cascade_path
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(
        gray_image_array, 1.3, 5, minSize=(50, 50))


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors


def get_labels():
    return {
        0: 'angry',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }


def get_class_to_arg():
    return {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6,
        'crazy': 5,
        'euphoria': 3,
        'wonder': 2,
        'delighted': 3,
        'pro-wrestler': 0
    }
