import cv2
import os
import traceback


OPACITY = 0.4
remote_API = False
BLUE = (232, 167, 35)
GREEN = (62, 184, 144)
YELLOW = (0, 255, 255)
PURPLE = (68, 54, 66)
BRAND = "partypi.net"
VERSION = "0.1.6"
hat_path = 'images/hat.png'


def print_traceback():
    print ("Exception:")
    print ('-' * 60)
    traceback.print_exc()
    print ('-' * 60)
    pass


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def draw_text(coordinates, image_array, text, color=(255, 255, 255), x_offset=0, y_offset=0,
              font_scale=2, thickness=1):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (int(x + x_offset), int(y + y_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def new_image_path():
    """
    Get path for saving a new image.
    """
    img_prefix = 'img_'
    extension = '.png'
    nr = 0
    photos_path = os.path.abspath('../photos')
    if not os.path.exists(photos_path):
        os.mkdir(photos_path)
    for file in os.listdir(photos_path):
        if file.endswith(extension):
            file = file.replace(img_prefix, '')
            file = file.replace(extension, '')
            # print file
            file_nr = int(file)
            nr = max(nr, file_nr)
    img_nr = nr + 1
    image_path = os.path.join(photos_path, str(
        img_prefix) + str(img_nr) + str(extension))
    return image_path
