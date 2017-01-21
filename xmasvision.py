#!/usr/bin/env python

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import cv2
import sys
import io
import time
import picamera
import picamera.array
import os
import pygame

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Taken from "Capturing to an OpenCV object"
# http://picamera.readthedocs.org/en/latest/recipes1.html

# Create the in-memory stream
stream = io.BytesIO()

face = 0
while face == 0:
    # Acquiring pic
    with picamera.PiCamera() as camera:
        camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
        camera.vflip = True
        time.sleep(1)
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            image = stream.array

    # Adapted from "Haar-cascade Detection in OpenCV"
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Got pic, checking for faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    if len(faces) > 0:
        face = 1
        print "Found Face! "
        cv2.imwrite('foundface.jpg', image)

cv2.destroyAllWindows()

# driver selection routine borrowed from:
# https://web.archive.org/web/20130601053413/http://www.karoltomala.com/blog/?p=679

screen = None

drivers = ['fbcon', 'directfb', 'svgalib']

for driver in drivers:
    if not os.getenv('SDL_VIDEODRIVER'):
        os.putenv('SDL_VIDEODRIVER', driver)
    try:
        print "trying: " + driver
        pygame.display.init()
    except pygame.error:
        print 'Driver: {0} failed.'.format(driver)
        continue
    found = True
    break

    if not found:
        raise Exception('No suitable video driver found!')

original_width = pygame.display.Info().current_w
original_height = pygame.display.Info().current_h

# match camera/image resolution
width = CAMERA_WIDTH
height = CAMERA_HEIGHT

# Use the face photo as our canvas
screen = pygame.display.set_mode((width, height))
pygame.mouse.set_visible(False)
bg = pygame.image.load('foundface.jpg')
bg = pygame.transform.scale(bg, (width, height))
screen.blit(bg, bg.get_rect())

# hat is longer on the right than the wearable
# area (because of the little puff ball) tweak
# value for your own hats
hat_offset = 330

# put the hat on the cat
if len(faces) > 0:
    for (x, y, w, h) in faces:
        hat = pygame.image.load('hat.png').convert_alpha()
        hat_size = int((hat.get_width() - hat_offset) / w)
        if hat_size < 1:
            hat_size = 1
        hat_offset = int(hat_offset * (1.0 / hat_size))
        hat = pygame.transform.scale(hat, (int(
            hat.get_width() * (1.0 / hat_size)), int(hat.get_height() * (1.0 / hat_size))))
        hat_w = hat.get_width()
        hat_h = hat.get_height()
        # pygame.draw.rect(screen, (255, 0, 0), (x, y - hat_h, hat_w, hat_h),
        # 1) # hat border, helpful for debugging
        print "x: " + str(x)
        print "y: " + str(y)
        # fudge placement a little to put hat on, rather than over
        fx = int(x * 0.96)
        fy = int(y * 1.04)
        # fudge placement a little to put hat on, rather than over
        screen.blit(hat, (fx, fy - hat_h, hat_w, hat_h))
        # pygame.draw.rect(screen, (0, 255, 0), (x, y, w, h), 1) # face border

# Uncomment if you want to see the intermediary face + hat photo
# pygame.display.update()
pygame.image.save(screen, 'hatted.png')

# Resize canvas to fit monitor
width = original_width
height = original_height

# load background and photo (with hat) into objects
# display background over photo, allowing transparent region to
# show the photo behind it.
screen = pygame.display.set_mode((width, height))
bg = pygame.image.load('xmascam.png').convert_alpha()
bg = pygame.transform.scale(bg, (width, height))
photo = pygame.image.load('hatted.png')
photo = pygame.transform.scale(
    photo, (int(1.339 * photo.get_width()), int(1.339 * photo.get_height())))
screen.blit(photo, (622, 115, photo.get_width(), photo.get_height()))
screen.blit(bg, bg.get_rect())
pygame.display.update()

time.sleep(10)
sys.exit

pygame.display.quit()
