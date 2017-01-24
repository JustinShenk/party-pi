#!/usr/lib/env python

import os
import cv2
import random
import time
import numpy as np
import json

from uploader import Uploader

BRAND = "partypi.net"
VERSION = "0.1.2"
PURPLE = (68, 54, 66)

# Load string constants from json file.
with open('emotions.json', 'r') as f:
    d = f.read()
    data = json.loads(d)
    EMOTIONS = data['emotions']
    WAIT_CAPTIONS = data['wait_captions']

REDUCTION_FACTOR = 1.  # Reduction factor for timing.
FONT = cv2.FONT_HERSHEY_SIMPLEX
OPACITY = 0.4
HAT_PATH = 'images/hat.png'

# Set Haar cascade path.
if cv2.__version__.startswith('3'):
    CASCADE_PATH = "/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
else:  # Assume OpenCV version is 2.
    CASCADE_PATH = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def find_faces(frame):
    """
    Find faces using Haar cascade.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        frame_gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=0)
    return faces


def add_text(frame, text, origin, size=1.0, color=(255, 255, 255), thickness=1):
    """
    Put text on current frame.
    """
    originx, originy = origin
    origin = (int(originx), int(originy))
    cv2.putText(frame, text, origin,
                FONT, size, color, 2)


def get_image_path():
    """
    Get path for saving a new image.
    """
    img_prefix = 'img_'
    extension = '.png'
    nr = 0
    for file in os.listdir(os.getcwd() + '/img'):
        if file.endswith(extension):
            file = file.replace(img_prefix, '')
            file = file.replace(extension, '')
            # print file
            file_nr = int(file)
            nr = max(nr, file_nr)
    img_nr = nr + 1
    imagePath = 'img/' + str(img_prefix) + \
        str(img_nr) + str(extension)
    return imagePath


class PartyPi(object):

    def __init__(self, piCam=False, windowSize=(1200, 1024), resolution=(1280 // 2, 1024 // 2), gray=False):
        self.piCam = piCam
        self.windowSize = windowSize
        self.gray = gray
        self.faceSelect = False
        self.easyMode = None

        # TODO: Integrate into `EMOTIONS`.
        # EMOTIONS2 = ['psycho', 'John Cena', 'ecstasy', 'duckface']
        self.photo = cv2.imread('img_1.png')  # To prepare window transition.
        self.screenwidth, self.screenheight = self.windowSize

        # Setup for Raspberry Pi.
        if 'raspberrypi' in os.uname():
            self.initialize_raspberry(resolution)
        else:
            self.initialize_webcam()

        # Reinitialize screenwidth and height in case changed by system.
        self.screenwidth, self.screenheight = self.frame.shape[:2]
        print("first screenheight:", self.screenheight,
              self.screenwidth, self.frame.shape)
        print("Window size:", self.frame.shape)
        self.uploading_caption_location = self.screenwidth // 5, self.screenheight // 3 if self.raspberry else self.screenheight // 4 + 30

        # Complete setup.
        self.setup_game()

    def initialize_webcam(self):
        """ Initialize camera and screenwidth and screenheight.
        """
        self.raspberry = False
        self.cam = cv2.VideoCapture(0)
        _, self.frame = self.cam.read()
        # Update class variables.
        self.screenheight, self.screenwidth = self.frame.shape[:2]
        print(self.frame.shape)
        self.cam.set(3, self.screenwidth)
        self.cam.set(4, self.screenheight)
        _, self.frame = self.cam.read()

    def initialize_raspberry(self, resolution):
        """ Set up piCamera module or webcam.

        """
        print("PartyPi v0.1.0 for Raspberry Pi, Coxi Christmas Party Edition")
        self.raspberry = True
        self.resolution = resolution
        # Set up picamera module.
        if self.piCam:
            self.setup_picamera()
        else:  # Use webcam (Note: not completely tested).
            self.cam = cv2.VideoCapture(0)
            _, self.frame = self.cam.read()
            # self.cam.set(3, self.screenwidth)
            # self.cam.set(4, self.screenheight)

    def setup_picamera(self):
        """ Set up piCamera for rasbperry pi camera module.

        """
        from picamera import PiCamera
        from picamera.array import PiRGBArray
        piCamera = PiCamera()
        # self.piCamera.resolution = (640, 480)
        piCamera.resolution = self.resolution[0], self.resolution[1]
        self.screenwidth, self.screenheight = piCamera.resolution
        # self.piCamera.framerate = 10
        piCamera.hflip = True
        piCamera.brightness = 55
        self.rawCapture = PiRGBArray(
            piCamera, size=(self.screenwidth, self.screenheight))
        self.frame = np.empty(
            (self.screenheight, self.screenwidth, 3), dtype=np.uint8)
        self.piCamera = piCamera
        time.sleep(1)

    def setup_game(self):
        """ Initialize variables, set up icons and face cascade.

        """        
        self.looping = True
        self.faceSelect = False
        self.easyMode = None
        self.currentEmotion = EMOTIONS[0]
        self.countx = None

        # Initialize mouse click positions.
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.click_point_right_x = None
        self.click_point_right_y = None

        self.calibrated = False
        self.looping = True
        self.currentCaptionIndex = 0
        self.tickcount = 0
        self.modeLoadCount = 0
        self.curr_level = 0
        self.result = []
        self.uploader = Uploader()

        # TODO: Place icons in a dictionary or object.
        self.easyIcon = cv2.imread('images/easy.png')
        self.hardIcon = cv2.imread('images/hard.png')
        self.playIcon = cv2.imread('images/playagain.png')
        self.playIconOriginal = self.playIcon.copy()
        # self.playIcon1 = cv2.imread('playagain1.png')
        # self.playIcon2 = cv2.imread('playagain2.png')
        # self.playIcon3 = cv2.imread('playagain3.png')
        self.easySize = self.hardSize = self.easyIcon.shape[:2]
        self.playSize = self.playIcon.shape[:2]
        self.christmas = cv2.imread(
            'images/christmas.png', cv2.IMREAD_UNCHANGED)
        self.hat = cv2.imread(HAT_PATH, cv2.IMREAD_UNCHANGED)
        if self.hat is None:
            raise ValueError('No hat image found at `{}`'.format(HAT_PATH))

        print("Camera initialized")
        # if not self.raspberry:
        #     print "MAC or PC initialize"
        #     self.cam.set(3, self.screenwidth)
        #     self.cam.set(4, self.screenheight)
        self.flashon = False
        self.showAnalyzing = False
        self.currCount = None
        # `self.static`: Used to indicate if emotion should stay on screen.
        self.static = False
        self.photoMode = False
        cv2.namedWindow("PartyPi", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("PartyPi", self.mouse)
        cv2.resizeWindow("PartyPi", self.windowSize[0], self.windowSize[1])
        # Returns - TypeError: Required argument 'prop_value' (pos 3) not found
        # cv2.setWindowProperty(
        #     "PartyPi", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PartyPi", cv2.WND_PROP_AUTOSIZE,
        #                       cv2.WINDOW_AUTOSIZE)

        if self.piCam == True:
            # Capture frames from the camera.
            for _frame in self.piCamera.capture_continuous(self.rawCapture, format='bgr', use_video_port=True):
                # self.frame = cv2.flip(_frame.array, 1)
                self.frame = _frame.array
                self.frame.flags.writeable = True
                self.screenheight, self.screenwidth = self.frame.shape[:2]
                # TODO: Consider passing frame as local variable rather than
                # global.
                self.game_loop()
        else:
            while self.looping:
                self.game_loop()

    def game_loop(self):
        """ Start the game loop. Listen for escape key.

        """
        # TODO: Check if following line is redundant.
        self.screenheight, self.screenwidth = self.frame.shape[:2]
        if self.curr_level == 0:
            self.level0()
        elif self.curr_level == 1:
            self.level1()
        elif self.curr_level == 2:
            self.level2()

        # Catch escape key 'q'.
        if self.curr_level == 2:
            keypress = cv2.waitKey(500) & 0xFF
        else:
            keypress = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame.
        if self.piCam == True:
            self.rawCapture.truncate(0)

        self.listen_for_end(keypress)

    def level0(self):
        """ Select a mode: Easy or Hard.

        """
        self.tickcount + 1

        if self.raspberry:
            self.tickcount += 1
        self.capture_frame()

        # Draw "Easy" and "Hard".
        add_text(self.frame, "Easy", (self.screenwidth // 8,
                                      (self.screenheight * 3) // 4), size=3)
        add_text(self.frame, "Hard", (self.screenwidth // 2,
                                      (self.screenheight * 3) // 4), size=3)

        # Listen for mode selection.
        if self.currPosX and self.currPosX < self.screenwidth / 2:
            cv2.rectangle(self.overlay, (0, 0), (self.screenwidth // 2,
                                                 int(self.screenheight)), (211, 211, 211), -1)
        else:
            cv2.rectangle(self.overlay, (self.screenwidth // 2, 0),
                          (self.screenwidth, self.screenheight), (211, 211, 211), -1)
        if self.click_point_x:  # If user clicks left mouse button.
            # self.easyMode = True if self.click_point_x < self.screenwidth / 2
            # else False # For positional selection.
            self.easyMode = True
            self.tickcount = 0
            self.curr_level = 1
            self.click_point_x = None
            self.click_point_right_x = None

        if self.click_point_right_x:
            self.easyMode = False
            self.tickcount = 0
            self.curr_level = 1
            self.click_point_x = None
            self.click_point_right_x = None

        # Draw faces.
        faces = find_faces(self.frame)
        if self.faceSelect:
            self.select_mode(faces)
        cv2.addWeighted(self.overlay, OPACITY, self.frame,
                        1 - OPACITY, 0, self.frame)
        # Display frame.
        add_text(self.frame, BRAND, ((self.screenwidth // 5) * 4,
                                     self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        # Draw Christmas logo.
        self.draw_hat(self.frame, faces)
        self.draw_christmas_logo(self.frame)
        # Show image.
        cv2.imshow('PartyPi', self.frame)

        # if not self.calibrated and self.tickcount == 10:
        #     self.t1 = time.clock() - t0
        #     print "t1", self.t1
        #     self.calibrated = True

    def level1(self):
        """ Display emotion prompts, upload image, and display results.

        """
        self.showBegin = False
        self.capture_frame()
        self.tickcount += 1
        if self.raspberry:
            self.tickcount += 1
        timer = int(self.tickcount * REDUCTION_FACTOR)
        self.prompt_emotion()

        # Show 'Begin' after some time
        if timer < 70:
            pass
        elif timer < 80:
            self.showBegin = True

        # Start count "3..."
        elif timer < 100:
            pass
        elif timer >= 100 and timer <= 110:
            self.showBegin = False
            self.currCount = 3
            self.countx = self.screenwidth - (self.screenwidth / 5) * 4
        elif timer >= 110 and timer < 120:
            self.currCount = 2
            self.countx = self.screenwidth - (self.screenwidth / 5) * 3
        elif timer >= 120 and timer < 130:
            self.currCount = 1
            self.countx = self.screenwidth - (self.screenwidth / 5) * 2

        # Flash, and then show analyzing
        elif timer >= 130 and timer < 136:
            self.flashon = True
            self.currCount = None
            self.countx = -100  # make it disappear
            if not self.raspberry:
                if timer >= 133 and timer < 134:
                    self.photoMode = True
                    self.photo = self.frame.copy()
                    self.startProcess = True
                else:
                    self.startProcess = False
                    self.showAnalyzing = True
            else:
                # Raspberry-specific timing
                if timer >= 133 and timer <= 134:
                    self.photoMode = True
                    self.photo = self.frame.copy()
                    self.startProcess = True
                else:
                    self.startProcess = False
                    self.showAnalyzing = True

        # Else take photo at timer == 136.
        else:
            self.startProcess = False
            self.flashon = False
            self.showAnalyzing = False
            self.curr_level = 2
            self.click_point_y = None

        # Draw the count "3..".
        if self.currCount:
            self.overlay = self.frame.copy()
            cv2.rectangle(self.overlay, (0, int(self.screenheight * (4. / 5))),
                          (self.screenwidth, self.screenheight), (224, 23, 101), -1)
            cv2.addWeighted(self.overlay, OPACITY, self.frame,
                            1 - OPACITY, 0, self.frame)
            cv2.putText(self.frame, str(self.currCount), (int(self.countx), int(
                self.screenheight * (7. / 8))), FONT, 1.0, (255, 255, 255), 2)

        # Draw other text and flash on screen.
        textSize = 2.8 if self.raspberry else 3.6
        if self.showBegin:
            cv2.putText(self.frame, "Begin!", (self.screenwidth // 3, self.screenheight // 2),
                        FONT, textSize, (255, 255, 255), 2)
        elif self.flashon:
            cv2.rectangle(self.frame, (0, 0), (self.screenwidth,
                                               self.screenheight), (255, 255, 255), -1)
        if self.showAnalyzing:
            textSize = 0.7 if self.raspberry else 1.7
            add_text(self.frame, WAIT_CAPTIONS[self.currentCaptionIndex % len(
                WAIT_CAPTIONS)], self.uploading_caption_location, textSize, color=(224, 23, 101))
            self.draw_christmas_logo(self.frame)
        # Display image.
        add_text(self.frame, BRAND, ((self.screenwidth // 5) * 4,
                                     self.screenheight // 7), color=PURPLE, size=0.5, thickness=0.5)
        cv2.imshow('PartyPi', self.frame)

        if self.photoMode and self.startProcess:
            print("take photo")
            self.take_photo()

    def level2(self):
        """ Show analyzing, then present photo, then reset game.

        """
        self.tickcount += 1
        self.currentCaptionIndex += 1
        # self.capture_frame()

        if self.raspberry:
            self.tickcount += 1

        cv2.putText(self.photo, "[Press any button]", (self.screenwidth // 2, int(
            self.screenheight * (6. / 7))), FONT, 1.0, (62, 184, 144), 2)

        # if self.tickcount % 5 == 0:
        if self.faceSelect:
            faces = find_faces(self.frame)
            if len(faces):
                rightFace = max([x for x, y, w, h in faces])
                bottomFace = max([y for x, y, w, h in faces])
                if rightFace > self.screenwidth - self.playSize[1] * 1.2:
                    self.playIcon = self.playIcon1.copy()
                    if bottomFace > self.screenheight / 2:
                        self.playIcon = self.playIcon2.copy()
                else:
                    self.playIcon = self.playIconOriginal.copy()
            else:
                self.playIcon = self.playIconOriginal.copy()

            # Draw a rectangle around the faces.
            for (x, y, w, h) in faces:
                cv2.rectangle(self.frame, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                if x > (self.screenwidth - self.playSize[1]) and y > (self.screenheight - self.playSize[0]):
                    self.reset()

            # Show live image in corner.
            self.photo[self.screenheight - self.easySize[0]:self.screenheight, self.screenwidth - self.easySize[0]:self.screenwidth] = self.frame[
                self.screenheight - self.easySize[1]: self.screenheight, self.screenwidth - self.easySize[1]: self.screenwidth]

        # TODO: Move to separate function.
        # Face-based replay and mode selection (disabled).
        # self.overlay = self.photo.copy()
        # Show 'Play Again'. Disabled for party.
        # self.overlay[self.screenheight - self.playSize[1]: self.screenheight, self.screenwidth - self.playSize[1]: self.screenwidth] = self.playIcon[
        #     0: self.playSize[1], 0: self.playSize[0]]

        # Blend photo with overlay.
        # cv2.addWeighted(self.overlay, OPACITY, self.photo,
        #                 1 - OPACITY, 0, self.photo)

        # Draw logo or title.
        add_text(self.photo, BRAND, ((self.screenwidth // 5) * 4,
                                     self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        self.draw_christmas_logo(self.photo)
        # self.draw_hat(self.photo, faces)
        cv2.imshow('PartyPi', self.photo)

    def mouse(self, event, x, y, flags, param):
        """ Listen for mouse.

        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.currPosX, self.currPosY = x, y
            # print "curposX,Y", x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.click_point_x, self.click_point_y = x, y
            if self.curr_level == 0:
                self.easyMode = True
                self.curr_level = 1
            if self.curr_level == 2:
                self.reset()

        elif event == cv2.EVENT_RBUTTONUP:
            self.click_point_right_x, self.click_point_right_y = x, y
            if self.level2:
                self.reset()
                self.easyMode = False
                self.curr_level = 1

    def reset(self):
        """ Reset to beginning state.

        """
        self.curr_level = 0
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.click_point_right_x = None
        self.currentEmotion = EMOTIONS[1]
        self.result = []
        self.tickcount = 0
        self.static = False
        self.playIcon = self.playIconOriginal.copy()

    def select_mode(self, faces):
        """ Select between easy and hard modes.

        """
        # Draw a rectangle around the faces.
        for (x, y, w, h) in faces:

            # Select easy mode with face
            if x + w < self.easySize[1] and y > self.screenheight - self.easySize[0]:
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = True
                    self.modeLoadCount = 0
                    self.curr_level = 1
                    self.tickcount = 0

            # Select hard mode with face.
            elif x + w > (self.screenwidth - self.hardSize[1]) and y > (self.screenheight - self.hardSize[0]):
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = False
                    self.modeLoadCount = 0
                    self.curr_level = 1
                    self.tickcount = 0

        # Draw easy mode selection box.
        if not self.raspberry:
            self.overlay[self.screenheight - self.easySize[0]:self.screenheight,
                         0:self.easySize[1]] = self.easyIcon

            self.overlay[self.screenheight - self.hardSize[0]:self.screenheight,
                         self.screenwidth - self.hardSize[1]:self.screenwidth] = self.hardIcon

    def draw_christmas_logo(self, frame):
        """ Draw Christmas logo on top right screen.

        """
        if self.screenheight < 700:
            y0 = 0
        else:
            y0 = (self.screenheight // 7) + 0
        y1 = y0 + self.christmas.shape[0]
        if self.screenwidth < 700:
            x0 = 0
        else:
            x0 = 2 * self.screenwidth // 3
        x1 = x0 + self.christmas.shape[1]

        # Remove black background from png image.
        for c in range(0, 3):
            xmasSlice = self.christmas[:, :, c] * \
                (self.christmas[:, :, 3] / 255.0)
            backgroundSlice = frame[y0:y1, x0:x1, c] * \
                (1.0 - self.christmas[:, :, 3] / 255.0)
            frame[y0:y1, x0:x1, c] = xmasSlice + backgroundSlice

    def draw_hat(self, frame, faces):
        """ Draws hats above detected faces.

        """
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        w_offset = 1.3
        x_offset = -20
        y_offset = 80

        for x, y, w, h in faces:
            hat = self.hat.copy()

            # Scale hat to fit face.
            hat_width = int(w * w_offset)
            hat_height = int(hat_width * hat.shape[0] / hat.shape[1])
            hat = cv2.resize(hat, (hat_width, hat_height))

            # Clip hat if outside frame.
            hat_left = 0
            hat_top = 0
            hat_bottom = hat_height
            hat_right = hat_width
            y0 = y - hat_height + y_offset
            if y0 < 0:  # If the hat starts above the frame, clip it.
                hat_top = abs(y0)  # Find beginning of hat ROI.
                y0 = 0
            y1 = y0 + hat_height - hat_top
            if y1 > frame_height:
                hat_bottom = hat_height - (y1 - frame_height)
                y1 = frame_height
            x0 = x + x_offset
            if x0 < 0:
                hat_left = abs(x0)
                x0 = 0
            x1 = x0 + hat_width - hat_left
            if x1 > frame_width:
                hat_right = hat_width - (x1 - frame_width)
                x1 = frame_width

            # Remove background from hat image.
            for c in range(0, 3):
                hat_slice = hat[hat_top:hat_bottom, hat_left:hat_right, c] * \
                    (hat[hat_top:hat_bottom, hat_left:hat_right, 3] / 255.0)
                bg_slice = frame[y0:y1, x0:x1, c] * \
                    (1.0 - hat[hat_top:hat_bottom, hat_left:hat_right, 3]
                        / 255.0)
                frame[y0:y1, x0:x1, c] = hat_slice + bg_slice

    def capture_frame(self):
        """ Capture frame-by-frame.


        """

        if not self.piCam:
            ret, frame = self.cam.read()
            self.frame = cv2.flip(frame, 1)

        self.overlay = self.frame.copy()

    def take_photo(self):
        """ Take photo and prepare to write, then send to PyImgur.

        """
        imagePath = get_image_path()

        # Get faces for Christmas hat.
        faces = find_faces(self.photo)

        # If internet connection is poor, use black and white image.
        if self.gray:
            bwphoto = cv2.cvtColor(self.photo, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(imagepath, bwphoto)
            self.result = self.uploader.upload_img(imagePath)
        else:
            add_text(self.photo, BRAND, (int((self.screenwidth / 5) * 4),
                                         int(self.screenheight / 7)), color=(255, 255, 255), size=0.5, thickness=0.5)
            self.draw_christmas_logo(self.photo)
            self.draw_hat(self.photo, faces)
            cv2.imwrite(imagePath, self.photo)
            self.result = self.uploader.upload_img(imagePath)
        self.display()

    def display(self):
        """ Display results of game on screen, with winner and scores for emotions.

        """
        scores = []
        maxfirstemo = None
        maxsecondemo = None
        firstEmotion = None

        if self.result:  # If faces present.

            # Get lists of player points.
            firstEmoList = [
                (round(x['scores'][self.currentEmotion] * 100)) for x in self.result]
            secondEmoList = [(round(
                x['scores'][self.seccurrentEmotion] * 100)) for x in self.result]

            # Compute the scores into `scoresList`.
            scoresList = []
            if self.easyMode:  # Easy mode is points for first emotion.
                scoresList = firstEmoList  # playerNumber, scores
            # Hard mode scores are product of points of both emotions.
            else:
                for i in range(len(firstEmoList)):
                    scoresList.append(
                        (firstEmoList[i] + 1) * (secondEmoList[i] + 1))
            print("scoresList:", scoresList)
            textSize = 0.5 if self.raspberry else 0.8
            # Draw the scores for the faces.
            for idx, currFace in enumerate(self.result):
                faceRectangle = currFace['faceRectangle']

                # Get points for first emotion.
                firstEmotion = firstEmoList[idx]
                secEmotion = secondEmoList[idx]

                # Format points.
                if firstEmotion == 1:
                    textToWrite = "%i point: %s" % (
                        firstEmotion, self.currentEmotion)
                else:
                    textToWrite = "%i points: %s" % (
                        firstEmotion, self.currentEmotion)
                if secEmotion == 1:
                    secondLine = "%i point: %s" % (
                        secEmotion, self.seccurrentEmotion)
                else:
                    secondLine = "%i points: %s" % (
                        secEmotion, self.seccurrentEmotion)

                # Display points.
                scoreHeightOffset = 10 if self.easyMode else 40
                cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                    'top'] - scoreHeightOffset), FONT, textSize, (232, 167, 35), 2)

                if not self.easyMode:
                    cv2.putText(self.photo, secondLine, (faceRectangle['left'], faceRectangle[
                                'top'] - 10), FONT, textSize, (232, 167, 35), 2)

            # Display 'Winner: ' above player with highest score.
            oneWinner = True
            finalScores = scoresList
            winner = finalScores.index(max(finalScores))
            maxScore = max(finalScores)

            # Multiple winners - tie breaker.
            if finalScores.count(maxScore) > 1:
                print("Multiple winners!")
                oneWinner = False
                tiedWinners = []
                for ind, i in enumerate(finalScores):
                    if i == maxScore:
                        tiedWinners.append(ind)

            print("Scores:", finalScores, "Winner:", winner)

            # Identify winner's face.
            firstRectLeft = self.result[winner]['faceRectangle']['left']
            firstRectTop = self.result[winner]['faceRectangle']['top']
            if oneWinner:
                tiedTextHeightOffset = 40 if self.easyMode else 70
                cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - tiedTextHeightOffset),
                            FONT, textSize, (232, 167, 35), 2)
            else:
                tiedTextHeightOffset = 40 if self.easyMode else 70
                print("tiedWinners:", tiedWinners)
                for winner in tiedWinners:
                    # FIXME: show both
                    firstRectLeft = self.result[
                        winner]['faceRectangle']['left']
                    firstRectTop = self.result[winner]['faceRectangle']['top']
                    cv2.putText(self.photo, "Tied: ", (firstRectLeft, firstRectTop - tiedTextHeightOffset),
                                FONT, textSize, (232, 167, 35), 2)

        else:
            print("No results found.")

    def prompt_emotion(self):
        """ Display prompt for emotion on screen.

        """
        textSize = 1.0 if self.raspberry else 1.2
        width = self.screenwidth // 5 if self.easyMode else 10
        cv2.putText(self.frame, "Show " + self.random_emotion() + '_',
                    (width, 3 * (self.screenheight // 4)), FONT, textSize, (255, 255, 255), 2)

    def random_emotion(self):
        """ Pick a random emotion from list of emotions.

        """
        if self.tickcount * REDUCTION_FACTOR > 30 or self.static:
            self.static = True
            emotionString = str(
                self.currentEmotion) if self.easyMode else self.currentEmotion + '+' + self.seccurrentEmotion
            return emotionString
        else:
            self.currentEmotion = random.choice(EMOTIONS)
            randnum = (EMOTIONS.index(self.currentEmotion) +
                       random.choice(list(range(1, 7)))) % 8
            self.seccurrentEmotion = EMOTIONS[randnum]
            if self.easyMode:
                return self.currentEmotion
            else:
                return self.currentEmotion + '+' + self.seccurrentEmotion

    def listen_for_end(self, keypress):
        """ Listen for 'q', left, or right keys to end game.

        """
        if keypress != 255:
            print(keypress)
            if keypress == ord('q'):  # 'q' pressed to quit
                print("Escape key entered")
                self.looping = False
                self.end_game()
            elif self.curr_level == 0:
                if keypress == 81 or keypress == 2:  # left
                    self.easyMode = True
                    self.tickcount = 0
                    self.curr_level = 1
                elif keypress == 83 or keypress == 3:  # right
                    self.easyMode = False
                    self.tickcount = 0
                    self.curr_level = 1
            elif self.curr_level == 2:
                self.reset()

    def end_game(self):
        """ When everything is done, release the capture.

        """
        if not self.piCam:
            self.cam.release()
            add_text(self.frame, "Press any key to quit_",
                     (self.screenwidth // 4, self.screenheight // 3))
            # self.presentation(self.frame)
            add_text(self.frame, BRAND, ((self.screenwidth // 5) * 4,
                                         self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        else:
            self.piCamera.close()

        cv2.imshow("PartyPi", self.frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """ Run application.
    """
    app = PartyPi()


if __name__ == '__main__':
    main()
