#!/usr/lib/env python
import sys
import os
import cv2
from uploader import Uploader
import random
import time
import numpy as np

brand = "Party Pi v0.0.2"


class PartyPi(object):

    def __init__(self, piCam=False, resolution=(1280 / 2, 1024 / 2), windowSize=(1200, 1024), blackAndWhite=False):
        self.piCam = piCam
        self.windowSize = windowSize
        self.blackAndWhite = blackAndWhite
        self.looping = True
        self.faceSelect = False
        self.easyMode = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.emotions = ['anger', 'contempt', 'disgust',
                         'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.photo = cv2.imread('img_1.png')
        self.resolution = resolution
        self.screenwidth, self.screenheight = self.windowSize
        self.colors = [(0, 100, 0), (4, 4, 230)]
        self.analyzingLabels = ["Make Christmas Party Great Again", "Christmas Elves are Analyzing...", "A Tribe of Unicorns is Working for You",
                                "Take a Sip", "Turn around two times", "Saddling the unicorn", "Nothing to see here", "Uploading to Facebook..j/k", "Computing makes me thirsty"]
        self.currentAnalLabel = 0
        # Setup for Raspberry Pi.
        if 'raspberrypi' in os.uname():
            self.initRaspberryPi()
        else:
            self.initWebcam()
            cv2.setMouseCallback("PartyPi", self.mouse)

        # Reinitialize screenwidth and height in case changed by system.
        self.screenwidth, self.screenheight = self.frame.shape[:2]
        print "first screenheight:", self.screenheight, self.screenwidth, self.frame.shape

        # Finish setup.
        self.setupGame()

    def initWebcam(self):
        self.raspberry = False
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, self.screenwidth)
        self.cam.set(4, self.screenheight)
        _, self.frame = self.cam.read()

    def initRaspberryPi(self):
        print "PartyPi v0.0.2 for Raspberry Pi, Coxi Christmas Party Edition"
        self.raspberry = True

        # Set up picamera module.
        if self.piCam:
            self.pyIt()
        else:
            self.cam = cv2.VideoCapture(0)
            _, self.frame = self.cam.read()
            # self.cam.set(3, self.screenwidth)
            # self.cam.set(4, self.screenheight)

    def pyIt(self):
        """
        Set up piCamera for rasbperry pi camera module.
        """
        from picamera import PiCamera
        from picamera.array import PiRGBArray
        self.piCamera = PiCamera()
        # self.piCamera.resolution = (640, 480)
        self.piCamera.resolution = self.resolution[0], self.resolution[1]
        self.screenwidth, self.screenheight = self.piCamera.resolution
        # self.piCamera.framerate = 10
        self.piCamera.hflip = True
        self.piCamera.brightness = 55
        self.rawCapture = PiRGBArray(
            self.piCamera, size=(self.screenwidth, self.screenheight))
        self.frame = np.empty(
            (self.screenheight, self.screenwidth, 3), dtype=np.uint8)
        time.sleep(1)

    def setupGame(self):
        """
        Initialize variables, set up icons and face cascade.
        """
        self.currEmotion = self.emotions[0]
        self.countx = None
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.calibrated = False
        self.tickcount = 0
        self.modeLoadCount = 0
        self.level = 0
        self.result = []
        self.uploader = Uploader()
        cascPath = "face.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.pretimer = None

        # TODO Consider placing icons in a dictionary.
        self.easyIcon = cv2.imread('easy.png')
        self.hardIcon = cv2.imread('hard.png')
        self.playIcon = cv2.imread('playagain.png')
        self.playIconOriginal = self.playIcon.copy()
        # self.playIcon1 = cv2.imread('playagain1.png')
        # self.playIcon2 = cv2.imread('playagain2.png')
        # self.playIcon3 = cv2.imread('playagain3.png')
        self.easySize = self.hardSize = self.easyIcon.shape[:2]
        self.playSize = self.playIcon.shape[:2]
        self.christmas = cv2.imread('christmas.png', -1)
        self.hat = cv2.imread('hat.png', -1)

        print "Camera initialize"
        # if not self.raspberry:
        #     print "MAC or PC initialize"
        #     self.cam.set(3, self.screenwidth)
        #     self.cam.set(4, self.screenheight)
        self.flashon = False
        self.showAnalyzing = False
        self.opacity = 0.4
        self.redfactor = 1.  # Reduction factor for timing.
        self.currCount = None
        # `self.static`: Used to indicate if emotion should stay on screen.
        self.static = False
        self.photoMode = False
        cv2.namedWindow("PartyPi", cv2.WINDOW_NORMAL)
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
                self.gameLoop()
        else:
            while self.looping:
                self.gameLoop()

    def gameLoop(self):
        """
        Start the game loop. Listen for escape key.
        """
        # TODO: Check if following line is redundant.
        self.screenheight, self.screenwidth = self.frame.shape[:2]
        if self.level == 0:
            self.level0()
        elif self.level == 1:
            self.level1()
        elif self.level == 2:
            self.level2()

        # Catch escape key 'q'.
        keypress = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame.
        if self.piCam == True:
            self.rawCapture.truncate(0)

        self.listenForEnd(keypress)

    def level0(self):
        """
        Select a mode: Easy or Hard.
        """
        self.tickcount += 1

        if self.raspberry:
            self.tickcount += 1
        self.captureFrame()

        # Draw "Easy" and "Hard".
        self.addText(self.frame, "Easy", (self.screenwidth / 8,
                                          (self.screenheight * 3) / 4), size=3)
        self.addText(self.frame, "Hard", (self.screenwidth / 2,
                                          (self.screenheight * 3) / 4), size=3)

        # Listen for mode selection.
        if self.currPosX and self.currPosX < self.screenwidth / 2:
            cv2.rectangle(self.overlay, (0, 0), (self.screenwidth / 2,
                                                 self.screenheight), (211, 211, 211), -1)
        else:
            cv2.rectangle(self.overlay, (self.screenwidth / 2, 0),
                          (self.screenwidth, self.screenheight), (211, 211, 211), -1)
        if self.click_point_x:
            # self.easyMode = True if self.click_point_x < self.screenwidth / 2
            # else False # For positional selection.
            self.easyMode = True
            self.tickcount = 0
            self.level = 1
            self.click_point_x = None
            self.click_point_right_x = None

        if self.click_point_right_x:
            self.easyMode = False
            self.tickcount = 0
            self.level = 1
            self.click_point_x = None
            self.click_point_right_x = None

        # Draw faces.
        faces = self.findFaces(self.frame)
        if self.faceSelect:
            self.selectMode(faces)

        # Display frame.
        self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        # Draw Christmas logo.
        self.drawChristmasLogo(self.frame)
        self.drawHat(self.frame, faces)
        # Show image.
        cv2.imshow('PartyPi', self.frame)

        # if not self.calibrated and self.tickcount == 10:
        #     self.t1 = time.clock() - t0
        #     print "t1", self.t1
        #     self.calibrated = True

    def level1(self):
        """
        Display emotion prompts, upload image, and display results.
        """
        self.showBegin = False
        self.captureFrame()
        self.tickcount += 1
        if self.raspberry:
            self.tickcount += 1
        timer = int(self.tickcount * self.redfactor)
        self.promptEmotion()

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
            self.level = 2
            self.click_point_y = None

        # Draw the count "3..".
        if self.currCount:
            self.overlay = self.frame.copy()
            cv2.rectangle(self.overlay, (0, int(self.screenheight * (4. / 5))),
                          (self.screenwidth, self.screenheight), (224, 23, 101), -1)
            cv2.addWeighted(self.overlay, self.opacity, self.frame,
                            1 - self.opacity, 0, self.frame)
            cv2.putText(self.frame, str(self.currCount), (int(self.countx), int(
                self.screenheight * (7. / 8))), self.font, 1.0, (255, 255, 255), 2)

        # Draw other text and flash on screen.
        textSize = 2.8 if self.raspberry else 3.6
        if self.showBegin:
            cv2.putText(self.frame, "Begin!", (self.screenwidth / 3, self.screenheight / 2),
                        self.font, textSize, (255, 255, 255), 2)
        elif self.flashon:
            cv2.rectangle(self.frame, (0, 0), (self.screenwidth,
                                               self.screenheight), (255, 255, 255), -1)
        if self.showAnalyzing:
            textSize = 0.7 if self.raspberry else 1.7
            self.addText(self.frame, self.analyzingLabels[self.currentAnalLabel % len(self.analyzingLabels)], (
                self.screenwidth / 5, self.screenheight / 4 + 30), textSize, color=(224, 23, 101))
            self.drawChristmasLogo(self.frame)
        # Display image.
        self.addText(self.frame, brand, ((self.screenwidth / 5) * 4,
                                         self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        cv2.imshow('PartyPi', self.frame)

        if self.photoMode and self.startProcess:
            print "take photo"
            self.takePhoto()

    def level2(self):
        """
        Reset game with face detection.
        """
        self.tickcount += 1
        self.currentAnalLabel += 1
        self.captureFrame()

        if self.raspberry:
            self.tickcount += 1

        # overlay = self.photo.copy()
        # if self.currPosY >= self.screenheight * (4. / 5) and self.currPosY < self.screenheight:
        #     cv2.rectangle(overlay, (0, int(self.screenheight * (3. / 4))),
        #                   (self.screenwidth, self.screenheight), (224, 23, 101), -1)
        #     cv2.addWeighted(overlay, self.opacity, self.photo,
        #                     1 - self.opacity, 0, self.frame)

        # Select mode with mouse button.
        if self.click_point_y > self.screenheight - self.playSize[0] and self.click_point_x > self.screenwidth - self.playSize[1]:
            self.reset()

        # cv2.putText(self.photo, "[Click to play again]", (self.screenwidth / 2, int(
        # self.screenheight * (6. / 7))), self.font, 0.7, (62, 184, 144), 2)

        # if self.tickcount % 5 == 0:
        if self.faceSelect:
            faces = self.findFaces(self.frame)
            # else:
            # faces = []
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
                    # self.playIcon = self.playIcon3.copy()
                    # Timer function useful when checking for faces in every frame
                    # if self.raspberry:
                    #     self.pretimer = 10
                    # else:
                    #     self.pretimer = 100
                    # if not self.pretimer:
                    #     self.reset()
                    # else:
                    #     self.pretimer -= 1
                    self.reset()

            # Show live image in corner.
            self.photo[self.screenheight - self.easySize[0]:self.screenheight, self.screenwidth - self.easySize[0]:self.screenwidth] = self.frame[
                self.screenheight - self.easySize[1]: self.screenheight, self.screenwidth - self.easySize[1]: self.screenwidth]

        self.overlay = self.photo.copy()

        # Show 'Play Again'.
        self.overlay[self.screenheight - self.playSize[1]: self.screenheight, self.screenwidth - self.playSize[1]: self.screenwidth] = self.playIcon[
            0: self.playSize[1], 0: self.playSize[0]]

        # Blend photo with overlay.
        cv2.addWeighted(self.overlay, self.opacity, self.photo,
                        1 - self.opacity, 0, self.photo)

        # Draw logo or title.
        self.addText(self.photo, brand, ((self.screenwidth / 5) * 4,
                                         self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        self.drawChristmasLogo(self.photo)
        # self.drawHat(self.photo, faces)
        cv2.imshow('PartyPi', self.photo)

    def mouse(self, event, x, y, flags, param):
        """
        Listen for mouse.
        """

        if event == cv2.EVENT_MOUSEMOVE:
            self.currPosX, self.currPosY = x, y
            # print "curposX,Y", x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.click_point_x, self.click_point_y = x, y
            # print "x,y", x, y

        elif event == cv2.EVENT_RBUTTONUP:
            self.click_point_right_x, self.click_point_right_y = x, y

    def reset(self):
        """
        Reset to beginning state.
        """
        self.level = 0
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.click_point_right_x = None
        self.currEmotion = self.emotions[1]
        self.result = []
        self.tickcount = 0
        self.static = False
        # self.playIcon = self.playIconOriginal

    def selectMode(self, faces):

        # Draw a rectangle around the faces.
        for (x, y, w, h) in faces:

            # Select easy mode with face
            if x + w < self.easySize[1] and y > self.screenheight - self.easySize[0]:
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = True
                    self.modeLoadCount = 0
                    self.level = 1
                    self.tickcount = 0

            # Select hard mode with face.
            elif x + w > (self.screenwidth - self.hardSize[1]) and y > (self.screenheight - self.hardSize[0]):
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = False
                    self.modeLoadCount = 0
                    self.level = 1
                    self.tickcount = 0

        # Draw easy mode selection box.
        if not self.raspberry:
            self.overlay[self.screenheight - self.easySize[0]:self.screenheight,
                         0:self.easySize[1]] = self.easyIcon

            self.overlay[self.screenheight - self.hardSize[0]:self.screenheight,
                         self.screenwidth - self.hardSize[1]:self.screenwidth] = self.hardIcon
        cv2.addWeighted(self.overlay, self.opacity, self.frame,
                        1 - self.opacity, 0, self.frame)

    def drawChristmasLogo(self, frame):
        if self.screenheight < 700:
            y0 = 0
        else:
            y0 = (self.screenheight / 7) + 0
        y1 = y0 + self.christmas.shape[0]
        if self.screenwidth < 700:
            x0 = 0
        else:
            x0 = 2 * self.screenwidth / 3
        x1 = x0 + self.christmas.shape[1]

        # Remove black background from png image.
        for c in range(0, 3):
            xmasSlice = self.christmas[:, :, c] * \
                (self.christmas[:, :, 3] / 255.0)
            backgroundSlice = frame[y0:y1, x0:x1, c] * \
                (1.0 - self.christmas[:, :, 3] / 255.0)
            frame[y0:y1, x0:x1, c] = xmasSlice + backgroundSlice

    def drawHat(self, frame, faces):
        hat = self.hat.copy()
        hatHeight = hat.shape[0]
        hatWidth = hat.shape[1]
        hat = cv2.resize(hat, (hat.shape[1] * 2, hat.shape[0] * 2))
        hatHeight = hat.shape[0]
        hatWidth = hat.shape[1]
        hatAlignY = 40  # Number of pixels down to move hat.
        hatAlignX = 0
        xOffset = 20  # Number of pixels left to move hat.
        yOffset = 30
        # Extra offset for width and height scaling.
        wOffset = 80
        hOffset = 40

        # TODO: Find out why this doesn't work as expected.
        if self.raspberry:
            wOffset = 40
            hOffset = 30
            yOffset = 20
            xOffset = 10

        for (x, y, w, h) in faces:
            hatx0 = haty0 = 0
            hatx1 = hatWidth
            haty1 = hatHeight

            # Scale hat respective to face width.
            if w > hatWidth:
                hatScale = float(w) / float(hatWidth)
                hat = cv2.resize(
                    hat, (int(hatScale * hatWidth) + wOffset, int(hatScale * hatHeight) + hOffset))
            else:
                hatScale = float(w) / float(hatWidth)
                hat = cv2.resize(
                    hat, (int(hatScale * hatWidth) + wOffset, int(hatScale * hatHeight) + hOffset))

            # Adjust position of hat in frame with respect to face.
            # y: Face rectangle top-left corner.
            # y0: Height of hat in frame.
            # yOffset: Number of pixels to move hat up.
            y0 = y - hat.shape[0] + yOffset

            # Allow clipping.
            if y0 < 0:
                haty0 = abs(y0)
                y0 = 0

            y1 = y0 + hat.shape[0]

            x0 = x - xOffset

            if x0 < 0:
                hatx0 = abs(x0)
                x0 = 0

            # Allow clipping.
            x1 = x0 + hat.shape[1]

            if x1 > self.screenwidth:
                x1 = self.screenwidth
                hatx1 = x1 - self.screenwidth

            if y1 > self.screenheight:
                y1 = self.screenheight
                haty1 = y1 - self.screenheight

            if x0 < 0 or y0 < 0 or x1 > self.screenwidth or y1 > self.screenheight:
                pass
            else:
                # Remove black background from png file.
                for c in range(0, 3):
                    # hatSlice = hat[hatx0:hatx1, haty0:haty1, c] * \
                    #     (hat[haty0:haty1, hatx0:hatx1, 3] / 255.0)
                    # backgroundSlice = frame[y0:y1, x0:x1, c] * \
                    #     (1.0 - hat[:, :, 3] / 255.0)
                    # print hatSlice.shape, backgroundSlice.shape, frame.shape, hat.shape, hatx0, hatx1, haty0, haty1
                    # frame[y0:y1, x0:x1, c] = hatSlice + backgroundSlice
                    if hat[:, :, c].shape == frame[y0:y1, x0:x1, c].shape:
                        hatSlice = hat[:, :, c] * \
                            (hat[:, :, 3] / 255.0)
                        backgroundSlice = frame[
                            y0:y1, x0:x1, c] * (1.0 - hat[:, :, 3] / 255.0)
                        frame[y0:y1, x0:x1, c] = hatSlice + backgroundSlice

    def addText(self, frame, text, origin, size=1.0, color=(255, 255, 255), thickness=1):
        """
        Put text on current frame.
        """
        cv2.putText(frame, text, origin,
                    self.font, size, color, 2)

    def captureFrame(self):
        """
        Capture frame-by-frame.
        """

        if not self.piCam:
            ret, frame = self.cam.read()
            self.frame = cv2.flip(frame, 1)

        self.overlay = self.frame.copy()

    def findFaces(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            flags=0)
        return faces

    def takePhoto(self):
        """
        Take photo and prepare to write, then send to PyImgur.
        """
        imagePath = self.getImagePath()

        # Get faces for Christmas hat.
        faces = self.findFaces(self.frame)

        # If internet connection is poor, use black and white image.
        if self.blackAndWhite:
            bwphoto = cv2.cvtColor(self.photo, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(imagePpath, bwphoto)
            self.result = self.uploader.upload_img(imagePath)
        else:
            self.addText(self.photo, brand, ((self.screenwidth / 5) * 4,
                                             self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
            self.drawChristmasLogo(self.photo)
            self.drawHat(self.photo, faces)
            cv2.imwrite(imagePath, self.photo)
        self.result = self.uploader.upload_img(imagePath)
        self.display()

    def getImagePath(self):
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

    def display(self):
        """
        Display results of game on screen, with winner and scores for emotions.
        """
        scores = []
        maxfirstemo = None
        maxsecondemo = None
        firstEmotion = None

        if self.result:  # If faces present.

            # Get lists of player points.
            firstEmoList = [
                (round(x['scores'][self.currEmotion] * 100)) for x in self.result]
            secondEmoList = [(round(
                x['scores'][self.secCurrEmotion] * 100)) for x in self.result]

            # Compute the scores into `scoresList`.
            scoresList = []
            if self.easyMode:  # Easy mode is points for first emotion.
                scoresList = firstEmoList  # playerNumber, scores
            # Hard mode scores are product of points of both emotions.
            else:
                for i in range(len(firstEmoList)):
                    scoresList.append(
                        (firstEmoList[i] + 1) * (secondEmoList[i] + 1))
            print "scoresList:", scoresList
            textSize = 0.5 if self.raspberry else 0.8
            # Draw the scores for the faces.
            for idx, currFace in enumerate(self.result):
                faceRectangle = currFace['faceRectangle']

                # Draw rectangles over all faces
                # cv2.rectangle(self.photo, (faceRectangle['left'], faceRectangle['top']),
                #               (faceRectangle['left'] + faceRectangle['width'], faceRectangle['top'] +
                # faceRectangle['height']),
                # color=random.choice(self.colors[1]), thickness=4)

                # Get points for first emotion.
                firstEmotion = firstEmoList[idx]
                secEmotion = secondEmoList[idx]

                # Format points.
                if firstEmotion == 1:
                    textToWrite = "%i point: %s" % (
                        firstEmotion, self.currEmotion)
                else:
                    textToWrite = "%i points: %s" % (
                        firstEmotion, self.currEmotion)
                if secEmotion == 1:
                    secondLine = "%i point: %s" % (
                        secEmotion, self.secCurrEmotion)
                else:
                    secondLine = "%i points: %s" % (
                        secEmotion, self.secCurrEmotion)

                # Display points.
                scoreHeightOffset = 10 if self.easyMode else 40
                cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                    'top'] - scoreHeightOffset), self.font, textSize, (232, 167, 35), 2)

                if not self.easyMode:
                    cv2.putText(self.photo, secondLine, (faceRectangle['left'], faceRectangle[
                                'top'] - 10), self.font, textSize, (232, 167, 35), 2)

            # Display 'Winner: ' above player with highest score.
            oneWinner = True
            finalScores = scoresList
            winner = finalScores.index(max(finalScores))
            maxScore = max(finalScores)

            # Multiple winners - tie breaker.
            if finalScores.count(maxScore) > 1:
                print "Multiple winners!"
                oneWinner = False
                tiedWinners = []
                for ind, i in enumerate(finalScores):
                    if i == maxScore:
                        tiedWinners.append(ind)

            print "Scores:", finalScores, "Winner:", winner

            # Identify winner's face.
            firstRectLeft = self.result[winner]['faceRectangle']['left']
            firstRectTop = self.result[winner]['faceRectangle']['top']
            if oneWinner:
                tiedTextHeightOffset = 40 if self.easyMode else 70
                cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - tiedTextHeightOffset),
                            self.font, textSize, (232, 167, 35), 2)
            else:
                tiedTextHeightOffset = 40 if self.easyMode else 70
                print "tiedWinners:", tiedWinners
                for winner in tiedWinners:
                    # FIXME: show both
                    firstRectLeft = self.result[
                        winner]['faceRectangle']['left']
                    firstRectTop = self.result[winner]['faceRectangle']['top']
                    cv2.putText(self.photo, "Tied: ", (firstRectLeft, firstRectTop - tiedTextHeightOffset),
                                self.font, textSize, (232, 167, 35), 2)

        else:
            print "No results found."

    def promptEmotion(self):
        """
        Display prompt for emotion on screen.
        """
        textSize = 1.0 if self.raspberry else 1.2
        width = self.screenwidth / 5 if self.easyMode else 10
        cv2.putText(self.frame, "Show " + self.randomEmotion() + '_',
                    (width, 3 * (self.screenheight / 4)), self.font, textSize, (255, 255, 255), 2)

    def randomEmotion(self):
        """
        Pick a random emotion from list of emotions.
        """
        if self.tickcount * self.redfactor > 30 or self.static:
            self.static = True
            emotionString = str(
                self.currEmotion) if self.easyMode else self.currEmotion + '+' + self.secCurrEmotion
            return emotionString
        else:
            self.currEmotion = random.choice(self.emotions)
            randnum = (self.emotions.index(self.currEmotion) +
                       random.choice(range(1, 7))) % 8
            self.secCurrEmotion = self.emotions[randnum]
            if self.easyMode:
                return self.currEmotion
            else:
                return self.currEmotion + '+' + self.secCurrEmotion

    def listenForEnd(self, keypress):
        """
        Listen for 'q', left, or right keys to end game.
        """
        if keypress != 255:
            print(keypress)
            if keypress == ord('q'):  # 'q' pressed to quit
                print "Escape key entered"
                self.looping = False
                self.endGame()
            elif self.level == 0:
                if keypress == 81 or keypress == 2:  # left
                    self.easyMode = True
                    self.tickcount = 0
                    self.level = 1
                elif keypress == 83 or keypress == 3:  # right
                    self.easyMode = False
                    self.tickcount = 0
                    self.level = 1
            elif self.level == 2:
                self.reset()

    def endGame(self):
        """
        When everything is done, release the capture.
        """
        if not self.piCam:
            self.cam.release()
            self.addText(self.frame, "Press any key to quit_",
                         (self.screenwidth / 4, self.screenheight / 3))
            # self.presentation(self.frame)
            self.addText(self.frame, brand, ((self.screenwidth / 5) * 4,
                                             self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        else:
            self.piCamera.close()

        cv2.imshow("PartyPi", self.frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """
    Load application.
    """
    # sys.argv[1] = Using piCamera module
    if len(sys.argv) == 2:
        if 'picam' or '-p' in sys.argv[1]:
            application = PartyPi(True)
        else:
            print "Load default settings"
            application = PartyPi()
    elif len(sys.argv) == 3:
        if 'x' in sys.argv[2]:
            res = sys.argv[2]
            w = res.split('x')[0]
            h = res.split('x')[1]
            application = PartyPi(True, (w, h))
        else:
            print "Load default settings"
            application = PartyPi(True)
    else:
        application = PartyPi()


if __name__ == '__main__':
    main()
