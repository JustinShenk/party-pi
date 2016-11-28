#!/usr/lib/env python
import sys
import os
import cv2
from emotionpi import emotion_api
import pyimgur
import random
import time
import numpy as np


class PartyPi(object):

    def __init__(self, piCam=False, resolution=(1280 / 2, 1024 / 2), windowSize=(1200, 1024)):
        self.piCam = piCam
        print self.piCam
        self.level = 0
        self.looping = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.emotions = ['anger', 'contempt', 'disgust',
                         'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.photo = cv2.imread('img_1.png')
        self.resolution = resolution
        self.screenwidth, self.screenheight = self.windowSize

        # Setup for Raspberry Pi.
        if 'raspberrypi' in os.uname():
            print "PartyPi v0.0.2 for Raspberry Pi"
            self.raspberry = True

            # Set up picamera module.
            if self.piCam:
                self.pyIt()
            else:
                self.cam = cv2.VideoCapture(0)
                _, self.frame = self.cam.read()
                # self.cam.set(3, self.screenwidth)
                # self.cam.set(4, self.screenheight)
        else:
            self.raspberry = False
            self.cam = cv2.VideoCapture(0)
            self.cam.set(3, self.screenwidth)
            self.cam.set(4, self.screenheight)
            _, self.frame = self.cam.read()

        self.screenwidth, self.screenheight = self.frame.shape[:2]
        print "first screenheight:", self.screenheight, self.screenwidth, self.frame.shape
        self.currEmotion = 'anger'
        self.countx = None
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.calibrated = False
        self.tickcount = 0
        self.modeLoadCount = 0
        self.initPyimgur()
        self.setupGame()

    def pyIt(self):
        """
        Set up piCamera for rasbperry pi camera module.
        """
        from picamera import PiCamera
        from picamera.array import PiRGBArray
        self.piCamera = PiCamera()
        # self.piCamera.resolution = (640, 480)
        self.piCamera.resolution = (self.resolution[0], self.resolution[1])
        self.screenwidth, self.screenheight = self.piCamera.resolution
        # self.piCamera.framerate = 24
        self.rawCapture = PiRGBArray(
            self.piCamera)

        self.frame = np.empty(
            (self.screenheight, self.screenwidth, 3), dtype=np.uint8)
        time.sleep(0.1)

    def setupGame(self):
        """
        Set up icons and face cascade.
        """
        self.easyIcon = cv2.imread('easy.png')
        self.hardIcon = cv2.imread('hard.png')
        self.playIcon = cv2.imread('playagain.png')
        self.playIconOriginal = self.playIcon.copy()
        self.playIcon1 = cv2.imread('playagain1.png')
        self.playIcon2 = cv2.imread('playagain2.png')
        self.playIcon3 = cv2.imread('playagain3.png')
        self.easySize = self.hardSize = self.easyIcon.shape[:2]
        self.playSize = self.playIcon.shape[:2]
        cascPath = "face.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.pretimer = None

        print "Camera initialize"
        # if not self.raspberry:
        #     print "MAC or PC initialize"
        #     self.cam.set(3, self.screenwidth)
        #     self.cam.set(4, self.screenheight)
        self.flashon = False
        self.showAnalyzing = False
        self.opacity = 0.4
        self.currCount = None
        self.static = False
        self.photoMode = False
        cv2.namedWindow("PartyPi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PartyPi", self.windowSize[0], self.windowSize[1])
        # Returns - TypeError: Required argument 'prop_value' (pos 3) not found
        # cv2.setWindowProperty(
        #     "PartyPi", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PartyPi", cv2.WND_PROP_AUTOSIZE,
        #                       cv2.WINDOW_AUTOSIZE)
        if not self.raspberry:
            cv2.setMouseCallback("PartyPi", self.mouse)
        self.redfactor = 1.
        print "self.piCam:", self.piCam
        if self.piCam == True:
            print "self.piCam:", self.piCam
            # capture frames from the camera
            for _frame in self.piCamera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
                # grab the raw NumPy array representing the image, then initialize the timestamp
                # and occupied/unoccupied text
                self.frame = cv2.flip(_frame.array, 1)
                self.gameLoop()
        else:
            while self.looping:
                self.gameLoop()

    def initPyimgur(self):
        """
        Initialize variables and parameters for PyImgur.
        """
        self._url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        self._key = '1cc9418279ff4b2683b5050cfa6f3785'
        self._maxNumRetries = 10
        CLIENT_ID = "525d3ebc1147459"
        CLIENT_SECRET = "75b8f9b449462150f374ae68f10154f2f392aa9b"
        ALBUM_ID = "3mdlF"
        self.im = pyimgur.Imgur(CLIENT_ID)

        # TODO: Turn on album uploading.
        # self.im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)
        # self.im.change_authentication(
        # refresh_token="814bed15fbea91dbc6131205f881fc45f8ee0715")
        # self.im.refresh_access_token()
        # user = im.get_user(spacemaker)
        # self.album = self.im.get_album(ALBUM_ID)
        self.result = []

    def gameLoop(self):
        """
        Start the game loop. Listen for escape key.
        """
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
        # if not self.calibrated and self.tickcount == 10:
        #     t0 = time.clock()

        if self.raspberry:
            self.tickcount += 1
        self.captureFrame()

        self.addText(self.frame, "Easy", (self.screenwidth / 8,
                                          (self.screenheight * 3) / 4), size=3)
        self.addText(self.frame, "Hard", (self.screenwidth / 2,
                                          (self.screenheight * 3) / 4), size=3)
        if self.currPosX and self.currPosX < self.screenwidth / 2:
            cv2.rectangle(self.overlay, (0, 0), (self.screenwidth / 2,
                                                 self.screenheight), (211, 211, 211), -1)
        else:
            cv2.rectangle(self.overlay, (self.screenwidth / 2, 0),
                          (self.screenwidth, self.screenheight), (211, 211, 211), -1)
        if self.click_point_x:
            if self.click_point_x < self.screenwidth / 2:
                self.easyMode = True  # Easy mode selected
            else:
                self.easyMode = False  # Hard mode selected
            self.tickcount = 0
            self.level = 1
            self.click_point_x = None

        # Draw faces
        if self.tickcount % 5 == 0:
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                frame_gray,
                scaleFactor=1.1,
                minNeighbors=15,
                minSize=(70, 70),
                #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                flags=0
            )
        else:
            faces = []
        # Draw a rectangle around the faces

        for (x, y, w, h) in faces:
            cv2.rectangle(self.frame, (x, y),
                          (x + w, y + h), (0, 0, 255), 2)
            # Select easy mode with face
            if x + w < self.easySize[1] and y > self.screenheight - self.easySize[0]:
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = True
                    self.modeLoadCount = 0
                    self.level = 1
                    self.tickcount = 0
            # Select hard mode with face
            elif x + w > (self.screenwidth - self.hardSize[1]) and y > (self.screenheight - self.hardSize[0]):
                self.modeLoadCount += 1
                if self.modeLoadCount is 20:
                    self.easyMode = False
                    self.modeLoadCount = 0
                    self.level = 1
                    self.tickcount = 0

            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = img[y:y + h, x:x + w]

        # Draw easy mode selection box.

        # FIXME: Uncomment following lines
        if not self.raspberry:
            self.overlay[self.screenheight - self.easySize[0]:self.screenheight,
                         0:self.easySize[1]] = self.easyIcon

            self.overlay[self.screenheight - self.hardSize[0]:self.screenheight,
                         self.screenwidth - self.hardSize[1]:self.screenwidth] = self.hardIcon
        cv2.addWeighted(self.overlay, self.opacity, self.frame,
                        1 - self.opacity, 0, self.frame)
        # Display frame
        self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
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
        if self.showBegin:
            cv2.putText(self.frame, "Begin!", (self.screenwidth / 3, self.screenheight / 2),
                        self.font, 2.0, (255, 255, 255), 2)
        elif self.flashon:
            cv2.rectangle(self.frame, (0, 0), (self.screenwidth,
                                               self.screenheight), (255, 255, 255), -1)
        if self.showAnalyzing:
            self.addText(self.frame, "Analyzing...", (self.screenwidth / 5,
                                                      self.screenheight / 4), size=2.2, color=(224, 23, 101))
        # Display image.
        self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
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
        self.captureFrame()
        if self.raspberry:
            self.tickcount += 1
        # overlay = self.photo.copy()
        # if self.currPosY >= self.screenheight * (4. / 5) and self.currPosY < self.screenheight:
        #     cv2.rectangle(overlay, (0, int(self.screenheight * (3. / 4))),
        #                   (self.screenwidth, self.screenheight), (224, 23, 101), -1)
        #     cv2.addWeighted(overlay, self.opacity, self.photo,
        #                     1 - self.opacity, 0, self.frame)
        if self.click_point_y > self.screenheight - self.playSize[0] and self.click_point_x > self.screenwidth - self.playSize[1]:
            self.reset()

        # cv2.putText(self.photo, "[Click to play again]", (self.screenwidth / 2, int(
        # self.screenheight * (6. / 7))), self.font, 0.7, (62, 184, 144), 2)

        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        if self.tickcount % 5 == 0:
            faces = self.faceCascade.detectMultiScale(
                frame_gray, scaleFactor=1.1, minNeighbors=15, minSize=(70, 70),
                #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                flags=0
            )
        else:
            faces = []
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

        # Show live image
        self.photo[self.screenheight - self.easySize[0]:self.screenheight, self.screenwidth - self.easySize[0]:self.screenwidth] = self.frame[
            self.screenheight - self.easySize[1]:self.screenheight, self.screenwidth - self.easySize[1]:self.screenwidth]

        self.overlay = self.photo.copy()
        # Show 'Play Again'
        self.overlay[self.screenheight - self.playSize[1]:self.screenheight, self.screenwidth - self.playSize[1]:self.screenwidth] = self.playIcon[
            0:self.playSize[1], 0:self.playSize[0]]

        cv2.addWeighted(self.overlay, self.opacity, self.photo,
                        1 - self.opacity, 0, self.photo)
        self.addText(self.photo, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
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

    def reset(self):
        """
        Reset to beginning.
        """
        self.level = 0
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.currEmotion = 'happiness'
        self.result = []
        self.tickcount = 0
        self.static = False
        self.playIcon = self.playIconOriginal

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

    def takePhoto(self):
        """
        Take photo and prepare to write, then send to PyImgur.
        """
        img_nr = self.get_last_image_nr()
        self.imagepath = 'img/' + str(self.img_name) + \
            str(img_nr) + str(self.img_end)
        bwphoto = cv2.cvtColor(self.photo, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.imagepath, bwphoto)
        img_nr += 1
        self.upload_img()

    def upload_img(self):
        """
        Send image to PyImgur.
        """
        print "Initate upload"

        # try:
        #     pass
        # except Exception as e:
        #     raise
        # else:
        #     pass
        # finally:
        #     pass
        uploaded_image = self.im.upload_image(
            self.imagepath, title="Uploaded with PyImgur")

        # TODO: Turn on album uploading
        # uploaded_image = self.im.upload_image(
        #     self.imagepath, title="Uploaded with PyImgur", album=self.album)
        print(uploaded_image.title)
        print(uploaded_image.link)
        print(uploaded_image.size)
        print(uploaded_image.type)

        print "Analyze image"
        data = emotion_api(uploaded_image.link)
        self.result = data
        self.display()

    def get_last_image_nr(self):
        self.img_name = 'img_'
        img_nr = 0
        self.img_end = ".png"
        nr = 0
        for file in os.listdir(os.getcwd() + '/img'):
            if file.endswith(self.img_end):
                file = file.replace(self.img_name, '')
                file = file.replace(self.img_end, '')
                print file
                file_nr = int(file)
                nr = max(nr, file_nr)
        return nr + 1

    def display(self):
        """
        Display results of game on screen, with winner and scores for emotions.
        """
        scores = []
        maxfirstemo = None
        maxsecondemo = None
        firstEmotion = None
        if self.tickcount % 20 == 0:
            print "Load Display ", self.tickcount
        if self.result:  # if faces present
            for idx, currFace in enumerate(self.result):
                faceRectangle = currFace['faceRectangle']
                cv2.rectangle(self.photo, (faceRectangle['left'], faceRectangle['top']),
                              (faceRectangle['left'] + faceRectangle['width'], faceRectangle['top'] +
                               faceRectangle['height']), color=(255, 255, 0), thickness=4)
                # self.currEmotion = max(currFace['scores'].items(),
                # key=operator.itemgetter(1))[0]
                firstEmotion = currFace['scores'][self.currEmotion] * 100
                secEmotion = currFace['scores'][self.secCurrEmotion] * 100
                # scores.append((firstEmotion+1)+(secEmotion+1))
                scores.append((firstEmotion + 1) * (secEmotion + 1) * 400)
                # if firstEmotion > maxfirstemo:
                #   maxfirstemo = idx
                # if secEmotion > maxsecondemo:
                #   maxsecondemo = idx
                if firstEmotion > 0:
                    textToWrite = "%i points: %s" % (
                        firstEmotion, self.currEmotion)
                else:
                    textToWrite = "%i point: %s" % (
                        firstEmotion, self.currEmotion)
                if secEmotion > 0:
                    secondLine = "%i points: %s" % (
                        secEmotion, self.secCurrEmotion)
                else:
                    secondLine = "%i point: %s" % (
                        secEmotion, self.secCurrEmotion)
                if self.easyMode:
                    cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                                'top'] - 10), self.font, 0.8, (232, 167, 35), 2)
                else:
                    cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                                'top'] - 40), self.font, 0.8, (232, 167, 35), 2)
                    cv2.putText(self.photo, secondLine, (faceRectangle['left'], faceRectangle[
                                'top'] - 10), self.font, 0.8, (232, 167, 35), 2)

            if firstEmotion:
                winner = scores.index(max(scores))
                firstRectLeft = self.result[winner]['faceRectangle']['left']
                firstRectTop = self.result[winner]['faceRectangle']['top']
                if self.easyMode:
                    cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - 40),
                                self.font, 0.8, (232, 167, 35), 2)
                else:
                    cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - 70),
                                self.font, 0.8, (232, 167, 35), 2)

            # if self.currPosY >= self.screenheight*(4./5) and self.currPosY < self.screenheight:
            ##
            # cv2.rectangle(overlay,(0,int(self.screenheight*(4./5))),(self.screenwidth,self.screenheight),(224,23,101), -1)
            # cv2.addWeighted(overlay,self.opacity,self.photo,1-self.opacity, 0, self.frame)
            # else:
            # pass
        else:
            print "no results found"

    def promptEmotion(self):
        """
        Display prompt for emotion on screen.
        """
        if self.easyMode:
            cv2.putText(self.frame, "Show " + self.randomEmotion() + '_',
                        (self.screenwidth / 5, 3 * (self.screenheight / 4)), self.font, 1.0, (255, 255, 255), 2)
        else:
            self.addText(self.frame, "Show " + self.randomEmotion() +
                         '_', (10, 3 * self.screenheight / 4))

    def randomEmotion(self):
        """
        Pick a random emotion from list of emotions.
        """
        if self.tickcount * self.redfactor > 30 or self.static:
            self.static = True
            if self.easyMode:
                return str(self.currEmotion)
            else:
                return self.currEmotion + '+' + self.secCurrEmotion
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
            if keypress == 113 or keypress == 27:  # 'q' pressed to quit
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
        When everything is done, release the capture
        """
        if not self.piCam:
            self.cam.release()
            self.addText(self.frame, "Press any key to quit_",
                         (self.screenwidth / 4, self.screenheight / 3))
            # self.presentation(self.frame)
            self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                        self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
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
