#!/usr/lib/env python
import cv2
import uuid
from emotionpi import emotion_api
import pyimgur
import numpy as np
import operator
import random
import os
import time
import sys


class PartyPi():

    def __init__(self):

        self.level = 0
        self.looping = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.emotions = ['anger', 'contempt', 'disgust',
                         'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.photo = cv2.imread('img_1.png')
        # initialize the camera and grab a reference to the raw camera capture
        self.raspberry = True
        if 'raspberry' in os.uname():
            self.raspberry = True
            self.pyIt()
        else:
            self.raspberry = False
            self.cam = cv2.VideoCapture(0)
        self.currEmotion = 'anger'
        self.screenwidth = 1280 / 2
        self.screenheight = 1024 / 2
        self.countx = None
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = -1
        self.click_point_y = -1
        self.tickcount = 0
        self.initPyimgur()
        self.setupGame()

    def pyIt(self):
        self.piCamera = PiCamera()
        # self.piCamera.resolution = (640, 480)
        self.piCamera.resolution = (1280 / 2, 1024 / 2)
        self.piCamera.framerate = 16
        self.rawCapture = PiRGBArray(self.piCamera, size=(1280 / 2, 1024 / 2))
        time.sleep(0.1)

    def setupGame(self):
        # cascPath = "face.xml"
        # self.faceCascade = cv2.CascadeClassifier(cascPath)
        print "Camera initialize"
        if not self.raspberry:
            self.cam.set(3, self.screenwidth)
            self.cam.set(4, self.screenheight)
        self.flashon = False
        self.showAnalyzing = False
        scale = 0.5  # font scale
        self.opacity = 0.4
        self.currCount = None
        self.static = False
        self.photoMode = False
        cv2.namedWindow("PartyPi", 0)
        # cv2.setWindowProperty(
        #     "PartyPi", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PartyPi", cv2.WND_PROP_AUTOSIZE,
        #                       cv2.WINDOW_AUTOSIZE)
        if not self.raspberry:
            cv2.setMouseCallback("PartyPi", self.mouse)
        self.redfactor = 1.
        if self.raspberry:
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
        # Initialize variables and parameters

        self._url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        self._key = '1cc9418279ff4b2683b5050cfa6f3785'
        self._maxNumRetries = 10
        self.result = []

    def gameLoop(self):
        if self.level == 0:
            self.level0()
        elif self.level == 1:
            self.level1()
        elif self.level == 2:
            self.level2()

        # Catch escape key
        keypress = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        if self.raspberry:
            self.rawCapture.truncate(0)

        if keypress != 255:
            print(keypress)
            # if keypress == 32:
            # self.tickcount = 0
            # self.photoMode = True
            # self.photo = takeself.photo()

            if keypress == 113 or keypress == 27:  # 'q' pressed to quit
                print "Escape key entered"
                self.looping = False
                self.endGame()
            elif keypress == 81 and self.level == 0:  # left
                self.easyMode = True
                self.tickcount = 0
                self.level = 1
            elif keypress == 83 and self.level == 0:  # right
                self.easyMode = False
                self.tickcount = 0
                self.level = 1
            if self.level == 2:
                self.reset()

    def level0(self):
        self.tickcount += 1
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
        if self.click_point_x >= 0:
            print "click point x is greater than 0"
            if self.click_point_x < self.screenwidth / 2:
                self.easyMode = True  # Easy mode selected
            else:
                self.easyMode = False  # Hard mode selected
            self.tickcount = 0
            self.level = 1
        cv2.addWeighted(self.overlay, self.opacity, self.frame,
                        1 - self.opacity, 0, self.frame)
        # Display frame
        self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        cv2.imshow('PartyPi', self.frame)

    def level1(self):
        self.showBegin = False
        self.captureFrame()
        self.tickcount += 1
        if self.raspberry:
            self.tickcount += 1
        timer = int(self.tickcount * self.redfactor)
        print "tickcount:", self.tickcount, timer

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
                if timer >= 133 and timer <= 134:
                    self.photoMode = True
                    self.photo = self.frame.copy()
                    self.startProcess = True
                else:
                    self.startProcess = False
                    self.showAnalyzing = True

        # Else take photo at timer == 136
        else:
            self.startProcess = False
            self.flashon = False
            self.showAnalyzing = False
            self.level = 2
            self.click_point_y = -1

        # Draw the count "3.."
        if self.currCount:
            self.overlay = self.frame.copy()
            cv2.rectangle(self.overlay, (0, int(self.screenheight * (4. / 5))),
                          (self.screenwidth, self.screenheight), (224, 23, 101), -1)
            cv2.addWeighted(self.overlay, self.opacity, self.frame,
                            1 - self.opacity, 0, self.frame)
            cv2.putText(self.frame, str(self.currCount), (int(self.countx), int(
                self.screenheight * (7. / 8))), self.font, 1.0, (255, 255, 255), 2)

        # Draw other text and flash on screen
        if self.showBegin:
            cv2.putText(self.frame, "Begin!", (self.screenwidth / 3, self.screenheight / 2),
                        self.font, 2.0, (255, 255, 255), 2)
        elif self.flashon:
            cv2.rectangle(self.frame, (0, 0), (self.screenwidth,
                                               self.screenheight), (255, 255, 255), -1)

        if self.showAnalyzing:
            self.addText(self.frame, "Analyzing...", (self.screenwidth / 5,
                                                      self.screenheight / 4), size=2.2, color=(224, 23, 101))
        # Display image
        self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        cv2.imshow('PartyPi', self.frame)

        if self.photoMode and self.startProcess:
            print "take photo"
            self.takePhoto()

    def level2(self):
        self.tickcount += 1
        if self.raspberry:
            self.tickcount += 1
        overlay = self.photo.copy()
        if self.currPosY >= self.screenheight * (4. / 5) and self.currPosY < self.screenheight:
            cv2.rectangle(overlay, (0, int(self.screenheight * (3. / 4))),
                          (self.screenwidth, self.screenheight), (224, 23, 101), -1)
            cv2.addWeighted(overlay, self.opacity, self.photo,
                            1 - self.opacity, 0, self.frame)
        if self.click_point_y >= self.screenheight * (4. / 5) and self.click_point_y < self.screenheight:
            self.reset()
        cv2.putText(self.photo, "[Click to play again]", (self.screenwidth / 2, int(
            self.screenheight * (6. / 7))), self.font, 0.7, (62, 184, 144), 2)

        self.addText(self.photo, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                    self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        cv2.imshow('PartyPi', self.photo)

    def mouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE:
            self.currPosX, self.currPosY = x, y
            # print "curposX,Y", x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.click_point_x, self.click_point_y = x, y
            # print "x,y", x, y

    def reset(self):
        self.level = 0
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.currEmotion = 'happiness'
        self.result = []
        self.tickcount = 0
        self.static = False

    def addText(self, frame, text, origin, size=1.0, color=(255, 255, 255), thickness=1):
        cv2.putText(frame, text, origin,
                    self.font, size, color, 2)

    def captureFrame(self):
        # Capture frame-by-frame
        if not self.raspberry:
            ret, frame = self.cam.read()
            self.frame = cv2.flip(frame, 1)
        self.overlay = self.frame.copy()

    def takePhoto(self):
        img_nr = self.get_last_image_nr()
        self.imagepath = 'img/' + str(self.img_name) + \
            str(img_nr) + str(self.img_end)
        cv2.imwrite(self.imagepath, self.photo)
        img_nr += 1
        self.upload_img()

    def upload_img(self):
        print "Initate upload"
        CLIENT_ID = "525d3ebc1147459"

        im = pyimgur.Imgur(CLIENT_ID)
        uploaded_image = im.upload_image(
            self.imagepath, title="Uploaded with PyImgur")
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
        scores = []
        maxfirstemo = None
        maxsecondemo = None
        firstEmotion = None
        if self.tickcount % 20 == 0:
            print "Load Display ", self.tickcount
        if self.result:  # if faces present

            for currFace in self.result:
                faceRectangle = currFace['faceRectangle']
                cv2.rectangle(self.photo, (faceRectangle['left'], faceRectangle['top']),
                              (faceRectangle['left'] + faceRectangle['width'], faceRectangle['top'] +
                               faceRectangle['height']), color=(255, 255, 0), thickness=5)

            for idx, currFace in enumerate(self.result):
                faceRectangle = currFace['faceRectangle']
                # self.currEmotion = max(currFace['scores'].items(), key=operator.itemgetter(1))[0]
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
            ##        cv2.rectangle(overlay,(0,int(self.screenheight*(4./5))),(self.screenwidth,self.screenheight),(224,23,101), -1)
            ##        cv2.addWeighted(overlay,self.opacity,self.photo,1-self.opacity, 0, self.frame)
            # else:
            # pass
        else:
            print "no results found"

    def promptEmotion(self):
        if self.easyMode:
            cv2.putText(self.frame, "Show " + self.randomEmotion() + '_',
                        (self.screenwidth / 5, 3 * (self.screenheight / 4)), self.font, 1.0, (255, 255, 255), 2)
        else:
            self.addText(self.frame, "Show " + self.randomEmotion() +
                         '_', (10, 3 * self.screenheight / 4))

    def randomEmotion(self):
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

    def endGame(self):
        # When everything is done, release the capture
        if not self.raspberry:
            self.cam.release()
        if self.photoMode:
            self.addText(self.photo, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                        self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
            cv2.imshow("PartyPi", self.photo)
        else:
            self.addText(self.frame, "Press any key to quit_",
                         (self.screenwidth / 4, self.screenheight / 3))
            # self.presentation(self.frame)
            self.addText(self.frame, "PartyPi v0.0.2", ((self.screenwidth / 5) * 4,
                                                        self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
            cv2.imshow("PartyPi", self.frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    import os
    if 'raspberry' in os.uname():
        from picamera.array import PiRGBArray
        from picamera import PiCamera
    application = PartyPi()


if __name__ == '__main__':
    main()
