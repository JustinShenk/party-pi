#!/ usr/bin/python
import cv2
import uuid
from emotionpi import emotion_api
import pyimgur
import numpy as np
import operator
from Tkinter import *
import random
import os
import sys
# from __future__ import print_function
# import easygui

# Initialize variables and parameters

_url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
_key = '1cc9418279ff4b2683b5050cfa6f3785'
_maxNumRetries = 10
result = []

self.static = False
self.currPosX = -1
self.currPosY = -1
self.countx = 0

# Reset the game state


class PartyPi():

    def __init__(self):
        self.tickcount = 0
        self.showText = False
        self.happiness = 0
        self.emotions = ['anger', 'contempt', 'disgust',
                         'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.photo = cv2.imread('image.png')
        self.facecount = 0
        self.looping = False
        self.self.photoMode = False
        self.rects = []
        self.currCount = None
        self.click_point = []
        self.click_point_x = -1
        self.click_point_y = -1
        self.currLevel = 0
        self.screenwidth = 1280 / 2
        self.screenheight = 1024 / 2
        self.flashon = False
        self.easyMode = True
        self.secEmotion = ''
        self.img_name = 'img_'
        self.img_nr = 0
        self.img_end = ".png"
        self.imagepath = ''
        self.showAnalyzing = False
        self.opacity = 0.4
        self.redfactor = 1
        self.cam = cv2.VideoCapture(0)
        self.img_nr = self.get_last_image_nr()
        self.startGame()
        self.endCapture()

    def startGame(self):
        # Begin main program
        cascPath = "face.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        print "Camera initialize"
        self.cam.set(3, self.screenwidth)
        self.cam.set(4, self.screenheight)
        scale = 0.5  # font scale

        self.looping = True
        # cv2.namedWindow('PartyPi',cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('PartyPi', 0)
        #cv2.setWindowProperty('PartyPi', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        cv2.setMouseCallback('PartyPi', mouse)
        self.gameLoop()

    def gameLoop(self):
        while self.looping:
            self.tickcount += 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Capture self.frame-by-self.frame
            ret, frame = cam.read()
            self.frame = cv2.flip(frame, 1)
            overlay = self.frame.copy()

           # Display the resulting self.frame
           # Display face / text
            self.display(result, self.frame)
            if self.currLevel is 0:
                self.level0()
            elif self.currLevel == 2:
                ##        overlay = self.frame.copy()
                # if self.currPosY >= self.screenheight*(4./5) and self.currPosY < self.screenheight:
                ##            cv2.rectangle(overlay,(0,int(self.screenheight*(3./4))),(self.screenwidth,self.screenheight),(224,23,101), -1)
                ##            cv2.addWeighted(overlay,self.opacity,self.frame,1-self.opacity, 0, self.frame)
                # if self.click_point_y >= self.screenheight*(4./5) and self.click_point_y < self.screenheight:
                # print "Restarting"
                if self.click_point_y > 0:
                    reset()
            if not self.photoMode:
                self.showGame(self.frame)
                if self.flashon:
                    cv2.rectangle(self.frame, (0, 0), (self.screenwidth,
                                                       self.screenheight), (255, 255, 255), -1)
                    if self.showAnalyzing:
                        self.addText(self.frame, "Analyzing...", (self.screenwidth / 5,
                                                                  self.screenheight / 4), size=2.2, color=(224, 23, 101))
                self.presentation(self.frame)
            else:  # self.photo mode is on
                self.showGame(self.photo)
                self.presentation(self.photo)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress != 255:
                print(keypress)
                # if keypress == 32:
                # self.tickcount = 0
                # self.photoMode = True
                # self.photo = takeself.photo()

                if keypress == 113 or 27:  # 'q' pressed to quit
                    print "Escape key entered"
                    self.looping = False
                    break

    def level0(self):
        self.addText(self.frame, "Easy", (self.screenwidth / 8,
                                          (self.screenheight * 3) / 4), size=3)
        self.addText(self.frame, "Hard", (self.screenwidth / 2,
                                          (self.screenheight * 3) / 4), size=3)
        if self.currPosX >= 0 and self.currPosX < self.screenwidth / 2:
            cv2.rectangle(overlay, (0, 0), (self.screenwidth / 2,
                                            self.screenheight), (211, 211, 211), -1)
        else:
            cv2.rectangle(overlay, (self.screenwidth / 2, 0),
                          (self.screenwidth, self.screenheight), (211, 211, 211), -1)
        if self.click_point_x >= 0:
            print "click point x is greater than 0"
            if self.click_point_x < self.screenwidth / 2:
                self.easyMode = True  # Easy mode selected
            else:
                self.easyMode = False  # Hard mode selected
            self.tickcount = 0
            self.currLevel = 1
        cv2.addWeighted(overlay, self.opacity, self.frame,
                        1 - self.opacity, 0, self.frame)

    # Put text on screen

    def addText(self, frame, text, origin, size=1.0, color=(255, 255, 255), thickness=1):
        cv2.putText(frame, text, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

    # Put game title on screen

    def presentation(self):
        self.addText(self.frame, "Feature-match v0.0.1", ((self.screenwidth / 5) * 4,
                                                          self.screenheight / 7), color=(68, 54, 66), size=0.5, thickness=0.5)
        cv2.imshow('Feature-match', self.frame)

    # Check for most recent image number

    def get_last_image_nr(self):
        nr = 0
        for file in os.listdir(os.getcwd() + '/img'):
            if file.endswith(img_end):
                file = file.replace(img_name, '')
                file = file.replace(img_end, '')
                print file
                file_nr = int(file)
                nr = max(nr, file_nr)
        return nr + 1

    # Mouse click and position

    def mouse(self, event, x, y, flags, param):
        global click_point
        global self.click_point_x
        global self.click_point_y
        global self.currPosX
        global self.currPosY

        if event == cv2.EVENT_MOUSEMOVE:
            self.currPosX, self.currPosY = x, y
            # print x,y
        if event == cv2.EVENT_LBUTTONUP:
            self.click_point_x, self.click_point_y = x, y
            # print x,y

    # Capture and save self.photo

    # def takePhoto(self, self.frame):
    #     global img_nr
    #     global imagepath
    #     global self.photo
    #     imagepath = 'img/' + str(img_name) + str(img_nr) + str(img_end)
    #     cv2.imwrite(imagepath, self.frame)
    #     img_nr += 1
    #     self.photo = self.frame.copy()
    #     print "capture key pressed"
    #     upload_img(self.photo)
    #     return self.photo

    # Pick a random emotion

    def randomEmotion(self, self.easyMode):
        if self.tickcount * self.redfactor > 30 or self.static:
            self.static = True
            if self.easyMode:
                return str(self.currEmotion)
            else:
                return self.currEmotion + '+' + self.secCurrEmotion
        else:
            self.currEmotion = random.choice(emotions)
            randnum = (self.emotions.index(self.currEmotion) +
                       random.choice(range(1, 7))) % 8
            self.secCurrEmotion = self.emotions[randnum]
            if self.easyMode:
                return self.currEmotion
            else:
                return self.currEmotion + '+' + self.secCurrEmotion

    # Display game

    def showGame(self, self.frame):
        self.showBegin = False
        if self.currLevel is 1:

            if self.easyMode:
                cv2.putText(self.frame, "Show " + self.randomEmotion(self.easyMode) + '_', (self.screenwidth / 5,
                                                                                            3 * (self.screenheight / 4)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                self.addText(self.frame, "Show " + self.randomEmotion(self.easyMode) +
                             '_', (10, 3 * self.screenheight / 4))
            if self.tickcount * self.redfactor < 70:
                pass
            elif self.tickcount * self.redfactor < 80:
                self.showBegin = True
            elif self.tickcount * self.redfactor < 100:
                pass
            elif self.tickcount * self.redfactor >= 100 and self.tickcount * self.redfactor <= 110:
                self.showBegin = False
                self.currCount = 3
                self.countx = self.screenwidth - (self.screenwidth / 5) * 4
            elif self.tickcount * self.redfactor >= 110 and self.tickcount * self.redfactor < 120:
                self.currCount = 2
                self.countx = self.screenwidth - (self.screenwidth / 5) * 3
            elif self.tickcount * self.redfactor >= 120 and self.tickcount * self.redfactor < 130:
                self.currCount = 1
                self.countx = self.screenwidth - (self.screenwidth / 5) * 2
            elif self.tickcount * self.redfactor >= 130 and self.tickcount * self.redfactor < 135:
                self.flashon = True
                if self.tickcount * self.redfactor > 133:
                    self.showAnalyzing = True
                self.countx = -100  # make it disappear
            else:
                self.flashon = False
                self.countx = -100  # make it disappear
                self.photoMode = True
                self.photo = self.takePhoto(self.frame)
                self.currLevel = 2
                self.click_point_y = -1
            if self.currCount > 0:
                overlay = self.frame.copy()
                cv2.rectangle(overlay, (0, int(self.screenheight * (4. / 5))),
                              (self.screenwidth, self.screenheight), (224, 23, 101), -1)
                cv2.addWeighted(overlay, self.opacity, self.frame,
                                1 - self.opacity, 0, self.frame)
                cv2.putText(self.frame, str(self.currCount), (int(self.countx), int(
                    self.screenheight * (7. / 8))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if self.showBegin:
                cv2.putText(self.frame, "Begin!", (self.screenwidth / 3, self.screenheight / 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

    def reset(self):

        self.static = False
        self.currLevel = 0
        self.currCount = None
        self.countx = 0
        self.photoMode = False
        self.click_point_x = -1
        self.click_point_y = -1
        self.showBegin = False

    def display(self, result, self.frame):
        scores = []
        maxfirstemo = -1
        maxsecondemo = -1
        firstEmotion = -1
        if self.tickcount % 20 == 0:
            print "Load Display ", self.tickcount
        for currFace in result:
            faceRectangle = currFace['faceRectangle']
            cv2.rectangle(self.photo, (faceRectangle['left'], faceRectangle['top']),
                          (faceRectangle['left'] + faceRectangle['width'], faceRectangle['top'] +
                           faceRectangle['height']), color=(255, 255, 0), thickness=5)

        for idx, currFace in enumerate(result):
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
            textToWrite = "%i points: %s" % (firstEmotion, self.currEmotion)
            secondLine = "%i points: %s" % (secEmotion, self.secCurrEmotion)
            if self.easyMode:
                cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                            'top'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 167, 35), 2)
            else:
                cv2.putText(self.photo, textToWrite, (faceRectangle['left'], faceRectangle[
                            'top'] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 167, 35), 2)
                cv2.putText(self.photo, secondLine, (faceRectangle['left'], faceRectangle[
                            'top'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 167, 35), 2)

        if firstEmotion >= 0:
            winner = scores.index(max(scores))
            firstRectLeft = result[winner]['faceRectangle']['left']
            firstRectTop = result[winner]['faceRectangle']['top']
            if self.easyMode:
                cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 167, 35), 2)
            else:
                cv2.putText(self.photo, "Winner: ", (firstRectLeft, firstRectTop - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (232, 167, 35), 2)

        # if self.currPosY >= self.screenheight*(4./5) and self.currPosY < self.screenheight:
        ##
        ##        cv2.rectangle(overlay,(0,int(self.screenheight*(4./5))),(self.screenwidth,self.screenheight),(224,23,101), -1)
        ##        cv2.addWeighted(overlay,self.opacity,self.photo,1-self.opacity, 0, self.frame)
        # else:
        # pass

        if self.currLevel == 2:
            self.level2()

    def level2(self):
        print "self.currLevel is 2"
        overlay = self.photo.copy()
        if self.currPosY >= self.screenheight * (4. / 5) and self.currPosY < self.screenheight:
            cv2.rectangle(overlay, (0, int(self.screenheight * (3. / 4))),
                          (self.screenwidth, self.screenheight), (224, 23, 101), -1)
            cv2.addWeighted(overlay, self.opacity, self.photo,
                            1 - self.opacity, 0, self.frame)
        if self.click_point_y >= self.screenheight * (4. / 5) and self.click_point_y < self.screenheight:
            print "Restarting"
        cv2.putText(self.photo, "[Click to play again]", (self.screenwidth / 2, int(
            self.screenheight * (6. / 7))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (62, 184, 144), 2)

    def upload_img(self, frame):
        print "Initate upload"
        self.currLevel = 2

        CLIENT_ID = "525d3ebc1147459"

        im = pyimgur.Imgur(CLIENT_ID)
        uploaded_image = im.upload_image(
            imagepath, title="Uploaded with PyImgur")
        print(uploaded_image.title)
        print(uploaded_image.link)
        print(uploaded_image.size)
        print(uploaded_image.type)

        print "Analyze image"
        data = emotion_api(uploaded_image.link)
        result = data
        self.display(result, frame)

    def endCapture(self):
        # When everything is done, release the capture
        self.cam.release()
        if self.photoMode:
            self.presentation(self.photo)
        else:
            self.addText(self.frame, "Press any key to quit_",
                         (self.screenwidth / 4, self.screenheight / 3))
            self.presentation(self.frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
