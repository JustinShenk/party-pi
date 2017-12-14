#! /usr/bin/env python
import cv2
import json
import numpy as np
import os
import random
import time

from keras.models import load_model
from statistics import mode
from uploader import Uploader
from utils.inference import detect_faces
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import get_class_to_arg
from utils.inference import get_labels
from utils.misc import *
from utils.preprocessor import preprocess_input

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Parameters for loading data and images
detection_model_path = './face.xml'

# Use pretrained model (TODO: Add link to source repo)
emotion_model_path = '../emotion_model.hdf5'
emotion_labels = get_labels()

# Hyperparameters for bounding box
emotion_offsets = (20, 40)

# Load models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Get input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

EMOTIONS = list(get_labels().values())

# Load string constants from json file.
with open('emotions.json', 'r') as f:
    d = f.read()
    data = json.loads(d)
    WAIT_CAPTIONS = data['wait_captions']

FONT = cv2.FONT_HERSHEY_SIMPLEX

face_detection = load_detection_model()


class PartyPi(object):

    def __init__(self, piCam=False, windowSize=(1200, 1024), resolution=(1280 // 2, 1024 // 2), gray=False, debug=False):
        self.piCam = piCam
        self.windowSize = windowSize
        self.gray = gray
        self.debug = debug

        # TODO: Integrate into `EMOTIONS`.
        # EMOTIONS2 = ['psycho', 'John Cena', 'ecstasy', 'duckface']
        self.screenwidth, self.screenheight = self.windowSize
        self.resolution = resolution
        self.initialize_webcam()

        # Reinitialize screenwidth and height in case changed by system.
        upload_caption_x = self.screenwidth // 5
        upload_caption_y = self.screenheight // 3 if self.raspberry else self.screenheight // 4 + 30
        self.uploading_caption_coord = (upload_caption_x, upload_caption_y)

        # Complete setup.
        self.setup_game()

    def initialize_webcam(self):
        """ Initialize camera and screenwidth and screenheight.
        """
        device = 'raspberry' if 'raspberrypi' in os.uname() else None
        self.raspberry = True if 'raspberry' == device else False
        if self.piCam:
            camera = self.setup_picamera()
            self.piCamera = camera
            return

        cam = cv2.VideoCapture(0)
        frame = None
        while frame is None:
            try:
                _, frame = cam.read()
                # Update class variables.
                self.screenheight, self.screenwidth = frame.shape[:2]
                cam.set(3, self.screenwidth)
                cam.set(4, self.screenheight)
            except:
                pass
        self.cam = cam
        return

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
        time.sleep(1)
        return piCamera

    def setup_game(self):
        """ Initialize variables, set up icons and face cascade.

        """
        self.status = []
        self.current_emotion = EMOTIONS[0]
        self.countdown = 3

        # Initialize mouse click positions.
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.click_point_right_x = None
        self.click_point_right_y = None

        # Initialize settings
        self.easy_mode = True
        self.current_caption_index = 0
        self.tickcount = 0
        self.curr_level = 0
        self.show_begin = False
        self.uploader = Uploader()

        # Load images
        _paths = {'banner': 'images/partypi_banner.png',
                  'christmas': 'images/christmas.png',
                  'hat': 'images/hat.png'
                  }
        self.webBanner = cv2.imread(
            _paths['banner'], cv2.IMREAD_UNCHANGED)
        self.christmas = cv2.imread(
            _paths['christmas'], cv2.IMREAD_UNCHANGED)
        self.hat = cv2.imread(_paths['hat'], cv2.IMREAD_UNCHANGED)
        # FIXME: add crown for winner
        # self.crown = cv2.imread('images/crown.png', cv2.IMREAD_UNCHANGED)

        if self.hat is None:
            raise ValueError(
                'No hat image found at `{}`'.format(hat_path))
        print("Camera initialized")
        self.flash_on = False
        self.show_analyzing = False
        self.photo_mode = False
        cv2.namedWindow("PartyPi", cv2.WINDOW_GUI_NORMAL)
        # cv2.setWindowProperty(
        #     "PartyPi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback("PartyPi", self.mouse)
        cv2.resizeWindow("PartyPi", self.windowSize[0], self.windowSize[1])

        try:
            if self.piCam:
                # Capture frames from the camera.
                for _frame in self.piCamera.capture_continuous(self.rawCapture, format='bgr', use_video_port=True):
                    # frame = cv2.flip(_frame.array, 1)
                    frame = _frame.array  # convert to array
                    frame.flags.writeable = True
                    self.pi_cam_frame = frame
                    self.screenheight, self.screenwidth = frame.shape[:2]
            else:  # webcam
                while True:
                    callback_status = self.game_loop()
                    if callback_status == "END":
                        break
        except:
            print_traceback()

        self.end_game()

    def game_loop(self):
        """ Start the game loop. Listen for escape key.

        """
        if self.curr_level == 0:
            self.select_mode()
        elif self.curr_level == 1:
            self.play_mode()
        elif self.curr_level == 2:
            self.present_mode()

        # Catch escape key 'q'.
        keypress = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame.
        if self.piCam:
            self.rawCapture.truncate(0)

        callback = self.listen_for_end(keypress)
        return callback

    def select_mode(self):
        """ Select a mode: Easy or Hard.

        """
        self.tickcount + 1

        if self.raspberry:
            self.tickcount += 1
        bgr_image = self.capture_frame()
        # Draw "Easy" and "Hard".
        # bgr_image = self.overlayUI(bgr_image)
        easy_coord = (self.screenwidth // 8, (self.screenheight * 3) // 4)
        draw_text(easy_coord, bgr_image, "Easy", font_scale=3)
        hard_coord = (self.screenwidth // 2, (self.screenheight * 3) // 4)
        draw_text(hard_coord, bgr_image, "Hard", font_scale=3)

        # Listen for mode selection.
        if self.currPosX and self.currPosX < self.screenwidth / 2:
            cv2.rectangle(self.overlay, (0, 0), (self.screenwidth // 2,
                                                 int(self.screenheight)), (211, 211, 211), -1)
        else:
            cv2.rectangle(self.overlay, (self.screenwidth // 2, 0),
                          (self.screenwidth, self.screenheight), (211, 211, 211), -1)
        if self.click_point_x:  # If user clicks left mouse button.
            # OPTIONAL: Positional mode selection
            # self.easy_mode = True if self.click_point_x < self.screenwidth / 2
            # else False
            self.easy_mode = True
            self.tickcount = 0
            self.curr_level = 1
            self.click_point_x = None
            self.click_point_right_x = None
        if self.click_point_right_x:
            self.easy_mode = False
            self.tickcount = 0
            self.curr_level = 1
            self.click_point_x = None
            self.click_point_right_x = None

        # Draw faces.
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detection, gray_image)
        cv2.addWeighted(self.overlay, OPACITY, bgr_image,
                        1 - OPACITY, 0, bgr_image)

        if DEBUG:
            for face in faces:
                draw_bounding_box(face, bgr_image, (255, 0, 0))
        # Draw Christmas logo.
        self.draw_hats(bgr_image, faces)
        self.draw_christmas_logo(bgr_image)  # Only for christmas
        # Show image.
        cv2.imshow('PartyPi', bgr_image)

    def play_mode(self):
        """ Display emotion prompt, upload image, and display results.

        """

        bgr_image = self.capture_frame()
        self.tick()  # update tickcount

        if SLOW:
            timer = self.tickcount // 2
        else:
            timer = self.tickcount
        self.prompt_emotion(bgr_image)
        # Show 'Begin' after some time
        if timer < 70:
            pass
        elif timer < 80:
            self.show_begin = True
        elif timer < 100:
            pass
        elif timer >= 100 and timer < 110:  # 3...
            self.show_begin = False
            bgr_image = self.draw_countdown(bgr_image)
        elif timer >= 110 and timer < 120:  # 2...
            self.countdown = 2
            bgr_image = self.draw_countdown(bgr_image)
        elif timer >= 120 and timer < 130:  # 1...
            self.countdown = 1
            bgr_image = self.draw_countdown(bgr_image)
        elif timer >= 130 and timer < 138:  # flash, save image, analyze
            self.flash_on = True
            if not self.raspberry:
                if timer == 134:  # take photo
                    self.photo_mode = True
                    self.photo = bgr_image.copy()
                    self.start_process = True
                else:
                    self.start_process = False
                    self.show_analyzing = True
            else:  # Raspberry-specific timing
                if timer == 134:
                    self.photo_mode = True
                    self.photo = bgr_image.copy()
                    self.start_process = True
                else:
                    self.start_process = False
                    self.show_analyzing = True
        elif timer == 138:  # reset settings and increase level
            self.start_process = False
            self.flash_on = False
            self.show_analyzing = False
            self.curr_level = 2
            self.click_point_y = None

        # Draw other text and flash on screen.
        text_size = 2.8 if self.raspberry else 3.6
        if self.flash_on:  # overlay white screen to add light
            cv2.rectangle(bgr_image, (0, 0), (self.screenwidth,
                                              self.screenheight), (255, 255, 255), -1)
        if self.show_analyzing:  # show waiting text
            bgr_image = self.draw_analyzing(bgr_image)

        # Display image.
        # bgr_image = self.overlayUI(bgr_image)
        cv2.imshow('PartyPi', bgr_image)
        if self.photo_mode and self.start_process:
            self.take_photo()

    def draw_analyzing(self, frame):
        text_size = 0.7 if self.raspberry else 1
        caption = WAIT_CAPTIONS[self.current_caption_index % len(
            WAIT_CAPTIONS)]
        draw_text(self.uploading_caption_coord, frame,
                  caption, font_scale=text_size, color=(244, 23, 101))
        # self.draw_christmas_logo(frame) # Only for christmas
        if 'Error' in self.status:
            error_coord = (self.uploading_caption_coord[
                0], self.uploading_caption_coord[1] + 80)
            draw_text(error_coord, frame, 'Please check your internet connection', color=(
                244, 23, 101), font_scale=text_size * 0.7)
        else:
            wait_coord = (self.uploading_caption_coord[
                0], self.uploading_caption_coord[1] + 80)
            draw_text(wait_coord, frame, 'Please wait',
                      font_scale=text_size * 0.7, color=(244, 23, 101))
        return frame

    def draw_countdown(self, frame):
        # Draw the count "3..".
        countdown_x_offset = 1 + self.countdown  # Offset from left edge
        countdown_x = int(self.screenwidth -
                          (self.screenwidth / 5) * countdown_x_offset)
        self.overlay = frame.copy()
        countdown_panel_y1 = int(self.screenheight * (4. / 5))
        cv2.rectangle(self.overlay, (0, countdown_panel_y1),
                      (self.screenwidth, self.screenheight), (224, 23, 101), -1)
        cv2.addWeighted(self.overlay, OPACITY, frame,
                        1 - OPACITY, 0, frame)
        countdown_y_offset = 20
        countdown_y = int((self.screenheight * 7. / 8) + countdown_y_offset)
        countdown_coord = (countdown_x, countdown_y)
        draw_text(countdown_coord, frame, str(self.countdown))
        return frame

    def present_mode(self):
        """ Show analyzing, then present photo, then reset game.

        """
        self.tickcount += 1
        self.current_caption_index += 1
        # self.capture_frame()

        if self.raspberry:
            self.tickcount += 1

        reset_text_coord = (self.screenwidth // 2, int(
            self.screenheight * (6. / 7)))
        draw_text(reset_text_coord, self.photo,
                  "[Press any button]", color=GREEN, font_scale=1)

        # Draw logo or title.
        # self.photo = self.overlayUI(self.photo)
        self.draw_christmas_logo(self.photo)  # Only for Christmas
        # self.draw_hats(self.photo, self.faces, crowns=crown_over_faces)
        cv2.imshow('PartyPi', self.photo)

    def overlayUI(self, frame):
        h, w, = self.webBanner.shape[:2]
        y, x, = frame.shape[:2]
        frame_loc = (y - h, y, x - w, x)
        banner_loc = (0, h, 0, w)
        subImage = self._remove_background(
            frame, self.webBanner, frame_loc, banner_loc)
        frame[y - h:y, x - w: x] = subImage

        return frame

    def _remove_background(self, frame, image, frame_loc, img_loc):
        """Remove black background from `image` and place on `frame`.

        Args:

        Returns:

        """
        y0, y1, x0, x1 = frame_loc
        img_y0, img_y1, img_x0, img_x1 = img_loc

        # Iterate over all channels
        for c in range(0, 3):
            img_slice = image[img_y0:img_y1, img_x0:img_x1, c] * \
                (image[img_y0:img_y1, img_x0:img_x1, 3] / 255.0)
            bg_slice = frame[y0:y1, x0:x1, c] * \
                (1.0 - image[img_y0:img_y1, img_x0:img_x1, 3]
                    / 255.0)
            frame[y0:y1, x0:x1, c] = img_slice + bg_slice
        return frame[y0:y1, x0:x1]

    def mouse(self, event, x, y, flags, param):
        """ Listen for mouse.

        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.currPosX, self.currPosY = x, y
            # print "curposX,Y", x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.click_point_x, self.click_point_y = x, y
            if self.curr_level == 0:
                self.easy_mode = True
                self.curr_level = 1
            if self.curr_level == 2:
                self.reset()

        elif event == cv2.EVENT_RBUTTONUP:
            self.click_point_right_x, self.click_point_right_y = x, y
            if self.present_mode:
                self.reset()
                print("DEBUG RIGHT CLICK")
                self.easy_mode = False
                self.curr_level = 1

    def tick(self):
        self.tickcount += 1
        # if self.raspberry: # FIXME: Test this
        #     self.tickcount += 1

    def reset(self):
        """ Reset to beginning state.

        """
        self.curr_level = 0
        self.currPosX = None
        self.currPosY = None
        self.click_point_x = None
        self.click_point_y = None
        self.click_point_right_x = None
        self.current_emotion = EMOTIONS[3]
        self.tickcount = 0

        self.show_begin = False
        self.countdown = 3

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

    def draw_hats(self, frame, faces, crowns=None):
        """ Draws hats above detected faces.

        """
        frame_height, frame_width = frame.shape[:2]

        w_offset = 1.3
        x_offset = 7
        y_offset = 40

        for ind, (x, y, w, h) in enumerate(faces):
            # if crowns is not None and ind in crowns:
            #     hat = self.crown.copy()
            # else:
            #     hat = self.hat.copy()
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

            frame[y0:y1, x0:x1] = self._remove_background(frame, hat, frame_loc=(
                y0, y1, x0, x1), img_loc=(hat_top, hat_bottom, hat_left, hat_right))

    def capture_frame(self):
        """ Capture frame-by-frame.

        """
        if self.piCam:
            self.overlay = self.pi_cam_frame.copy()
            return self.pi_cam_frame
        else:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            # Update overlay
            self.overlay = frame.copy()
            return frame

    def take_photo(self):
        """ Take photo and prepare to write, then send to PyImgur (optional).

        """
        faces = detect_faces(face_detection, self.photo)
        gray_image = cv2.cvtColor(self.photo, cv2.COLOR_RGB2GRAY)
        self.draw_hats(self.photo, faces)
        player_data = self.predict_emotions(faces, gray_image)
        self.rank_players(player_data)
        self.save_photo()

    def save_photo(self):
        image_path = new_image_path()
        cv2.imwrite(image_path, self.photo)

    def predict_emotions(self, faces, gray_image):
        player_data = []
        emotion_idx_lookup = get_class_to_arg()
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(
                face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            gray_face = cv2.resize(gray_face, emotion_target_size)
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_index = emotion_idx_lookup[self.current_emotion]
            print("EMOTION INDEX: ", emotion_index, emotion_prediction)
            emotion_score = emotion_prediction[0][emotion_index]
            self.current_emotion_score = emotion_score

            if self.debug:  # Show all emotions for each face
                self.show_all_emotions(emotion_prediction, face_coordinates)

            x, y, w, h = face_coordinates
            face_dict = {'left': x, 'top': y, 'right': x + w, 'bottom': y + h}
            player_data.append(
                {'faceRectangle': face_dict, 'scores': emotion_prediction[0]})
        return player_data

    def show_all_emotions(self, emotion_prediction, face_coordinates):
        emotion_labels = get_labels()
        for i in range(len(emotion_prediction[0])):
            x, y, w, h = face_coordinates
            emotion_text = emotion_labels[i]
            emotion_score = "{}: {:.2f}".format(
                emotion_text, emotion_prediction[0][i])

    def rank_players(self, player_data):
        """ Rank players and display.

        Args:
            player_data : list of dicts
        """
        scores = []
        max_first_emo = None
        max_second_emo = None
        first_emotion = None
        emotion_idx_lookup = get_class_to_arg()
        # Get lists of player points.
        first_emotion_idx = emotion_idx_lookup[self.current_emotion]
        second_emotion_idx = emotion_idx_lookup[self.second_current_emotion]
        first_emotion_scores = [
            (round(x['scores'][first_emotion_idx] * 100)) for x in player_data]
        second_emotion_scores = [(round(
            x['scores'][second_emotion_idx] * 100)) for x in player_data]

        # Collect scores into `scores_list`.
        scores_list = []
        if self.easy_mode:  # rank players by one emotion
            scores_list = first_emotion_scores
        else:  # hard mode scores are a product of percentage of both emotions
            for i in range(len(first_emotion_scores)):
                scores_list.append(
                    (first_emotion_scores[i] + 1) * (second_emotion_scores[i] + 1))
        print("INFO: Scores_list:", scores_list)
        text_size = 0.5 if self.raspberry else 0.8
        # Draw the scores for the faces.
        for i, currFace in enumerate(player_data):
            faceRectangle = currFace['faceRectangle']

            # Get points for first emotion.
            first_emotion = first_emotion_scores[i]
            second_emotion = second_emotion_scores[i]

            # Format points.
            if first_emotion == 1:  # singular 'point'
                first_emotion_caption = "%i point: %s" % (
                    first_emotion, self.current_emotion)
            else:
                first_emotion_caption = "%i points: %s" % (
                    first_emotion, self.current_emotion)
            if second_emotion == 1:  # singular 'point'
                second_emotion_caption = "%i point: %s" % (
                    second_emotion, self.second_current_emotion)
            else:
                second_emotion_caption = "%i points: %s" % (
                    second_emotion, self.second_current_emotion)

            # Display points.
            score_height_offset = 10 if self.easy_mode else 40
            first_emotion_coord = (faceRectangle['left'], faceRectangle['top'] -
                                   score_height_offset)
            draw_text(first_emotion_coord, self.photo, first_emotion_caption,
                      font_scale=text_size, color=YELLOW)

            if not self.easy_mode:  # second line
                second_emotion_coord = (faceRectangle['left'], faceRectangle[
                    'top'] - 10)
                draw_text(second_emotion_coord, self.photo, second_emotion_caption,
                          color=YELLOW, font_scale=text_size)

            # Display 'Winner: ' above player with highest score.
            one_winner = True
            final_scores = scores_list
            winner = final_scores.index(max(final_scores))
            max_score = max(final_scores)

            # Multiple winners - tie breaker.
            if final_scores.count(max_score) > 1:
                print("Multiple winners!")
                one_winner = False
                tied_winners = []
                for ind, i in enumerate(final_scores):
                    if i == max_score:
                        tied_winners.append(ind)

            # Identify winner's face.
            first_rect_left = player_data[winner]['faceRectangle']['left']
            first_rect_top = player_data[winner]['faceRectangle']['top']
            self.crown_over_faces = []
            if one_winner:
                tied_text_height_offset = 40 if self.easy_mode else 70
                draw_text((first_rect_left, first_rect_top -
                           tied_text_height_offset), self.photo, "Winner: ", color=YELLOW, font_scale=text_size)
                self.crown_over_faces = [winner]
            else:
                tied_text_height_offset = 40 if self.easy_mode else 70
                print("tied_winners:", tied_winners)
                for winner in tied_winners:
                    # FIXME: show both
                    first_rect_left = player_data[
                        winner]['faceRectangle']['left']
                    first_rect_top = player_data[winner]['faceRectangle']['top']
                    tied_coord = (first_rect_left,
                                  first_rect_top - tied_text_height_offset)
                    draw_text(tied_coord, self.photo, "Tied: ",
                              color=YELLOW, font_scale=text_size)
                self.crown_over_faces = tied_winners

    def prompt_emotion(self, img_array):
        """ Display prompt for emotion on screen.

        """
        text_size = 1.0 if self.raspberry else 1.2
        prompt_x0 = self.screenwidth // 10
        prompt_coord = (prompt_x0, 3 * (self.screenheight // 4))
        text = "Show " + self.random_emotion() + '_'
        draw_text(prompt_coord, img_array, text=text,
                  color=GREEN, font_scale=1.5)
        draw_text(prompt_coord, img_array, text=text,
                  color=GREEN, font_scale=1.5, thickness=2)

    def random_emotion(self):
        """ Pick a random emotion from list of emotions.

        """
        if self.tickcount < 30:  # generate random emotion
            self.current_emotion = random.choice(EMOTIONS)
            # Select another emotion for second emotion
            current_emotion_idx = EMOTIONS.index(self.current_emotion)
            new_emotion_idx = (current_emotion_idx +
                               random.choice(list(range(1, 7)))) % 7
            self.second_current_emotion = EMOTIONS[new_emotion_idx]
            if self.easy_mode:
                return self.current_emotion
            else:
                return self.current_emotion + '+' + self.second_current_emotion
        else:  # hold emotion for prompt
            emotionString = str(
                self.current_emotion) if self.easy_mode else self.current_emotion + '+' + self.second_current_emotion
            return emotionString

    def listen_for_end(self, keypress):
        """ Listen for 'q', left, or right keys to end game.

        """
        if keypress != 255:
            print(keypress)
            if keypress == ord('q'):  # 'q' pressed to quit
                print("Escape key entered")
                return "END"
            elif self.curr_level == 0:
                # Select mode
                self.curr_level = 1
                self.tickcount = 0
                if keypress == 81 or keypress == 2:  # left
                    self.easy_mode = True
                elif keypress == 83 or keypress == 3:  # right
                    self.easy_mode = False
            elif self.curr_level == 2:
                print("Resetting")
                self.reset()

    def end_game(self):
        """ When everything is done, release the capture.

        """
        if not self.piCam:
            self.cam.release()
            quit_coord = (self.screenwidth // 4, self.screenheight // 3)
            draw_text(quit_coord, self.photo,
                      "Press any key to quit_", font_scale=1)
            # self.presentation(frame)
            # self.photo = self.overlayUI(self.photo)
        else:
            self.piCamera.close()

        cv2.imshow("PartyPi", self.photo)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """ Run application.
    """
    app = PartyPi()


if __name__ == '__main__':
    main()
