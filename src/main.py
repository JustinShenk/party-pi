import base64
import cv2
import io
import numpy as np
import random
import re
import tensorflow as tf
import uuid

from flask import Flask, Response, request, render_template, send_file
from io import BytesIO
from keras.models import load_model
from keras import backend as K
from utils.inference import load_detection_model, detect_faces, draw_bounding_box
from utils.inference import get_class_to_arg, apply_offsets, get_labels
from utils.misc import *
from PIL import Image
# from play import PartyPi

# tf.keras.backend.clear_session()
emotion_classifier = load_model('../emotion_model.hdf5', compile=False)
graph = K.get_session().graph
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# party_pi = PartyPi(web=True)
print("Game loaded")
face_detector = load_detection_model()
if not os.path.exists('static/images'):
    os.mkdir('static/images')

# Get input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
EMOTIONS = list(get_labels().values())
current_emotion = EMOTIONS[3]


def rank_players(player_data, photo):
    """ Rank players and display.

    Args:
        player_data : list of dicts
        photo : numpy nd array
    """
    scores = []
    max_first_emo = None
    max_second_emo = None
    first_emotion = None
    easy_mode = True
    emotion_idx_lookup = get_class_to_arg()
    # Get lists of player points.
    first_emotion_idx = emotion_idx_lookup[current_emotion]
    # second_emotion_idx = emotion_idx_lookup[second_current_emotion]
    first_emotion_scores = [
        (round(x['scores'][first_emotion_idx] * 100)) for x in player_data]
    # second_emotion_scores = [(round(
    #     x['scores'][second_emotion_idx] * 100)) for x in player_data]

    # Collect scores into `scores_list`.
    scores_list = []
    # if easy_mode:  # rank players by one emotion
    scores_list = first_emotion_scores
    # else:  # hard mode scores are a product of percentage of both emotions
    #     for i in range(len(first_emotion_scores)):
    #         scores_list.append(
    #             (first_emotion_scores[i] + 1) * (second_emotion_scores[i] + 1))
    text_size = 0.8
    # Draw the scores for the faces.
    for i, currFace in enumerate(player_data):
        faceRectangle = currFace['faceRectangle']

        # Get points for first emotion.
        first_emotion = first_emotion_scores[i]
        # second_emotion = second_emotion_scores[i]

        # Format points.
        if first_emotion == 1:  # singular 'point'
            first_emotion_caption = "%i point: %s" % (
                first_emotion, current_emotion)
        else:
            first_emotion_caption = "%i points: %s" % (
                first_emotion, current_emotion)
        # if second_emotion == 1:  # singular 'point'
        #     second_emotion_caption = "%i point: %s" % (
        #         second_emotion, second_current_emotion)
        # else:
        #     second_emotion_caption = "%i points: %s" % (
        #         second_emotion, second_current_emotion)
        #
        # Display points.
        # score_height_offset = 10 if easy_mode else 40
        score_height_offset = 10
        first_emotion_coord = (faceRectangle['left'], faceRectangle['top'] -
                               score_height_offset)
        draw_text(first_emotion_coord, photo, first_emotion_caption,
                  font_scale=text_size, color=YELLOW)

        # if not easy_mode:  # second line
        #     second_emotion_coord = (faceRectangle['left'], faceRectangle[
        #         'top'] - 10)
        #     draw_text(second_emotion_coord, photo, second_emotion_caption,
        #               color=YELLOW, font_scale=text_size)

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
        crown_over_faces = []
        if one_winner:
            tied_text_height_offset = 40 if easy_mode else 70
            draw_text((first_rect_left, first_rect_top -
                       tied_text_height_offset), photo, "Winner: ", color=YELLOW, font_scale=text_size)
            crown_over_faces = [winner]
        else:
            tied_text_height_offset = 40 if easy_mode else 70
            print("tied_winners:", tied_winners)
            for winner in tied_winners:
                # FIXME: show both
                first_rect_left = player_data[
                    winner]['faceRectangle']['left']
                first_rect_top = player_data[winner]['faceRectangle']['top']
                tied_coord = (first_rect_left,
                              first_rect_top - tied_text_height_offset)
                draw_text(tied_coord, photo, "Tied: ",
                          color=YELLOW, font_scale=text_size)
            # crown_over_faces
    return photo


def random_emotion():
    """ Pick a random emotion from list of emotions.

    """
    # if tickcount < 30:  # generate random emotion
    current_emotion = random.choice(EMOTIONS)
    # Select another emotion for second emotion
    current_emotion_idx = EMOTIONS.index(current_emotion)
    new_emotion_idx = (current_emotion_idx +
                       random.choice(list(range(1, 7)))) % 7
    second_current_emotion = EMOTIONS[new_emotion_idx]
    # if easy_mode:
    return current_emotion
    # else:
    #     return current_emotion + '+' + second_current_emotion
    # else:  # hold emotion for prompt
    #     emotionString = str(
    #         current_emotion) if easy_mode else current_emotion + '+' + second_current_emotion
    #     return emotionString


def predict_emotions(faces, gray_image):
    global graph, current_emotion
    player_data = []
    # Hyperparameters for bounding box
    emotion_offsets = (20, 40)
    emotion_idx_lookup = get_class_to_arg()
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(
            face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        gray_face = cv2.resize(gray_face, emotion_target_size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        with graph.as_default():
            emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_index = emotion_idx_lookup[current_emotion]
        print("EMOTION INDEX: ", emotion_index, emotion_prediction)
        emotion_score = emotion_prediction[0][emotion_index]
        current_emotion_score = emotion_score

        x, y, w, h = face_coordinates
        face_dict = {'left': x, 'top': y, 'right': x + w, 'bottom': y + h}
        player_data.append(
            {'faceRectangle': face_dict, 'scores': emotion_prediction[0]})
    return player_data


def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    sbuf.seek(0)
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def get_face(frame):
    detection_model_path = './face.xml'
    face_detection = load_detection_model(detection_model_path)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(face_detection, gray_image)
    for face in faces:
        draw_bounding_box(face, frame, (255, 0, 0))
    return frame


@app.route('/image', methods=['POST', 'GET'])
def image():
    # data = request.data
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '',
                        image_b64)
    try:
        img = readb64(image_data)
        app.logger.debug(img.shape)
        # cv2.imwrite('player.jpg', img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detect_faces(face_detector, gray_image)
        app.logger.debug("Faces: ", len(faces))
        player_data = predict_emotions(faces, gray_image)
        photo = rank_players(player_data, img)
        # photo = party_pi.photo
        photo_path = 'static/images/{}.jpg'.format(str(uuid.uuid4()))
        cv2.imwrite(photo_path, photo)
        print("Saved image to {}".format(photo_path))
        #
        # _, img_encoded = cv2.imencode('.jpg', photo)
        # print("OUTPUT:", img_encoded.tostring())
        with open(photo_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
    except Exception as e:
        print("ERROR:", e)
        return ''
    return encoded_string


def get_image(empty=False, face=False):
    while True:
        if empty:
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        # _, frame = cam.read()
        # if face:
        #     frame = get_face(frame)
        # cv2.imwrite('t.jpg', frame)
        # yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' +
        #        open('t.jpg', 'rb').read() + b'\r\n')
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' +
               b'\r\n')


@app.route('/select_mode', methods=['POST'])
def select_mode(click_x):
    print("CLICKX: ", click_x)


@app.route('/video_feed')
def video_feed():
    return Response(get_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/face')
def face():
    return Response(get_image(face=True), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # ret, jpeg = cv2.imencode('.jpg', frame)
    # img = jpeg.tobytes()
    # # prepare headers for http request
    # content_type = 'image/jpeg'
    # headers = {'content-type': content_type}

    # img = cv2.imread('src/images/christmas.png')

    # send http request with image and receive response
    # response = requests.post(
    #     test_url, data=img_encoded.tostring(), headers=headers)
    return render_template('index.html')

    # response = {'message': '<h1>Hello world</h1>' + img, ''}
    # response_pickled = jsonpickle.encode(response)
    # return Response(response=response_pickled, status=200, mimetype="application/json", headers=)


@app.route('/none')
def none():
    Response(get_image(empty=True),
             mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
