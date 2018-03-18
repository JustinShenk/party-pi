import base64
import cv2
import io
import numpy as np
import os
import random
import re
import sys
import tensorflow as tf
import uuid

from flask import Flask, Response, request, render_template, jsonify
from flask_sslify import SSLify
from io import BytesIO
from keras.models import load_model
from keras import backend as K
from utils.inference import load_detection_model, detect_faces, draw_bounding_box
from utils.inference import get_class_to_arg, apply_offsets, get_labels
from utils.misc import *
from PIL import Image

# tf.keras.backend.clear_session()
graph = tf.get_default_graph()

emotion_classifier = load_model('../emotion_model.hdf5', compile=False)

app = Flask(__name__)
app.config.update(dict(
    PREFERRED_URL_SCHEME='https'
))

debug = False
if not debug:
    sslify = SSLify(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# party_pi = PartyPi(web=True)
print("Game loaded")
face_detector = load_detection_model()
if not os.path.exists('static/images'):
    os.mkdir('static/images')

# Get input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Get emotions
EMOTIONS = list(get_labels().values())


def rank_players(player_data, photo, current_emotion='happy'):
    """ Rank players and display.

    Args:
        player_data : list of dicts
        photo : numpy nd array
    """

    text_size = 0.5
    if len(player_data) < 1:
        draw_text((0.2 * photo.shape[0], 0.2 * photo.shape[1]), photo, "No faces found - try again!",
                  font_scale=text_size, color=YELLOW)
        return photo
    scores = []
    first_emotion = None
    easy_mode = True
    emotion_idx_lookup = get_class_to_arg()
    # Get lists of player points.
    first_emotion_idx = emotion_idx_lookup[current_emotion]
    # second_emotion_idx = emotion_idx_lookup[second_current_emotion]
    first_emotion_scores = [
        (round(x['scores'][first_emotion_idx] * 100)) for x in player_data]

    # Collect scores into `scores_list`.
    scores_list = []
    # if easy_mode:  # rank players by one emotion
    scores_list = first_emotion_scores

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
        #
        # Display points.
        score_height_offset = 10
        first_emotion_coord = (faceRectangle['left'], faceRectangle['top'] -
                               score_height_offset)
        draw_text(first_emotion_coord, photo, first_emotion_caption,
                  font_scale=text_size, color=YELLOW)

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


def predict_emotions(faces, gray_image, current_emotion='happy'):
    global graph
    player_data = []
    # Hyperparameters for bounding box
    emotion_offsets = (20, 40)
    emotion_idx_lookup = get_class_to_arg()
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(
            face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
            print("Exception:", e)
            return player_data
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        with graph.as_default():
            emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_index = emotion_idx_lookup[current_emotion]
        # app.logger.debug("EMOTION INDEX: ", emotion_index)
        emotion_score = emotion_prediction[0][emotion_index]
        x, y, w, h = face_coordinates
        face_dict = {'left': x, 'top': y, 'right': x + w, 'bottom': y + h}
        player_data.append(
            {'faceRectangle': face_dict, 'scores': emotion_prediction[0]})
    return player_data


def data_uri_to_cv2_img(uri):
    uri = uri.split(",")
    uri = uri[1]
    image_bytes = BytesIO()
    encoded = str.encode(uri)
    decoded = base64.decodestring(encoded)
    image_bytes.write(decoded)
    image_bytes.seek(0)
    image = Image.open(image_bytes)
    image = image.convert('RGB')
    np_img = np.array(image, dtype=np.uint8)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return img


def readb64(uri):
    uri = uri.split(",")
    base64_string = uri[1]
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    sbuf.seek(0)
    pil_img = Image.open(sbuf).convert('RGB')
    np_img = np.array(pil_img, dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(np_img, color_image_flag)
    return img


def get_face(frame):
    detection_model_path = './face.xml'
    face_detection = load_detection_model(detection_model_path)
    try:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detection, gray_image)
    except Exception as e:
        print("Exception:", e)
        return frame
    for face in faces:
        draw_bounding_box(face, frame, (255, 0, 0))
    return frame


@app.route('/image', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        print("POST request")
        try:
            f = request.form
            for key in f.keys():
                for value in f.getlist(key):
                    print(key, ":", value[:50])

            image_b64 = request.form.get('imageBase64')
            if image_b64 is None:
                print("No image in request.")
                return jsonify(success=False, photoPath='')
            print("image_b64 request received.", image_b64[:10])
            # Get emotion
            # emotion = json_data['emotion']
            emotion = request.form.get('emotion')
            if emotion is None:
                print("No emotion in request.")
                return jsonify(success=False, photoPath='')
            img = data_uri_to_cv2_img(image_b64)
            print(img.shape)
            w, h, c = img.shape
            if w > 480:
                print("Check yo' image size.")
                img = cv2.resize(img, (int(480 * w / h), 480))
                print("New size {}.".format(img.shape))
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_detector, gray_image)
            player_data = predict_emotions(
                faces, gray_image, emotion)
            photo = rank_players(player_data, img, emotion)
            # photo = party_pi.photo
            photo_path = 'static/images/{}.jpg'.format(str(uuid.uuid4()))
            cv2.imwrite(photo_path, photo)
            print("Saved image to {}".format(photo_path))
            return jsonify(success=True,
                           photoPath=photo_path
                           )
        except Exception as e:
            print("ERROR:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            # response = jsonify({'image': photo_path,
            #                     'score': False})
            # response.status_code = 500
            return jsonify(success=False, photoPath='')
        print("What happened here?")
        return jsonify(success=False, photoPath='')


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


# @app.after_request
# def add_header(r):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
#     return r


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
    try:
        debug_js = 'true' if debug else 'false'
        print("Page accessed, debug mode:", debug_js)
        return render_template('index.html', debug=debug_js)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    # response = {'message': '<h1>Hello world</h1>' + img, ''}
    # response_pickled = jsonpickle.encode(response)
    # return Response(response=response_pickled, status=200, mimetype="application/json", headers=)


@app.route('/none')
def none():
    Response(get_image(empty=True),
             mimetype='multipart/x-mixed-replace; boundary=frame')

# HTTP Errors handlers


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    threaded = False
    if os.path.exists('cert.pem'):  # local environment only
        app.run(host='0.0.0.0', ssl_context=(
            'cert.pem', 'key.pem'), debug=debug, threaded=threaded)
    elif os.path.exists('/etc/letsencrypt/live/openhistoryproject.com/'):
        app.run(host='0.0.0.0', ssl_context=(
            '/etc/letsencrypt/live/openhistoryproject.com/cert.pem', '/etc/letsencrypt/live/openhistoryproject.com/privkey.pem'), debug=debug, threaded=threaded)
    else:
        app.run(host='0.0.0.0', debug=debug, threaded=threaded)
