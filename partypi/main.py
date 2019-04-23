# -*- coding: utf-8 -

import base64
from io import BytesIO
import logging
import random
import sys
import uuid

import flask
import numpy as np
import requests
from flask import Flask, Response, request, render_template, jsonify, make_response
from PIL import Image

from partypi.utils.inference import (
    load_detection_model,
    detect_faces,
    draw_bounding_box,
)
from partypi.utils.inference import get_class_to_arg, apply_offsets, get_labels
from partypi.utils.tweeter import tweet_image, tweet_message
from partypi.utils.misc import *

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True
app.config.update(dict(PREFERRED_URL_SCHEME='https'))
app.config['GOOGLE_LOGIN_REDIRECT_SCHEME'] = "https"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.secret_key = os.environ.get("FLASK_SECRET_KEY")

app.logger.info("Game loaded")

face_detector = load_detection_model()

if os.environ.get("PARTYPI_VERSION") == 2:
    import google_auth_oauthlib
    import google
    import googleapiclient

    CLIENT_SECRETS_FILE = 'client_secret.json'
    if not os.path.exists(CLIENT_SECRETS_FILE):
        # Load google auth config
        goog_config = os.environ.get('GOOGWeb')
        with open(CLIENT_SECRETS_FILE, 'w') as outfile:
            outfile.write(goog_config)

    # Set up v2
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    RANGE_PREFIX = 'ICML2018!'
    RANGE_NAME = RANGE_PREFIX + 'A:Z'
    SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID')
    API_SERVICE_NAME = 'sheets'
    API_VERSION = 'v4'

# Get input model shapes for inference
emotion_target_size = (64, 64)

# Get emotions
EMOTIONS = list(get_labels().values())


def draw_logo(photo, logo="PartyPi.png"):
    """Draws logo on `photo` in bottom right corner."""
    logo = cv2.imread(logo, cv2.IMREAD_UNCHANGED)
    if logo is None:
        app.logger.error(os.listdir())
        app.logger.error("No image found")
        return
    photoRows, photoCols = photo.shape[:2]
    rows, cols = logo.shape[:2]
    y0, y1, x0, x1 = photoRows - rows, photoRows, 0, cols
    for c in range(0, 3):
        logo_slice = logo[:, :, c] * (logo[:, :, 3] / 255.0)
        bg_slice = photo[y0:y1, x0:x1, c] * (1.0 - logo[:, :, 3] / 255.0)
        photo[y0:y1, x0:x1, c] = logo_slice + bg_slice
    return photo


def rank_players(player_data, photo, current_emotion='happy',
                 one_player=False):
    """ Rank players and display.

    Args:
        player_data : list of dicts
        photo : numpy nd array

    """
    text_size = 0.5
    if len(player_data) < 1:
        draw_text(
            (0.2 * photo.shape[0], 0.2 * photo.shape[1]),
            photo,
            "No faces found - try again!",
            font_scale=text_size,
            color=YELLOW,
        )
        if one_player:
            return photo, [], 1
        else:
            return photo, []

    easy_mode = True
    emotion_idx_lookup = get_class_to_arg()
    # Get lists of player points.
    first_emotion_idx = emotion_idx_lookup[current_emotion]
    first_emotion_scores = [(round(x['scores'][first_emotion_idx] * 100))
                            for x in player_data]

    # Collect scores into `scores_list`.
    scores_list = first_emotion_scores

    emotion_offsets = (20, 40)

    largest_face = 0
    player_index = None

    faces_with_scores = []

    # Draw the scores for the faces.
    for i, currFace in enumerate(player_data):
        faceRectangle = currFace['faceRectangle']
        x1, x2 = faceRectangle['left'], faceRectangle['right']
        y1, y2 = faceRectangle['top'], faceRectangle['bottom']

        if one_player:  # mode
            if (x2 - x1) * (y2 - y1) > largest_face:
                largest_face = largest_face
                player_index = i

        # Convert back to coordinates to get offset
        face_coordinates = (x1, y1, x2 - x1, y2 - y1)
        # Get points for first emotion.
        first_emotion = first_emotion_scores[i]
        face_photo_path = 'static/images/face_{}.jpg'.format(str(uuid.uuid4()))
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        face_image = photo[y1:y2, x1:x2]
        cv2.imwrite(face_photo_path, face_image)
        app.logger.info("Saved face to {}".format(face_photo_path))
        faces_with_scores.append((face_photo_path, first_emotion))
        # second_emotion = second_emotion_scores[i]

        # Format points.
        if first_emotion == 1:  # singular 'point'
            first_emotion_caption = "%i point: %s" % (first_emotion,
                                                      current_emotion)
        else:
            first_emotion_caption = "%i points: %s" % (first_emotion,
                                                       current_emotion)
        #
        # Display points.
        score_height_offset = 10
        first_emotion_coord = (
            faceRectangle['left'],
            faceRectangle['top'] - score_height_offset,
        )
        draw_text(
            first_emotion_coord,
            photo,
            first_emotion_caption,
            font_scale=text_size,
            color=YELLOW,
        )

        # Display 'Winner: ' above player with highest score.
        one_winner = True
        final_scores = scores_list
        winner = final_scores.index(max(final_scores))
        max_score = max(final_scores)

        # Multiple winners - tie breaker.
        if final_scores.count(max_score) > 1:
            app.logger.info("Multiple winners!")
            one_winner = False
            tied_winners = []
            for ind, i in enumerate(final_scores):
                if i == max_score:
                    tied_winners.append(ind)

        # Identify winner's face.
        first_rect_left = player_data[winner]['faceRectangle']['left']
        first_rect_top = player_data[winner]['faceRectangle']['top']
        if one_winner:
            tied_text_height_offset = 40 if easy_mode else 70
            draw_text(
                (first_rect_left, first_rect_top - tied_text_height_offset),
                photo,
                "Winner: ",
                color=YELLOW,
                font_scale=text_size,
            )
        else:
            tied_text_height_offset = 40 if easy_mode else 70
            app.logger.info("tied_winners:", tied_winners)
            for winner in tied_winners:
                # FIXME: show both
                first_rect_left = player_data[winner]['faceRectangle']['left']
                first_rect_top = player_data[winner]['faceRectangle']['top']
                tied_coord = (first_rect_left,
                              first_rect_top - tied_text_height_offset)
                draw_text(tied_coord,
                          photo,
                          "Tied: ",
                          color=YELLOW,
                          font_scale=text_size)
    if one_player:
        return photo, faces_with_scores, player_index
    return photo, faces_with_scores


def random_emotion():
    """ Pick a random emotion from list of emotions.

    """
    current_emotion = random.choice(EMOTIONS)

    # Select another emotion for second emotion
    current_emotion_idx = EMOTIONS.index(current_emotion)
    new_emotion_idx = (current_emotion_idx +
                       random.choice(list(range(1, 7)))) % 7
    return current_emotion


def predict_emotions(faces: list,
                     gray_image: np.ndarray,
                     current_emotion: str = 'happy'):
    player_data = []

    # Hyperparameters for bounding box
    emotion_offsets = (20, 40)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
            print("Exception:", e)
            return player_data
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        app.logger.info(f"gray_face: {gray_face.shape}")

        SERVER_URL = 'http://tfserving:8501/v1/models/emotion_model:predict'
        response = requests.post(SERVER_URL,
                                 json={'instances': gray_face.tolist()})
        response.raise_for_status()

        emotion_predictions = response.json()['predictions']
        app.logger.info(f"predictions: {emotion_predictions}")

        x, y, w, h = face_coordinates
        face_dict = {'left': x, 'top': y, 'right': x + w, 'bottom': y + h}
        player_data.append({
            'faceRectangle': face_dict,
            'scores': emotion_predictions[0]
        })
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


@app.route('/')
def index():
    try:
        app.logger.info("Page accessed from {}".format(
            request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)))
        return render_template('index.html')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


@app.route('/image', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        app.logger.info("POST request")
        try:
            form = request.form
            image_b64 = form.get('imageBase64')
            if image_b64 is None:
                app.logger.error("No image in request.")
                return jsonify(success=False, photoPath='')

            # Get emotion
            # emotion = json_data['emotion']
            emotion = form.get('emotion')
            if emotion is None:
                app.logger.error("No emotion in request.")
                return jsonify(success=False, photoPath='')
            img = data_uri_to_cv2_img(image_b64)
            w, h, c = img.shape
            if w > 480:
                app.logger.info("Check yo' image size.")
                img = cv2.resize(img, (int(480 * w / h), 480))
                app.logger.info("New size {}.".format(img.shape))

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_detector, gray_image)
            player_data = predict_emotions(faces, gray_image, emotion)

            photo, faces_with_scores = rank_players(player_data, img, emotion)
            photo = draw_logo(photo)
            photo_path = 'static/images/{}.jpg'.format(str(uuid.uuid4()))
            cv2.imwrite(photo_path, photo)
            app.logger.info("Saved image to {}".format(photo_path))

            addr = request.environ.get('HTTP_X_FORWARDED_FOR',
                                       request.remote_addr)
            message = "Look who's {}".format(emotion)
            try:
                if form.get('canTweetPhoto') == 'true':
                    tweet_image(photo_path, message, public_account=True)
                else:
                    showing = False
                    if emotion in ['fear', 'surprise']:
                        showing = True
                    tweet_message(
                        "Someone is {}{} at {}".format(
                            "showing " if showing else "", emotion, addr),
                        public_account=True,
                    )
            except Exception as e:
                app.logger.error(f"ERROR {e}")
            return jsonify(
                success=True,
                photoPath=photo_path,
                emotion=emotion,
                facesWithScores=faces_with_scores,
                addr=addr,
            )
            status_code = 200
        except Exception as e:
            app.logger.error(f"{e}")
            return jsonify(success=False, photoPath='', statusCode=500)
        return make_response(response, status_code)


###
# Begin methods for singleplayer mode (partypi.net/v2)
###

@app.route('/singleplayer', methods=['POST', 'GET'])
def singleplayer():
    if 'credentials' not in flask.session:
        return flask.redirect('authorize')
    if request.method == 'POST':
        app.logger.info("POST request")
        emotion = 'angry'
        player_name = 'Player Name'
        try:
            form = request.form
            image_b64 = form.get('imageBase64')
            if image_b64 is None:
                app.logger.error("No image in request.")
                return jsonify(success=False, photoPath='')
            # Get emotion
            emotion = form.get('emotion')
            if emotion is None:
                print("No emotion in request.")
                return jsonify(success=False, photoPath='')
            img = data_uri_to_cv2_img(image_b64)
            w, h, c = img.shape
            if w > 480:
                print("Check yo' image size.")
                img = cv2.resize(img, (int(480 * w / h), 480))
                print("New size {}.".format(img.shape))
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_detector, gray_image)
            player_data = predict_emotions(faces, gray_image, emotion)
            photo, faces_with_scores, player_index = rank_players(
                player_data, img, emotion, one_player=True)
            response = get_player_contact()
            try:
                player_name = response[2]
            except Exception as e:
                app.logger.error(e)
            photo_path = 'static/images/{}.jpg'.format(str(uuid.uuid4()))
            if len(faces_with_scores) is 0:
                app.logger.error("No face found")
                return jsonify(
                    success=False,
                    photoPath='',
                    emotion=emotion,
                    facesWithScores=[],
                    playerIndex=None,
                    playerName=player_name,
                    statusCode=500,
                )
            else:
                cv2.imwrite(photo_path, photo)
                app.logger.info("Saved image to {}".format(photo_path))
                update_spreadsheet(faces_with_scores)
                return jsonify(
                    success=True,
                    photoPath=photo_path,
                    emotion=emotion,
                    facesWithScores=faces_with_scores,
                    playerIndex=player_index,
                    playerName=player_name,
                    statusCode=200,
                )
        except Exception as e:
            app.logger.error("ERROR:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            app.logger.error(exc_type, fname, exc_tb.tb_lineno)
            return jsonify(
                success=False,
                photoPath='',
                emotion='',
                playerIndex='',
                facesWithScores='',
                statusCode=501,
            )


def update_spreadsheet(faces_with_scores):
    try:
        score = faces_with_scores[0][1]
        return add_score(score)
    except IndexError:
        return None


def add_to_current(score, service):
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    print(len(values) - 1)
    last_row = len(values)
    next_col = chr(65 + len(values[-1]))
    if next_col < "G":
        next_col = "G"  # don't overwrite phone, etc.
    if next_col > "S":  # googleapiclient.errors.HttpError only to width of sheet
        return "Too many tries"
    range_name = '{}{}{}'.format(RANGE_PREFIX, next_col, last_row)
    values = [[score]]
    body = {'values': values}
    result = (service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=range_name,
        valueInputOption='USER_ENTERED',
        body=body,
    ).execute())
    response = 'Updated cells in {}'.format(result.get('updatedRange'))
    print(response)
    return response


def send_pic(image_path, to):
    """Email pic of player"""
    app.logger.info("Sending {} to {}".format(image_path, to))
    url = 'https://api.mailgun.net/v3/{}/messages'.format('www.partypi.net')
    auth = ('api', os.environ['MAILGUN_API_KEY'])
    data = {
        'from': 'Peltarion Email <no-reply@{}>'.format('partypi.net'),
        'to': to,
        'cc': 'justin@peltarion.com',
        'subject': 'Emotion Contest with Peltarion at TechFest',
        'text': 'Thanks for playing!',
        'html': '<html>Thanks for playing!<strong></strong></html>',
    }
    files = {"attachment": ("techfest.jpg", open(image_path, 'rb'))}
    with app.open_resource(image_path) as fp:
        files = {"attachment": ("techfest.jpg", open(image_path, 'rb'))}
    response = requests.post(url, auth=auth, data=data, files=files)
    response.raise_for_status()


def send_simple_message():
    """Email message"""
    return requests.post(
        "https://api.mailgun.net/v3/sandboxb77d7a26ac594bf7a5f4e008acb62696.mailgun.org/messages",
        auth=("api", "cac4b86b5496062cd33ffc688abaff93-770f03c4-d4803e31"),
        data={
            "from":
                "Mailgun Sandbox <postmaster@sandboxb77d7a26ac594bf7a5f4e008acb62696.mailgun.org>",
            "to":
                "Justin Shenk <shenkjustin@gmail.com>",
            "subject":
                "Hello Justin Shenk",
            "text":
                "Congratulations Justin Shenk, you just sent an email with Mailgun!  You are truly awesome!",
        },
    )


def get_player_contact():
    # Load credentials from the session.
    credentials = google.oauth2.credentials.Credentials(
        **flask.session['credentials'])

    service = googleapiclient.discovery.build(API_SERVICE_NAME,
                                              API_VERSION,
                                              credentials=credentials)
    flask.session['credentials'] = credentials_to_dict(credentials)
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    try:
        recent_player = values[-1]
        return recent_player
    except:
        return None


@app.route('/email', methods=['GET', 'POST'])
def email():
    form = request.form
    image_b64 = form.get('imageBase64')
    if image_b64 is None:
        app.logger.error("No image in request.")
        return jsonify(success=False, photoPath='')
    img = data_uri_to_cv2_img(image_b64)
    img_path = 'email.jpg'
    cv2.imwrite(img_path, img)
    if 'credentials' not in flask.session:
        return flask.redirect('authorize')

    # Load credentials from the session.
    credentials = google.oauth2.credentials.Credentials(
        **flask.session['credentials'])

    service = googleapiclient.discovery.build(API_SERVICE_NAME,
                                              API_VERSION,
                                              credentials=credentials)
    flask.session['credentials'] = credentials_to_dict(credentials)
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    recent_player = values[-1]

    email, name = recent_player[1:3]  # email, name, twitter
    with app.app_context():
        send_pic(img_path, email)
    return jsonify(success=True, photoPath=img_path)


def make_three(recent_player):
    if len(recent_player) is 2:
        recent_player.append('Player')
    return recent_player


@app.route('/tweet', methods=['GET', 'POST'])
def tweet():
    """Tweet to Peltarion_AI twitter feed."""
    form = request.form
    image_b64 = form.get('imageBase64')
    if image_b64 is None:
        app.logger.error("No image in request.")
        return jsonify(success=False, photoPath='')
    img = data_uri_to_cv2_img(image_b64)
    img_path = 'tweet.jpg'
    cv2.imwrite(img_path, img)
    if 'credentials' not in flask.session:
        return flask.redirect('authorize')

    # Load credentials from the session.
    credentials = google.oauth2.credentials.Credentials(
        **flask.session['credentials'])

    service = googleapiclient.discovery.build(API_SERVICE_NAME,
                                              API_VERSION,
                                              credentials=credentials)
    flask.session['credentials'] = credentials_to_dict(credentials)
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    recent_player = values[-1]
    email, name = recent_player[1:3]  # email, name, twitter
    try:
        twitter = recent_player[3]
    except:
        twitter = "My Name"
    message = "{} at #TechFest2018 with @Peltarion_ai".format(
        form.get('emotion'))
    app.logger.info("Tweeting {} {} {}".format(values, email, twitter))
    tweet_image(img_path, message)
    return jsonify(success=True, photoPath='tweet.jpg')


@app.route('/v2')
def v2():
    try:
        app.logger.info("Page accessed from {}".format(
            request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)))
        return render_template('index2.html')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        app.logger.error(exc_type, fname, exc_tb.tb_lineno)


def get_spreadsheet(service):
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    return values


def get_latest_entry(service):
    """Get last entry from spreadsheet"""
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    if not values:
        print('No data found.')
    else:
        return values[-1]  # Email, name, Twitter handle


@app.route('/add/<int:score>')
def add_score(score):
    """Add score to player."""
    if 'credentials' not in flask.session:
        return flask.redirect('authorize')

    # Load credentials from the session.
    credentials = google.oauth2.credentials.Credentials(
        **flask.session['credentials'])

    service = googleapiclient.discovery.build(API_SERVICE_NAME,
                                              API_VERSION,
                                              credentials=credentials)
    flask.session['credentials'] = credentials_to_dict(credentials)
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    print(len(values) - 1)
    last_row = len(values)
    next_col = chr(65 + len(values[-1]))
    if next_col < "G":
        next_col = "G"  # don't overwrite phone, etc.
    if next_col > "S":  # googleapiclient.errors.HttpError only to width of sheet
        return "Too many tries"
    range_name = '{}{}{}'.format(RANGE_PREFIX, next_col, last_row)
    values = [[score]]
    body = {'values': values}
    result = (service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=range_name,
        valueInputOption='USER_ENTERED',
        body=body,
    ).execute())
    response = 'Updated cells in {}'.format(result.get('updatedRange'))
    print(response)
    return jsonify(response)


@app.route('/test')
def test_api_request():
    if 'credentials' not in flask.session:
        return flask.redirect('authorize')

    # Load credentials from the session.
    credentials = google.oauth2.credentials.Credentials(
        **flask.session['credentials'])

    service = googleapiclient.discovery.build(API_SERVICE_NAME,
                                              API_VERSION,
                                              credentials=credentials)
    flask.session['credentials'] = credentials_to_dict(credentials)
    result = (service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID,
                                                  range=RANGE_NAME).execute())
    values = result.get('values', [])
    return jsonify(values)


def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes,
    }


@app.route('/authorize')
def authorize():
    # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
    # FIXME Remove statement
    app.logger.info("/authorize accessed")
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)

    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

    authorization_url, state = flow.authorization_url(
        # Enable offline access so that you can refresh an access token without
        # re-prompting the user for permission. Recommended for web server apps.
        access_type='offline',
        # Enable incremental authorization. Recommended as a best practice.
        include_granted_scopes='true',
    )

    # Store the state so the callback can verify the auth server response.
    flask.session['state'] = state
    return flask.redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
    # Specify the state when creating the flow in the callback so that it can
    # verified in the authorization server response.
    state = flask.session['state']

    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

    # Use the authorization server's response to fetch the OAuth 2.0 tokens.
    authorization_response = flask.request.url
    flow.fetch_token(authorization_response=authorization_response)

    # Store credentials in the session.
    # ACTION ITEM: In a production app, you likely want to save these
    #              credentials in a persistent database instead.
    credentials = flow.credentials
    flask.session['credentials'] = credentials_to_dict(credentials)

    return flask.redirect(flask.url_for('test_api_request'))


#####
# End singleplayer mode methods
#####

# HTTP Errors handlers

@app.errorhandler(404)
def url_error(e):
    return (
        """
    Wrong URL!
    <pre>{}</pre>""".format(e),
        404,
    )


@app.errorhandler(500)
def server_error(e):
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e),
        500,
    )


if __name__ == '__main__':
    if 'TRAVIS' in os.environ:
        sys.exit()
    print("RUNNING APP")
    app.run(
        host='localhost',
        ssl_context=('www.partypi.net.cert', 'www.partypi.net.key'),
        port=8080,
        threaded=False,
    )

if __name__ != '__main__':
    # Gunicorn running
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
