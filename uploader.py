#!/usr/bin/env python
import pyimgur
import json
from emotion_API import Emotion_API


def get_conf():
    """ Load settings into object.

    """
    class ConfObj:
        pass

    conf_dict = json.load(open('config.json'))
    conf_obj = ConfObj()
    for key, value in conf_dict.items():
        setattr(conf_obj, key, value)

    return conf_obj


class Uploader(object):

    def __init__(self):
        self.config = get_conf()
        self.emotion_API = Emotion_API()
        self.initialize_pyimgur()

    def initialize_pyimgur(self):
        """
        Initialize variables and parameters for PyImgur.
        """
        _url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        _maxNumRetries = 10
        CLIENT_ID = self.config.CLIENT_ID
        CLIENT_SECRET = self.config.CLIENT_SECRET
        self.album = self.config.album

        self.im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)
        self.im.change_authentication(
            refresh_token=self.config.refresh_token)
        self.im.refresh_access_token()

        # TODO: Complete Imgur user settings.
        # user = self.config.user

    def upload_img(self, imagepath):
        """
        Send image to PyImgur.
        """
        print("Initate upload")

        uploaded_image = self.im.upload_image(
            imagepath, title="Uploaded with PyImgur", album=self.album)

        # TODO: Turn on album uploading
        # uploaded_image = self.im.upload_image(
        #     self.imagepath, title="Uploaded with PyImgur", album=self.album)
        print((uploaded_image.title))
        print((uploaded_image.link))
        print((uploaded_image.size))
        print((uploaded_image.type))

        # Send emotion data to API and return to game.
        data = self.emotion_API.get_emotions(uploaded_image.link)

        return data
