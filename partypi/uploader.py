#!/usr/bin/env python
import json
import time
import requests

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
    """Upload files to pyimgur or directly to Microsoft Oxford API."""

    def __init__(self, local=True):
        self.config = get_conf()
        self.emotion_API = Emotion_API()
        if local:
            pass
        else:
            self.initialize_pyimgur()

    def initialize_pyimgur(self):
        """
        Initialize variables and parameters for PyImgur.
        """
        import pyimgur
        CLIENT_ID = self.config.CLIENT_ID
        CLIENT_SECRET = self.config.CLIENT_SECRET
        self.album = self.config.album

        # Retry if connection fails
        ok = False
        while not ok:
            try:
                self.im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)
                self.im.change_authentication(
                    refresh_token=self.config.refresh_token)
                self.im.refresh_access_token()
                ok = True
            except requests.exceptions.HTTPError:
                print("Imgur is temporarily over capacity. Please try again later.")
                time.sleep(4)

        # TODO: Complete Imgur user settings.
        # user = self.config.user

    def upload_img(self, imagepath, pyimgur=False):
        """
        Send local image directly or send image to PyImgur for hosting.
        """
        print("Initate upload")

        if pyimgur:
            uploaded_image = self.im.upload_image(
                imagepath, title="Uploaded with PyImgur", album=self.album)

            # TODO: Turn on album uploading
            # uploaded_image = self.im.upload_image(
            # self.imagepath, title="Uploaded with PyImgur", album=self.album)
            print((uploaded_image.title))
            print((uploaded_image.link))
            print((uploaded_image.size))
            print((uploaded_image.type))

            # Send emotion data to API and return to game.
            data = self.emotion_API.get_emotions(uploaded_image.link)
        else:
            data = self.emotion_API.get_emotions(imagepath, local=True)

        return data
