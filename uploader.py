#!/ usr/bin/env python
import pyimgur
from emotionpi import emotion_api
from secret import *

class Uploader(object):

    def __init__(self):
        self.initPyimgur()

    def initPyimgur(self):
        """
        Initialize variables and parameters for PyImgur.
        """
        self.album = "iX0uj"  # Testing.
        # self.album = "zzf6O"
        # self.album = "6U86u"
        # self.album = "JugqY"
        _url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        _key = get_key()
        _maxNumRetries = 10
        CLIENT_ID = get_client_id()
        CLIENT_SECRET = get_client_secret()

        self.im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)
        self.im.change_authentication(
            refresh_token=get_refresh_token())
        self.im.refresh_access_token()
        # user = self.im.get_user('spacemaker')
        # album = im.get_album(ALBUM_ID)

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

        print("Analyze image")
        data = emotion_api(uploaded_image.link)
        return data
