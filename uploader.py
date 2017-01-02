#!/ usr/bin/python
########### Python 2.7 #############
import pyimgur
from emotionpi import emotion_api


class Uploader(object):

    def __init__(self):
        self.initPyimgur()

    def initPyimgur(self):
        """
        Initialize variables and parameters for PyImgur.
        """
        # self.album = "iX0uj"  # Testing.
        self.album = "3mdlF"
        # self.album = "zzf6O"
        # self.album = "6U86u"
        # self.album = "JugqY"
        _url = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
        _key = '1cc9418279ff4b2683b5050cfa6f3785'
        _maxNumRetries = 10
        CLIENT_ID = "525d3ebc1147459"
        CLIENT_SECRET = "75b8f9b449462150f374ae68f10154f2f392aa9b"
        # ACCESS_TOKEN = "f6d9b3f3121cc259570acb4e4e1626cc53816d5b"
        # ACCOUNT_ID = "36381482"

        self.im = pyimgur.Imgur(CLIENT_ID, CLIENT_SECRET)
        self.im.change_authentication(
            refresh_token="814bed15fbea91dbc6131205f881fc45f8ee0715")
        try:
            self.im.refresh_access_token()
        except:
            print("Can't connect to internet")
        else:
            pass
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
