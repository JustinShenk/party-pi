#!/ usr/bin/env python
import http.client
import json

from urllib.parse import urlencode


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


class Emotion_API(object):

    def __init__(self):
        self.config = get_conf()

    @classmethod
    def get_emotions(self, url_path):
        
        self.config = get_conf()
        """ Send image to Microsoft/Oxford emotion API.

        """
        body = "{'URL':'" + url_path + "'}"
        
        headers = {
            # Request headers.
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.config.emotion_api,
        }

        params = urlencode({
        })

        try:
            conn = http.client.HTTPSConnection('api.projectoxford.ai')
            conn.request("POST", "/emotion/v1.0/recognize?%s" % params, body, headers)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            print(data)
            conn.close()
            json_data = json.loads(data)

            return json_data

        except Exception as e:
            print(("[Errno {}] ".format(e)))

def main():
    Emotion_API()


if __name__ == '__main__':
    # `url_path` to Imgur image.
    main()
