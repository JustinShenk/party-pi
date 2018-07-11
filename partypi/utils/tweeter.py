from __future__ import unicode_literals

import os
import tweepy

def twitter_api():
    consumer_key = None
    consumer_secret = None
    access_token = None
    access_token_secret = None
    try:
        consumer_key = os.environ.get('TWITTER_KEY')
        consumer_secret = os.environ.get('TWITTER_SECRET')
        access_token = os.environ.get('TWITTER_TOKEN')
        access_token_secret = os.environ.get('TWITTER_TOKEN_SECRET')
    except:
        print("No twitter auth found")
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # try:
    #     redirect_url = auth.get_authorization_url()
    # except tweepy.TweepError:
    #     print('Error! Failed to get request token.')
    # print(redirect_url)
    # verifier = raw_input('PIN: ').strip()
    # auth.get_access_token(verifier)
    # print 'ACCESS_KEY = "%s"' % auth.access_token.key
    # print 'ACCESS_SECRET = "%s"' % auth.access_token.secret
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

def tweet_message(message):
    api = twitter_api()
    try:
        api.update_status(status=message)
        print("Tweeted: {}".format(message))
    except tweepy.TweepError as e:
        print(e.reason)

def tweet_image(filename, message):
    api = twitter_api()
    api.update_with_media(filename, status=message)
    print("Tweeted: {}".format(message))

if __name__ == '__main__':
    tweet_image('img_car.jpg', 'testing the API!')
