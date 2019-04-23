from __future__ import unicode_literals

import os
import tweepy


def twitter_api(public_account=False):
    consumer_key = None
    consumer_secret = None
    access_token = None
    access_token_secret = None
    if public_account:  # @PlayPartyPi
        try:
            consumer_key = os.environ.get('TWITTER_KEY_PUBLIC')
            consumer_secret = os.environ.get('TWITTER_SECRET_PUBLIC')
            access_token = os.environ.get('TWITTER_TOKEN_PUBLIC')
            access_token_secret = os.environ.get('TWITTER_TOKEN_SECRET_PUBLIC')
        except:
            print("No twitter auth found")
    else:  # Official/private Twitter account
        try:
            consumer_key = os.environ.get('TWITTER_KEY')
            consumer_secret = os.environ.get('TWITTER_SECRET')
            access_token = os.environ.get('TWITTER_TOKEN')
            access_token_secret = os.environ.get('TWITTER_TOKEN_SECRET')
        except:
            print("No twitter auth found")
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def tweet_message(message, public_account=False):
    api = twitter_api(public_account=public_account)
    try:
        api.update_status(status=message)
        print("Tweeted: {}".format(message))
    except tweepy.TweepError as e:
        print(e.reason)


def tweet_image(filename, message, public_account=False):
    api = twitter_api(public_account=public_account)
    api.update_with_media(filename, status=message)
    print("Tweeted: {}".format(message))


if __name__ == '__main__':
    tweet_message("Testing the API!", True)
    # tweet_image('img_car.jpg', 'testing the API!')
