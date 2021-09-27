from .settings import *
import tweepy
from python_dict_wrapper import wrap
from tweepy.parsers import JSONParser

auth = tweepy.AppAuthHandler(TWITTER_APP_KEY, TWITTER_APP_SECRET)
api = tweepy.API(auth, parser=JSONParser())


def connectListener(streamListener, query):
    stream = tweepy.Stream(auth=api.auth, listener=streamListener)
    stream.filter(track=TRACK_TERMS)


def tweetObject(tweetDict):
    return wrap(tweetDict)


def queryAPI(query, count):
    print("search api")
    results = dict(api.search_tweets(query, count=count, lang="es"))
    if len(results["statuses"]) == 0:
        print("no tweet found")

    print("found {} tweets".format(len(results["statuses"])))

    return list(map(lambda tweet: tweetObject(tweet), results["statuses"]))
