import settings
import tweepy
from python_dict_wrapper import wrap

auth = tweepy.AppAuthHandler(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


def connectListener(streamListener, query):
    stream = tweepy.Stream(auth=api.auth, listener=streamListener)
    stream.filter(track=settings.TRACK_TERMS)


def tweetObject(tweetDict):
    return wrap(tweetDict)


def queryAPI(query, count):
    print("search api")
    results = api.search(query, count=count)
    if len(results["statuses"]) == 0:
        print("no tweet found")

    print("found {} tweets".format(len(results["statuses"])))

    return list(map(lambda tweet: tweetObject(tweet), results["statuses"]))
