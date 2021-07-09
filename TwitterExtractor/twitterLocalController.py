from .twitterService import *
import tweepy
from .twitterLocalModel import *


def saveStream(query):
    class StreamListener(tweepy.StreamListener):
        def on_status(self, tweet):
            saveTweet(tweet)

        def on_error(self, status_code):
            if status_code == 420:
                # returning False in on_data disconnects the stream
                return False

    connectListener(StreamListener(), query)


def saveQueryResult(query, count):
    print(f"searching for {count} results for query <{query}>")
    while count > 0:
        tweets = queryAPI(query, min(100, count))
        lenTweets = len(tweets)
        if lenTweets == 0:
            break
        count = count - lenTweets
        for tweet in tweets:
            saveTweet(tweet)


def getResults():
    return readTweets()
