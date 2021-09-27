from .twitterService import *
import tweepy
from .twitterLocalModel import *


def saveStream(query, db):
    class StreamListener(tweepy.StreamListener):
        def on_status(self, tweet):
            saveTweet(tweet, db)

        def on_error(self, status_code):
            if status_code == 420:
                # returning False in on_data disconnects the stream
                return False

    connectListener(StreamListener(), query)


def getTotal(db):
    return readTweetSize(db)


def saveQueryResult(config, db):
    count = config["total"] if config["total"] else 100
    search = config["search"]
    total = []
    print(f"searching for {count} results for query <{search}>")
    while count > 0:
        tweets = queryAPI(search, min(100, count))
        lenTweets = len(tweets)
        if lenTweets == 0:
            break
        count = count - lenTweets
        newTweets = saveTweets(tweets, db)
        total.extend(newTweets)
    return total


def saveConfig(config, db):
    setConfigDB(config, db)


def deleteQueryResults(db):
    deleteTweetsDB(db)


def getResults(db, query):
    return readTweetsDB(db, query)


def getCurrentConfig(db):
    return getLastConfig(db)
