import json
from .settings import *
from python_dict_wrapper import unwrap
from sqlalchemy.exc import ProgrammingError
from datetime import datetime


def saveTweet(twitterObject, db):
    text = twitterObject.text
    coordinates = twitterObject.coordinates
    userLocation = twitterObject.user.location
    followers = twitterObject.user.followers_count
    userStatuses = twitterObject.user.statuses_count
    userLanguage = twitterObject.user.lang
    userId = twitterObject.user.id
    id = twitterObject.id
    created = twitterObject.created_at
    retweets = twitterObject.retweet_count

    table = db["DEV_TABLE"]
    try:
        table.upsert(
            dict(
                id=id,
                text=text,
                coordinates=coordinates,
                created_at=created,
                retweet_count=retweets,
                user_id=userId,
                user_location=userLocation,
                user_language=userLanguage,
                user_followers=followers,
                user_statuses=userStatuses,
            ),
            ["id"],
        )
    except ProgrammingError as err:
        print(err)


def tweetToDB(tweet):
    return dict(
        id=tweet.id,
        text=tweet.text,
        coordinates=json.dumps(unwrap(tweet.coordinates))
        if tweet.coordinates
        else tweet.coordinates,
        created_at=tweet.created_at,
        retweet_count=tweet.retweet_count,
        user_id=tweet.user.id,
        user_location=tweet.user.location,
        user_language=tweet.lang,
        user_followers=tweet.user.followers_count,
        user_statuses=tweet.user.statuses_count,
    )


def saveTweets(twitterObjects, db):
    tweetsToSave = map(lambda tweet: tweetToDB(tweet), twitterObjects)
    table = db["DEV_TABLE"]
    try:
        table.upsert_many(list(tweetsToSave), ["id"])
    except ProgrammingError as err:
        print(err)


def deleteTweetsDB(db):
    table = db["DEV_TABLE"]
    try:
        table.delete()
    except ProgrammingError as err:
        print(err)


def readTweetsDB(db):
    table = db["DEV_TABLE"]
    try:
        return table.all()
    except ProgrammingError as err:
        print(err)


def getLastConfig(db):
    table = db["CONFIG"]
    try:
        results = table.find(order_by=["-created_at"])
        for result in results:
            return result
        return None
    except ProgrammingError as err:
        print(err)


def setConfigDB(config, db):
    table = db["CONFIG"]
    search = config["search"]
    alert = config["alert"]
    total = config["total"]
    try:
        table.upsert(
            dict(search=search, alert=alert, total=total, created_at=datetime.now()),
            ["search"],
        )
    except ProgrammingError as err:
        print(err)
