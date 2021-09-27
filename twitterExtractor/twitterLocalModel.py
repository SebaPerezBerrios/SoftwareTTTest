import json
from .settings import *
from python_dict_wrapper import unwrap
from sqlalchemy.exc import ProgrammingError
from datetime import datetime

from dateutil import parser


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
    try:
        tweet = tweet.retweeted_status
        return tweetToDB(tweet.retweeted_status)
    except:
        return dict(
            id=tweet.id,
            user=tweet.user.name,
            user_code=tweet.user.screen_name,
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
    newTweets = []
    try:
        for tweet in tweetsToSave:
            if table.find_one(id=tweet["id"]) is None:
                table.insert(tweet, ["id"])
                newTweets.append(tweet)
        return newTweets
    except ProgrammingError as err:
        print(err)


def deleteTweetsDB(db):
    table = db["DEV_TABLE"]
    try:
        table.delete()
    except ProgrammingError as err:
        print(err)


def readTweetsDB(db, query):
    table = db["DEV_TABLE"]
    try:
        if query is None:
            return table.all()
        else:
            dateStart = query.get("dateStart")
            dateEnd = query.get("dateEnd")
            retweetMin = query.get("retweetMin")
            userFollowerMin = query.get("followerMin")
            dateDict = {}

            queryDict = {}

            if dateStart is not None:
                dateDict[">="] = parser.parse(dateStart).strftime("%a %b %d %X %z %Y")
                queryDict["created_at"] = dateDict
            if dateEnd is not None:
                dateDict["<="] = parser.parse(dateEnd).strftime("%a %b %d %X %z %Y")
                queryDict["created_at"] = dateDict

            if retweetMin is not None:
                queryDict["retweet_count"] = {">=": 5}
            if userFollowerMin is not None:
                queryDict["user_followers"] = {">=": userFollowerMin}

            return table.find(**queryDict)
    except ProgrammingError as err:
        print(err)


def readTweetSize(db):
    table = db["DEV_TABLE"]
    try:
        return list(db.query("SELECT COUNT(*) count FROM DEV_TABLE"))[0]["count"]
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
