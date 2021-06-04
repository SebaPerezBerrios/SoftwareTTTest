import settings
from python_dict_wrapper import unwrap
import dataset
from sqlalchemy.exc import ProgrammingError

db = dataset.connect(settings.CONNECTION_STRING)


def saveTweet(twitterObject):
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

    if coordinates is not None:
        coordinates = unwrap(coordinates)

    table = db[settings.TABLE_NAME]
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


def readTweets():
    table = db[settings.TABLE_NAME]
    try:
        return table.all()
    except ProgrammingError as err:
        print(err)