TRACK_TERMS = ["100xCientoSinLetraChica"]
CONNECTION_STRING = "sqlite:///tweets.db"
TABLE_NAME = "sentimientos_retiro_100"

stopWords = set(["tweet", "twitter"])
minWordLength = 3

try:
    from private import *
except Exception:
    pass
