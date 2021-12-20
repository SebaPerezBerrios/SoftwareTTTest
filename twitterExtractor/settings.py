TRACK_LIMIT = 1
CONNECTION_STRING = "sqlite:///tweets.db"

stopWords = set(["tweet", "twitter"])
minWordLength = 3

try:
    from .private import *
except Exception:
    pass
