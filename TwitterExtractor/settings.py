TRACK_TERMS = ["jadue"]
TRACK_LIMIT = 5
CONNECTION_STRING = "sqlite:///tweets.db"
TABLE_NAME = "sentimientos_contingencia"

stopWords = set(["tweet", "twitter"])
minWordLength = 3

try:
    from .private import *
except Exception:
    pass
