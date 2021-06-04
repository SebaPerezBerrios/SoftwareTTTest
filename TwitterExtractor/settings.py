TRACK_TERMS = ["pacos", "carabineros", "yuta"]
CONNECTION_STRING = "sqlite:///tweets.db"
TABLE_NAME = "sentimientos_fuerza_publica"

stopWords = set(["http", "tweet", "twitter"])
minWordLength = 3

try:
    from private import *
except Exception:
    pass
