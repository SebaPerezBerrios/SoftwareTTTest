from flask_socketio import SocketIO
from nltk.stem import porter
from ..twitterExtractor.twitterDataProcessing import (
    SABayes,
    getMetrics,
    getMetricsNewData,
    getMetricsUsers,
    getTopWords,
    kmeansDoc2Vec,
    kmeansTFIDF,
    kmeansUsers,
    lemmatizer,
    processData,
    stemmer,
)
from ..twitterExtractor.twitterLocalController import (
    deleteQueryResults,
    getCurrentConfig,
    getResults,
    getTotal,
    saveConfig,
    saveQueryResult,
)
from flask import jsonify
from flask import request

from flask import Blueprint
from flask import Response


from .db import get_db

bp = Blueprint("base", __name__, url_prefix="/base")


@bp.route("/create", methods=(["POST"]))
def create():
    body = request.json
    db = get_db()
    deleteQueryResults(db)
    total = saveQueryResult(body, db)
    return jsonify(total=len(total))


@bp.route("/createMore", methods=(["POST"]))
def createMore():
    body = request.json
    db = get_db()

    try:
        alertWords = body["alertWords"]

    except:
        print(body)
        raise Exception("BAD REQUEST")

    config = getCurrentConfig(db)
    prevTotal = getTotal(db)
    newTweets = saveQueryResult(config, db)

    alertTweets = getMetricsNewData(newTweets, alertWords, stemmer())

    return jsonify(new=len(newTweets), prevTotal=prevTotal, tweets=alertTweets)


@bp.route("/getConfig", methods=(["GET"]))
def getConfig():
    db = get_db()
    config = getCurrentConfig(db)
    return jsonify(config=config)


@bp.route("/setConfig", methods=(["POST"]))
def setConfig():
    config = request.json
    db = get_db()
    config = saveConfig(config, db)
    return jsonify(message="OK")


@bp.route("/kmeans", methods=(["POST"]))
def kmeans():

    body = request.json
    db = get_db()

    stemmers = {"porter": stemmer(), "lemmatizer": lemmatizer()}
    kMeans = {"TFIDF": kmeansTFIDF, "Word2Vec": kmeansDoc2Vec}

    try:
        stemmerStrategy = stemmers[body["stemmer"]]
        kMeansStrategy = kMeans[body["vector"]]
        alertWords = body["alertWords"]
        groups = body["groups"]

    except:
        print(body)
        raise Exception("BAD REQUEST")

    corpus, tweets = processData(stemmerStrategy, db)

    model, closest = kMeansStrategy(corpus, groups)

    metrics = getMetrics(
        model,
        tweets,
        [stemmerStrategy(word) for word in alertWords],
    )
    data = list(map(lambda index: tweets[index], closest.tolist()))

    for key, value in metrics.items():
        data[key]["metrics"] = value

    resp = jsonify(tweets=data)
    return resp


@bp.route("/relevant-users", methods=(["POST"]))
def relevantUsers():
    body = request.json
    db = get_db()

    tweets = list(getResults(db, None))

    model, closest = kmeansUsers(tweets, body["groups"])

    metrics = getMetricsUsers(model, tweets)
    data = list(map(lambda index: tweets[index], closest.tolist()))

    for key, value in metrics.items():
        data[key]["metrics"] = value

    resp = jsonify(tweets=data)
    return resp


@bp.route("/wordlist", methods=(["POST"]))
def topWords():
    body = request.json
    db = get_db()

    kMeans = {"TFIDF": kmeansTFIDF, "Word2Vec": kmeansDoc2Vec}

    try:
        kMeansStrategy = kMeans[body["vector"]]

    except:
        print(body)
        raise Exception("BAD REQUEST")

    corpus, tweets = processData(lemmatizer(), db)

    model, closest = kMeansStrategy(corpus, body["groups"])

    data = getTopWords(corpus, model, body["total"], (1, body["ngram"]))

    return jsonify(words=data)


@bp.route("/sentimentanalysis", methods=(["POST"]))
def sentimentanalysis():
    body = request.json
    db = get_db()

    corpus, tweets = processData(lemmatizer(), db)

    data = SABayes(corpus)  # SALexicom(corpus)

    return jsonify(sentiments=data)


@bp.route("/advanced-search", methods=(["POST"]))
def advancedSearch():
    body = request.json
    db = get_db()

    kMeans = {"TFIDF": kmeansTFIDF, "Word2Vec": kmeansDoc2Vec}
    stemmers = {"porter": stemmer(), "lemmatizer": lemmatizer()}

    try:
        kMeansStrategy = kMeans[body["vector"]]
        stemmerStrategy = stemmers[body["stemmer"]]
        alertWords = body["alertWords"]
        groups = body["groups"]
        sentimentAnalysis = body["sentimentAnalysis"]

    except:
        print(body)
        raise Exception("BAD REQUEST")
    corpus, tweets = processData(stemmerStrategy, db, body)

    model, closest = kMeansStrategy(corpus, groups)

    metrics = getMetrics(
        model, tweets, [stemmerStrategy(word) for word in alertWords], sentimentAnalysis
    )
    data = list(map(lambda index: tweets[index], closest.tolist()))

    for key, value in metrics.items():
        data[key]["metrics"] = value

    resp = jsonify(tweets=data)
    return resp
