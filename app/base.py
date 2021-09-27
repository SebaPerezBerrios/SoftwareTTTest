from ..twitterExtractor.twitterDataProcessing import (
    SABayes,
    getMetrics,
    getTopWords,
    kmeansDoc2Vec,
    kmeansTFIDF,
    lemmatizer,
    processData,
    stemmer,
)
from ..twitterExtractor.twitterLocalController import (
    deleteQueryResults,
    getCurrentConfig,
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
    return jsonify(total=total)


@bp.route("/createMore", methods=(["POST"]))
def createMore():
    db = get_db()
    config = getCurrentConfig(db)
    total = saveQueryResult(config, db)
    return jsonify(total=total)


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
    kMeans = {"TFIDF": kmeansTFIDF, "word2vec": kmeansDoc2Vec}

    try:
        stemmerStrategy = stemmers[body["stemmer"]]
        kMeansStrategy = kMeans[body["vector"]]
        alertWords = body["alertWords"]

    except:
        print(body)
        raise Exception("BAD REQUEST")

    corpus, tweets = processData(stemmerStrategy, db)

    model, closest = kMeansStrategy(corpus, body["groups"])

    metrics = getMetrics(
        model, tweets, [stemmerStrategy(word) for word in alertWords], "bayes"
    )
    data = list(map(lambda index: tweets[index], closest.tolist()))

    for key, value in metrics.items():
        data[key]["metrics"] = value

    resp = jsonify(tweets=data)
    return resp


@bp.route("/wordlist", methods=(["POST"]))
def topWords():
    body = request.json
    db = get_db()

    corpus, tweets = processData(lemmatizer(), db)

    data = getTopWords(corpus, tweets, body["total"], (1, body["ngram"]))

    return jsonify(words=data)


@bp.route("/sentimentanalysis", methods=(["POST"]))
def sentimentanalysis():
    body = request.json
    db = get_db()

    corpus, tweets = processData(lemmatizer(), db)

    data = SABayes(corpus)  # SALexicom(corpus)

    return jsonify(sentiments=data)
