from gensim.models import Doc2Vec
import gensim
import numpy as np
import somoclu
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from scipy.cluster.vq import vq

from sentiment_analysis_spanish import sentiment_analysis
import statistics

from nltk.corpus import stopwords

from .twitterLocalController import getResults
from .settings import *

# Libraries for text preprocessing
import re
import nltk

nltk.download("stopwords")
nltk.download("wordnet")


stopWords = (
    set(stopwords.words("spanish"))
    .union(set(stopwords.words("english")))
    .union(stopWords)
)


def getMetricsClusterUser(tweets):

    totalTweets = 0

    retweets = []
    users = []

    def getStatistics(values):
        return {
            "avg": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) >= 2 else 0,
            "median": statistics.median(values),
        }

    for tweet in tweets:

        totalTweets = totalTweets + 1
        retweets.append(tweet["retweet_count"])
        users.append(tweet["user_followers"])

    return {
        "totalTweets": totalTweets,
        "retweets": getStatistics(retweets),
        "users": getStatistics(users),
    }


def getMetricsNewData(tweets, alertWords, processor):
    corpus, tweets = processTweets(tweets, processor)
    alertSet = set([processor(word) for word in alertWords])
    alertTweets = []

    for tweet in tweets:
        alertCountTweet = len(alertSet.intersection(tweet["processed"]))

        if alertCountTweet > 0:
            alertTweets.append(
                {
                    "text": tweet["text"],
                    "user": tweet["user"],
                    "created_at": tweet["created_at"],
                    "user_location": tweet["user_location"],
                }
            )

    return alertTweets


def getMetricsCluster(tweets, alertWords, analyzer=None):

    alertSet = set(alertWords)
    alertCount = 0
    alertTweets = []
    totalTweets = 0
    retweets = []
    sentiments = []

    def getStatistics(values):
        return {
            "avg": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) >= 2 else None,
            "median": statistics.median(values),
        }

    def brackets(values):
        res = [0, 0, 0, 0, 0]
        for x in values:
            if x < 0.2:
                res[0] += 1
                continue
            if x < 0.4:
                res[1] += 1
                continue
            if x < 0.6:
                res[2] += 1
                continue
            if x < 0.8:
                res[3] += 1
                continue
            res[4] += 1
        return res

    for tweet in tweets:
        alertCountTweet = len(alertSet.intersection(tweet["processed"]))

        if analyzer is not None:
            analysis = analyzer(tweet["processedText"])
            tweet["sentiment"] = analysis
            sentiments.append(analysis)

        if alertCountTweet > 0:
            alertTweets.append(
                {
                    "text": tweet["text"],
                    "user": tweet["user"],
                    "created_at": tweet["created_at"],
                    "user_location": tweet["user_location"],
                }
            )

        tweet["alertCount"] = alertCountTweet

        totalTweets = totalTweets + 1
        alertCount = alertCount + alertCountTweet

        retweets.append(tweet["retweet_count"])

    if analyzer is not None:
        sentimentsTweets = sorted(tweets, key=lambda tweet: tweet["sentiment"])
        best = list(
            map(
                lambda tweet: {
                    "text": tweet["text"],
                    "user": tweet["user"],
                    "created_at": tweet["created_at"],
                    "user_location": tweet["user_location"],
                },
                sentimentsTweets[-5:],
            )
        )
        worst = list(
            map(
                lambda tweet: {
                    "text": tweet["text"],
                    "user": tweet["user"],
                    "created_at": tweet["created_at"],
                    "user_location": tweet["user_location"],
                },
                sentimentsTweets[0:5],
            )
        )

        return {
            "totalTweets": totalTweets,
            "alertCount": alertCount,
            "alertTweets": alertTweets,
            "retweets": getStatistics(retweets),
            "sentiments": getStatistics(sentiments),
            "best": best,
            "worst": worst,
            "summary": brackets(sentiments),
        }

    return {
        "totalTweets": totalTweets,
        "alertCount": alertCount,
        "alertTweets": alertTweets,
        "retweets": getStatistics(retweets),
    }


def getMetrics(model, tweets, alertWords, sentimentAnalysis=False):

    analyzer = (
        sentiment_analysis.SentimentAnalysisSpanish().sentiment
        if sentimentAnalysis
        else None
    )
    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(tweets[i])

    metricClusters = collections.defaultdict()
    for label, tweets in clusters.items():
        metricClusters[label.item()] = getMetricsCluster(tweets, alertWords, analyzer)

    return metricClusters


def getMetricsUsers(model, tweets):

    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(tweets[i])

    metricClusters = collections.defaultdict()
    for label, tweets in clusters.items():
        metricClusters[label.item()] = getMetricsClusterUser(tweets)

    return metricClusters


def kmeansTFIDF(corpus, k):
    vectorizer = TfidfVectorizer(stop_words=list(stopWords))
    wordVector = vectorizer.fit_transform(corpus)
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100)
    model.fit(wordVector)
    closest, distance = vq(model.cluster_centers_, wordVector.todense())
    return model, closest


def kmeansUsers(tweets, k):
    vector = np.array(
        list(
            map(lambda tweet: [tweet["retweet_count"], tweet["user_followers"]], tweets)
        )
    )
    normalizedVector = vector / np.linalg.norm(vector)
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100)
    model.fit(normalizedVector)
    closest, distance = vq(model.cluster_centers_, normalizedVector)
    return model, closest


def kmeansDoc2Vec(corpus, k):
    wordVector = doc2vec(corpus)
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100)
    model.fit(wordVector)
    closest, distance = vq(model.cluster_centers_, wordVector)
    return model, closest


def pca_fun(n_components, data):
    pca = PCA(n_components=n_components).fit(data)
    data = pca.transform(data)
    return data


def som(data):
    som = somoclu.Somoclu(50, 50, data=data, maptype="toroid")
    som.train(data)
    return som


def doc2vec(corpus):
    document_tagged = []
    tagged_count = 0
    for _ in corpus:
        document_tagged.append(gensim.models.doc2vec.TaggedDocument(_, [tagged_count]))
        tagged_count += 1
    d2v = Doc2Vec(document_tagged)
    d2v.train(document_tagged, epochs=d2v.epochs, total_examples=d2v.corpus_count)
    return d2v.docvecs.vectors


def SABayes(corpus):
    sentimentAnalysis = sentiment_analysis.SentimentAnalysisSpanish()
    return [(sentimentAnalysis.sentiment(tweet), tweet) for tweet in corpus]


def SALexicom(corpus):
    sentimentAnalysis = SentiLeak()
    return [sentimentAnalysis.compute_sentiment(tweet) for tweet in corpus]


def som_test(corpus):
    # vectorizer = TfidfVectorizer(stop_words=list(stopWords))
    # wordVector = vectorizer.fit_transform(corpus)

    data = doc2vec(corpus)
    data = pca_fun(2, data)
    res = som(data)
    res.view_component_planes()


def showWordCloud(dataList):
    wordcloud = WordCloud(
        background_color="white",
        stopwords=stopWords,
        max_words=300,
        max_font_size=200,
    ).generate(str(dataList))

    figName = "wordCloud " + ", ".join(TRACK_TERMS)
    fig = plt.figure(figName)
    plt.imshow(wordcloud)
    plt.axis("off")
    fig.savefig(figName, dpi=200)
    plt.clf()


def getTopWords(corpus, model, wordNumber=None, ngrams=(1, 1)):
    def getTopWordsCluster(corpus):
        vectorizer = CountVectorizer(ngram_range=ngrams, max_features=2000).fit(corpus)
        wordVector = vectorizer.transform(corpus)
        totalWords = wordVector.sum(axis=0)
        wordFrequency = [
            (word, int(totalWords[0, idx]))
            for word, idx in vectorizer.vocabulary_.items()
        ]
        wordFrequency = sorted(wordFrequency, key=lambda x: x[1], reverse=True)
        return wordFrequency[
            :wordNumber
        ]  # Convert most freq words to dataframe for plotting bar plot

    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(corpus[i])

    topWordsClusters = collections.defaultdict()

    for label, clusterCorpus in clusters.items():
        topWordsClusters[label.item()] = getTopWordsCluster(clusterCorpus)
    return topWordsClusters


def showWordCount(wordFrequency, index):
    topWordsDataFrame = pd.DataFrame(wordFrequency)
    # Barplot of most freq words
    topWordsDataFrame.columns = ["Termino", "Frecuencia"]

    # sns.set(rc={"figure.figsize": (20, 10)})
    plt.figure(figsize=(20, 10))

    wordGraphics = sns.barplot(x="Termino", y="Frecuencia", data=topWordsDataFrame)
    wordGraphics.set_xticklabels(wordGraphics.get_xticklabels(), rotation=30)
    figName = "wordCount " + ", ".join(TRACK_TERMS) + f" {index}"
    plt.savefig(figName, dpi=200)
    plt.clf()


def lemmatizer():
    lem = WordNetLemmatizer()
    return lem.lemmatize


def stemmer():
    ps = PorterStemmer()
    return ps.stem


def processTweets(tweets, processor):
    corpus = []
    rawCorpus = []
    totalWords = 0

    for tweet in tweets:
        text = tweet["text"]
        # remove links
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"http?://\S+", "", text)
        # separate compound words
        text = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
        # remove all non letters
        text = re.sub("[^a-zA-ZÑñÁáÉéÍíÓóÚú]", " ", text)

        # Convert to lowercase
        text = text.lower()

        processedText = text

        # Convert to list from string
        text = text.split()
        text = [
            processor(word)
            for word in text
            if not word in stopWords and len(word) >= minWordLength
        ]
        if len(text) > 3:
            tweet["processed"] = text
            totalWords = totalWords + len(text)
            text = " ".join(text)
            corpus.append(text)
            tweet["processedText"] = processedText
            rawCorpus.append(tweet)
    return corpus, rawCorpus


def processData(processor, db, query=None):
    tweets = getResults(db, query)
    return processTweets(tweets, processor)
