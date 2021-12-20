from gensim.models import Doc2Vec
import gensim
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
from sentileak import SentiLeak
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


def getMetricsCluster(tweets, alertWords, sentimentAnalyzer):

    alertSet = set(alertWords)
    alertCount = 0
    totalTweets = 0

    retweets = []
    sentiments = []

    def getStatistics(values):
        return {
            "avg": statistics.mean(values),
            "std": statistics.stdev(values),
            "median": statistics.median(values),
        }

    for tweet in tweets:
        alertCountTweet = len(alertSet.intersection(tweet["processed"]))
        sentimentAnalysis = sentimentAnalyzer(tweet["processedText"])

        tweet["alertCount"] = alertCountTweet
        tweet["sentimentAnalysis"] = sentimentAnalysis

        totalTweets = totalTweets + 1
        alertCount = alertCount + alertCountTweet

        retweets.append(tweet["retweet_count"])
        sentiments.append(sentimentAnalysis)

    return {
        "totalTweets": totalTweets,
        "alertCount": alertCount,
        "retweets": getStatistics(retweets),
        "sentiments": getStatistics(sentiments),
    }


def getMetrics(model, tweets, alertWords, saType):

    sentimentAnalyzer = (
        sentiment_analysis.SentimentAnalysisSpanish().sentiment
        if saType == "bayes"
        else lambda tweet: SentiLeak().compute_sentiment(tweet)["global_sentiment"]
    )

    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(tweets[i])

    metricClusters = collections.defaultdict()
    for label, tweets in clusters.items():
        metricClusters[label.item()] = getMetricsCluster(
            tweets, alertWords, sentimentAnalyzer
        )

    return metricClusters


def kmeansTFIDF(corpus, k):
    vectorizer = TfidfVectorizer(stop_words=list(stopWords))
    wordVector = vectorizer.fit_transform(corpus)
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100)
    model.fit(wordVector)
    closest, distance = vq(model.cluster_centers_, wordVector.todense())
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


def getTopWords(corpus, tweets, wordNumber=None, ngrams=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngrams, max_features=2000).fit(corpus)
    wordVector = vectorizer.transform(corpus)
    totalWords = wordVector.sum(axis=0)
    wordFrequency = [
        (word, int(totalWords[0, idx])) for word, idx in vectorizer.vocabulary_.items()
    ]
    wordFrequency = sorted(wordFrequency, key=lambda x: x[1], reverse=True)
    return wordFrequency[
        :wordNumber
    ]  # Convert most freq words to dataframe for plotting bar plot


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


def processData(processor, db):
    tweets = getResults(db)
    corpus = []
    rawCorpus = []
    totalWords = 0

    for tweet in tweets:
        text = tweet["text"]
        # remove links
        text = re.sub(r"https?://\S+", "", text)
        # separate compound words
        text = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
        # remove all non letters
        text = re.sub("[^a-zA-ZÑñÁáÉéÍíÓóÚú]", " ", text)

        # Convert to lowercase
        text = text.lower()

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
            tweet["processedText"] = text
            rawCorpus.append(tweet)
    return corpus, rawCorpus
