import settings
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections

# Libraries for text preprocessing
import re
import nltk

nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stopWords = (
    set(stopwords.words("spanish"))
    .union(set(stopwords.words("english")))
    .union(settings.stopWords)
)


def kmeans(corpus, k):
    vectorizer = TfidfVectorizer(stop_words=list(stopWords))
    wordVector = vectorizer.fit_transform(corpus)
    model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)
    model.fit(wordVector)

    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(i)

    print(clusters)

    for label, indexes in clusters.items():
        print(f"{label}: {[corpus[idx] for idx in indexes]}")


def showWordCloud(dataList):
    wordcloud = WordCloud(
        background_color="white",
        stopwords=stopWords,
        max_words=300,
        max_font_size=200,
    ).generate(str(dataList))

    figName = "wordCloud " + ", ".join(settings.TRACK_TERMS)
    fig = plt.figure(figName)
    plt.imshow(wordcloud)
    plt.axis("off")
    fig.savefig(figName, dpi=200)
    plt.clf()


def getTopWords(corpus, wordNumber=None, ngrams=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngrams, max_features=2000).fit(corpus)
    wordVector = vectorizer.transform(corpus)
    totalWords = wordVector.sum(axis=0)
    wordFrequency = [
        (word, totalWords[0, idx]) for word, idx in vectorizer.vocabulary_.items()
    ]
    wordFrequency = sorted(wordFrequency, key=lambda x: x[1], reverse=True)
    return wordFrequency[
        :wordNumber
    ]  # Convert most freq words to dataframe for plotting bar plot


def showWordCount(wordFrequency, index):
    topWordsDataFrame = pd.DataFrame(wordFrequency)
    topWordsDataFrame.columns = ["Termino", "Frecuencia"]  # Barplot of most freq words

    # sns.set(rc={"figure.figsize": (20, 10)})
    plt.figure(figsize=(20, 10))

    wordGraphics = sns.barplot(x="Termino", y="Frecuencia", data=topWordsDataFrame)
    wordGraphics.set_xticklabels(wordGraphics.get_xticklabels(), rotation=30)
    figName = "wordCount " + ", ".join(settings.TRACK_TERMS) + f" {index}"
    plt.savefig(figName, dpi=200)
    plt.clf()


def processData():
    engine = sqlalchemy.create_engine(settings.CONNECTION_STRING)
    tweets = pd.read_sql_table(settings.TABLE_NAME, engine)
    corpus = []

    for tweet in tweets.text:
        text = tweet
        # remove links
        text = re.sub(r"https?://\S+", "", text)
        # separate compound words
        text = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
        # remove all non letters
        text = re.sub("[^a-zA-ZÑñÁáÉéÍíÓóÚú]", " ", text)

        # Convert to lowercase
        text = text.lower()

        ##Convert to list from string
        text = text.split()

        ##Stemming
        # ps=PorterStemmer()    #Lemmatisation
        lem = WordNetLemmatizer()
        text = [
            lem.lemmatize(word)
            for word in text
            if not word in stopWords and len(word) >= settings.minWordLength
        ]
        text = " ".join(text)
        corpus.append(text)

    return corpus
