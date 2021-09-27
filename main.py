"""This module serves as the entry point of TwitterExtractor."""
import timeit

from twitterExtractor.twitterDataProcessing import *


def benchmarkCorpusProcessing():
    corpusTime = timeit.timeit(
        lambda: processData(lemmatizer()),
        number=100,
    )
    print(f"corpus time with lemmatizer: {corpusTime}")
    corpusTime = timeit.timeit(
        lambda: processData(lemmatizer()),
        number=100,
    )
    print(f"corpus time with stemmer: {corpusTime}")


def benchmarkVectorization(corpus):
    kmeansTFIDFTime = timeit.timeit(lambda: kmeansTFIDF(corpus, 5), number=100)
    print(f"Kmeans (TFIDF) time: {kmeansTFIDFTime}")
    kmeansWord2VecTime = timeit.timeit(lambda: kmeansDoc2Vec(corpus, 5), number=100)
    print(f"Kmeans (Word2Vec) time: {kmeansWord2VecTime}")


def benchmarkGSDMM(corpus):
    GSDMMTime = timeit.timeit(lambda: GSDMM(corpus), number=100)
    print(f"GSDMM time: {GSDMMTime}")


def benchmarkSABayes(corpus):
    SABayesTime = timeit.timeit(lambda: SABayes(corpus), number=100)
    print(f"Sentiment Analysis time: {SABayesTime}")


def main():
    # save to db
    # twitterLocalController.saveQueryResult(
    #    " OR ".join(settings.TRACK_TERMS), settings.TRACK_LIMIT)
    # get wordCloud

    benchmarkCorpusProcessing()

    corpus, tweets, avgTweetLength = TwitterProcessing.processData(
        TwitterProcessing.lemmatizer()
    )

    print(f"Avg tweet length: {avgTweetLength}")

    benchmarkVectorization(corpus)

    benchmarkSABayes(corpus)

    # kmeans, closests = TwitterProcessing.kmeansDoc2Vec(corpus, 5)
    # print(closests)
    # for index in closests:
    #    print(f"{index}: {tweets[index]}")

    benchmarkGSDMM(corpus)

    # twitterDataVisualization.printClusters(kmeans, corpus)

    # twitterDataVisualization.kmeansTFIDF(corpus, 5)
    # twitterDataVisualization.kmeansDoc2Vec(corpus, 5)
    # twitterDataVisualization.som_test(corpus)

    # twitterDataVisualization.showWordCloud(corpus)
    # wordVector = twitterDataVisualization.getTopWords(corpus, 30)
    # twitterDataVisualization.showWordCount(wordVector, 1)
    # wordVector = twitterDataVisualization.getTopWords(corpus, 30, (1, 2))
    # twitterDataVisualization.showWordCount(wordVector, 2)
    # wordVector = twitterDataVisualization.getTopWords(corpus, 30, (2, 2))
    # twitterDataVisualization.showWordCount(wordVector, 3)


if __name__ == "__main__":
    main()
