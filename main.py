"""This module serves as the entry point of TwitterExtractor."""
import TwitterExtractor.twitterDataVisualization as twitterDataVisualization
import TwitterExtractor.twitterLocalController as twitterLocalController
import TwitterExtractor.settings as settings
import timeit


def main():
    # save to db
    twitterLocalController.saveQueryResult(
        " OR ".join(settings.TRACK_TERMS), settings.TRACK_LIMIT)
    # get wordCloud
    return

    # meassure corpus processing
    corpusTime = timeit.timeit(
        lambda: twitterDataVisualization.processData(), number=100)
    print(f"corpus time: {corpusTime}")

    corpus, tweets = twitterDataVisualization.processData()

    # kmeansTFIDFTime = timeit.timeit(
    #     lambda: twitterDataVisualization.kmeansTFIDF(corpus, 10), number=100)
    # print(f"Kmeans (TFIDF) time: {kmeansTFIDFTime}")
    # kmeansWord2VecTime = timeit.timeit(
    #     lambda: twitterDataVisualization.kmeansDoc2Vec(corpus, 10), number=100)
    # print(f"Kmeans (Word2Vec) time: {kmeansWord2VecTime}")

    kmeans, closests = twitterDataVisualization.kmeansTFIDF(corpus, 5)
    print(closests)
    for index in closests:
        print(f"{index}: {tweets[index]}")
    # twitterDataVisualization.printClusters(kmeans, corpus)

    #twitterDataVisualization.kmeansTFIDF(corpus, 5)
    #twitterDataVisualization.kmeansDoc2Vec(corpus, 5)
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
