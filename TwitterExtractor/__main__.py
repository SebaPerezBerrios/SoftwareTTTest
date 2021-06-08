"""This module serves as the entry point of TwitterExtractor."""
import twitterDataVisualization
import twitterLocalController
import settings


def main():
    # save to db
    twitterLocalController.saveQueryResult(" OR ".join(settings.TRACK_TERMS), 201)
    # get wordCloud
    corpus = twitterDataVisualization.processData()
    twitterDataVisualization.kmeans(corpus, 5)

    twitterDataVisualization.showWordCloud(corpus)
    wordVector = twitterDataVisualization.getTopWords(corpus, 30)
    twitterDataVisualization.showWordCount(wordVector, 1)
    wordVector = twitterDataVisualization.getTopWords(corpus, 30, (1, 2))
    twitterDataVisualization.showWordCount(wordVector, 2)
    wordVector = twitterDataVisualization.getTopWords(corpus, 30, (2, 2))
    twitterDataVisualization.showWordCount(wordVector, 3)


if __name__ == "__main__":
    main()
