#!/usr/bin/env python


class NLTKSentiment:
    """
        Use NLTK for sentiment analyzer
    """
    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def __call__(self, sentence):
        """
            Input:
                - sentence: string

            Output:
                - float from -1 to 1
        """
        return self.sentiment_analyzer.polarity_scores(sentence)['compound']


class Sentiment:
    def __new__(cls, choose="nltk", **args):
        """
        TODO: now only support nltk
        """
        return NLTKSentiment(**args)
