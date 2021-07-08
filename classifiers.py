import pickle
from abc import ABC, abstractmethod
import flair


class Classifier(ABC):

    @abstractmethod
    def predict(self, x):
        raise NotImplemented

class SpamClassifier(Classifier):
    def __init__(self):
        with open('models/spam_dataset_models/spam_classifier', 'rb') as training_model:
            self.model = pickle.load(training_model)
        with open('models/spam_dataset_models/spam_vectorizer', 'rb') as vectorizer_model:
            self.vectorizer = pickle.load(vectorizer_model)


    #x should be string
    def predict(self, x):
        x_vec = self.vectorizer.transform([x.replace(' , ', ', ').replace(' .', '.')])
        pred_class = self.model.predict(x_vec)[0]
        pred_proba = max(self.model.predict_proba(x_vec)[0])
        return pred_class, pred_proba


class DBpediaClassifier(Classifier):
    def __init__(self):
        with open('models/DBpedia_dataset_models/text_classifier', 'rb') as training_model:
            self.model = pickle.load(training_model)
        with open('models/DBpedia_dataset_models/text_vectorizer', 'rb') as vectorizer_model:
            self.vectorizer = pickle.load(vectorizer_model)

    #x should be string
    def predict(self, x):
        x_vec = self.vectorizer.transform([x.replace(' , ', ', ').replace(' .', '.')])
        pred_class = self.model.predict(x_vec)[0]
        pred_proba = max(self.model.predict_proba(x_vec)[0])
        return pred_class, pred_proba


class MovieSentimentClassifier(Classifier):
    def __init__(self):
        self.model = flair.models.TextClassifier.load('en-sentiment')

    #x should be string
    def predict(self, x):
        s = flair.data.Sentence(x.replace(' , ', ', ').replace(' .', '.'))
        self.model.predict(s)
        score, label = s.labels[0].score, s.labels[0].value
        if label == 'NEGATIVE':
            label = 0
        elif label == 'POSITIVE':
            label = 1
        else:
            raise ValueError
        return label, score

class TweetSentimentClassifier(Classifier):
    def __init__(self):
        with open('models/twitter_sentiment_models/text_classifier', 'rb') as training_model:
            self.model = pickle.load(training_model)
        with open('models/twitter_sentiment_models/text_vectorizer', 'rb') as vectorizer_model:
            self.vectorizer = pickle.load(vectorizer_model)

    #x should be string
    def predict(self, x):
        x_vec = self.vectorizer.transform([x.replace(' , ', ', ').replace(' .', '.')])
        pred_class = self.model.predict(x_vec)[0]
        pred_proba = max(self.model.predict_proba(x_vec)[0])
        return pred_class, pred_proba

class Stanford140Classifier(Classifier):
    def __init__(self):
        with open('models/stanford_140/text_classifier', 'rb') as training_model:
            self.model = pickle.load(training_model)
        with open('models/stanford_140/text_vectorizer', 'rb') as vectorizer_model:
            self.vectorizer = pickle.load(vectorizer_model)

    #x should be string
    def predict(self, x):
        x_vec = self.vectorizer.transform([x.replace(' , ', ', ').replace(' .', '.')])
        pred_class = self.model.predict(x_vec)[0]
        pred_proba = max(self.model.predict_proba(x_vec)[0])
        return pred_class, pred_proba


class CustomClassifier(Classifier):
    def __init__(self, models_dir):
        with open('{}text_classifier'.format(models_dir), 'rb') as training_model:
            self.model = pickle.load(training_model)
        with open('{}text_vectorizer'.format(models_dir), 'rb') as vectorizer_model:
            self.vectorizer = pickle.load(vectorizer_model)

    #x should be string
    def predict(self, x):
        x_vec = self.vectorizer.transform([x.replace(' , ', ', ').replace(' .', '.')])
        pred_class = self.model.predict(x_vec)[0]
        pred_proba = max(self.model.predict_proba(x_vec)[0])
        return pred_class, pred_proba
