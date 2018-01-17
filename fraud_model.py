import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle
from nlp import *
from data_cleanup import *
import json


class FraudModel(object):
    def __init__(self, alpha=0.1, n_jobs=-1, max_features='sqrt', n_estimators=1000,
                 RandomForest=True, KMeansFeatures=True, NaiveBayes=True):
        """
        INPUT:
        - alpha = Additive laplace smoothing parameter for NaiveBayes
        - n_jobs = Number of jobs to run RFC on
        - max_features = Number of featres to consider on RFC
        - n_estimators = Number of trees in RFC
        - RandomForest = Bool, run RFC
        - KMeansFeatures = Bool, include K means features in RFC
        - NaiveBayes = Bool, run MNB

        ATTRIBUTES:
        - RFC = Random Forest Classifier
        - MNB = Multinomial Naive Bayes Classifier
        """
        self.RFC = RandomForestClassifier(n_jobs=n_jobs, max_features=max_features,
                                          n_estimators=n_estimators)
        self.MNB = MultinomialNB(alpha=alpha)
        self.LogR = LogisticRegression()
        self.STK = StackingClassifier(
            classifiers=[self.RFC, self.MNB], meta_classifier=self.LogR, use_probas=True)

        self.RandomForest = RandomForest
        self.KMeansFeatures = KMeansFeatures
        self.NaiveBayes = NaiveBayes

    def fit(self, X, y):
        """
        INPUT:
        - X: dataframe representing feature matrix for training data
        - y: series representing labels for training data
        """

        # NLP
        if self.KMeansFeatures == True or self.NaiveBayes == True:
            desc_no_html = update_data_frame(X)
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=2)
            word_counts = self.tfidf.fit_transform(desc_no_html['description_no_HTML'])

            if self.KMeansFeatures == True:
                # K-means
                desc_kmeans = KMeans(n_clusters=5, random_state=56, n_jobs=-1)
                desc_kmeans.fit(word_counts)
                self.cluster_centers = desc_kmeans.cluster_centers_
                self.X_cluster = compute_cluster_distance(word_counts, self.cluster_centers)
                self.RF_X = pd.merge(self.X_cluster, X, left_index=True,
                                right_index=True).drop(columns=['description'])
        else:
            RF_X = X.drop(columns=['description'])

        # Random Forest
        #if self.RandomForest == True:
            # Random Forest
            #self.RFC.fit(RF_X, y)


        if self.NaiveBayes == True:
            # Naive Bayes
            self.MNB.fit(word_counts, y)

        # Stacked Classifier
        if self.RandomForest == True and self.NaiveBayes == True:
            RFCpipeline = make_pipeline(RF_X,
                                        self.RFC)

            MNBpipeline = make_pipeline(word_counts,
                                        self.MNB)

            self.STK.fit(y, classifiers=[RFCpipeline, MNBpipeline])

    def predict_proba(self, X):
        """
        INPUT:
        - X: dataframe representing feature matrix for data

        OUTPUT:
        - blah
        """
        if self.KMeansFeatures == True or self.NaiveBayes == True:
            desc_no_html = update_data_frame(X)
            word_counts = self.tfidf.transform(desc_no_html['description_no_HTML'])

            if self.KMeansFeatures == True:
                X_cluster = compute_cluster_distance(word_counts, self.cluster_centers)
                RF_X = pd.merge(X_cluster, X, left_index=True,
                                right_index=True).drop(columns=['description'])
        else:
            RF_X = X.drop(columns=['description'])

        if self.RandomForest == True and self.NaiveBayes == False:
            RFC_preds = self.RFC.predict_proba(RF_X)
            return RFC_preds
        elif self.RandomForest == False and self.NaiveBayes == True:
            NB_preds = self.MNB.predict_proba(word_counts)
            return NB_preds
        elif self.RandomForest == True and self.NaiveBayes == True:
            STK_preds = self.STK.predict_proba(X)
            return STK_preds

    def _log_loss(self, y_true, ):
        pass


def get_data(datafile):
    df = pd.read_json(datafile)
    X = clean_data(df)
    # clean X data
    y = _get_labels(df)
    return X, y


def _get_labels(df):
    acc_type_dict = {'fraudster': 'fraud',
                     'fraudster_att': 'fraud',
                     'fraudster_event': 'fraud',
                     'premium': 'premium',
                     'spammer': 'spam',
                     'spammer_limited': 'spam',
                     'spammer_noinvite': 'spam',
                     'spammer_warn': 'spam',
                     'spammer_web': 'spam',
                     'tos_lock': 'tos',
                     'tos_warn': 'tos',
                     'locked': 'tos'}

    df['acct_label'] = df['acct_type'].map(acc_type_dict)
    return df['acct_label']


if __name__ == '__main__':
    train_X, train_y = get_data('data/train_data.json')
    fraud_model = FraudModel()
    fraud_model.fit(train_X, train_y)
    with open('fraud_model.json', 'w') as f:
        # Write the model to a file as a JSON string .
        json.dump(fraud_model.json, f)
