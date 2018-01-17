import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import string
import unicodedata
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import nltk
import scipy as scs
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
import pickle

"""
Input: pd dataframe of raw features with no 'acct_type'.
Can be applied to training AND testing data.
"""


def delivery_method_categorize(data):
    data['delivery_method_0'] = data['delivery_method'] == 0
    data['delivery_method_1'] = data['delivery_method'] == 1
    data['delivery_method_3'] = data['delivery_method'] == 3
    return data


def payout_type_categorize(data):
    data['payout_type_check'] = data['payout_type'] == 'CHECK'
    data['payout_type_ach'] = data['payout_type'] == 'ACH'
    return data


def currency_categorize(data):
    data['usd'] = data['currency'] == 'USD'
    data['gbp'] = data['currency'] == 'GBP'
    data['cad'] = data['currency'] == 'CAD'
    data['aud'] = data['currency'] == 'AUD'
    data['eur'] = data['currency'] == 'EUR'
    data['nzd'] = data['currency'] == 'NZD'
    return data


def user_type_categorize(data):
    data['user_type_1'] = data['user_type'] == 1
    data['user_type_2'] = data['user_type'] == 2
    data['user_type_3'] = data['user_type'] == 3
    data['user_type_4'] = data['user_type'] == 4
    data['user_type_5'] = data['user_type'] == 5
    return data


def email_categorize(data):
    """
    Define a "rare_email" domain as one that occurs one or zero times within the
    training data.
    """
    emails = pd.DataFrame(data['email_domain'].value_counts() <= 1)
    emails['rare_email'] = emails['email_domain']
    common_emails = emails.index[emails['rare_email'] == False]
    data['rare_email'] = [domain not in common_emails for domain in data['email_domain']]
    return data


def event_data(data):
    """
    Calculate event duration from event end and start timestamps.
    """
    data['event_duration'] = data['event_end'] - data['event_start']
    return data


def listed_categorize(data):
    """
    Categorizes the 'listed' column in the pandas dataframe.

    INPUT:
        - data: pandas dataframe with 'listed' column as 'y' or 'n'

    OUTPUT:
        - data: pandas dataframe with 'listed' column replaced with booleans
    """
    data['listed'] = data['listed'] == 'y'
    return data


def country_data(data):
    """
    Takes a pandas dataframe and does some undetermined stuff with the countries

    INPUT:
        - data: pandas dataframe to get country data from and add engineered
                columns to.

    OUTPUT:
        - data: pandas dataframe with engineered country features added.
    """
    data['venue_country_change'] = (data['venue_country'] != data['country'])
    data['is_us'] = data['country'] == 'US'
    data['is_gb'] = data['country'] == 'GB'
    data['is_ca'] = data['country'] == 'CA'
    return data


def final_columns(data):
    wanted_columns = ['delivery_method_0', 'delivery_method_1', 'delivery_method_3',
                      'payout_type_check', 'payout_type_ach', 'usd', 'gbp', 'cad',
                      'aud', 'eur', 'nzd', 'user_type_1', 'user_type_2', 'user_type_3',
                      'user_type_4', 'user_type_5', 'rare_email', 'event_duration',
                      'listed', 'venue_country_change','is_us', 'is_gb', 'is_ca',
                      'body_length', 'channels', 'fb_published',
                      'has_analytics', 'has_logo', 'listed', 'name_length', 'show_map',
                      'user_age']
    data = data[wanted_columns]
    return data


def clean_data(data):
    """
    Cleans the entire data set.

    INPUT:
        - data: dataframe
    """
    clean_data = delivery_method_categorize(data)
    clean_data = country_data(data)
    clean_data = listed_categorize(data)
    clean_data = event_data(data)
    clean_data = email_categorize(data)
    clean_data = user_type_categorize(data)
    clean_data = currency_categorize(data)
    clean_data = payout_type_categorize(data)

    clean_data = final_columns(data)

    return clean_data
