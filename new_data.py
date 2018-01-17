import numpy as np
import pandas as pd
import pickle
import time
from data_cleanup import *
from pymongo import MongoClient
import requests
from model import Model

client = MongoClient('mongodb://localhost:27017/')
db = client.events_fraud
events = db.events

def get_predict_save(sequence_number=1):
    """
    Get data from the url, put it into a dataframe.
    """
    while True:
        api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
        url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'
        response = requests.post(url, json={'api_key': api_key,
                                            'sequence_number': sequence_number})
        raw_data = response.json()
        if raw_data['data'] != []:
            print('New Data has Arrived!')
            raw_data_df = pd.DataFrame(raw_data['data'])
            event_ids = pd.DataFrame(raw_data_df['object_id'])
            cleaned_live_data = _clean_event(raw_data_df)
            predictions = _make_prediction(cleaned_live_data)
            final_data = pd.DataFrame(predictions, columns=['fraud', 'tos', 'spam', 'largest'])
            final_data['event_id'] = event_ids
            final_data_json = final_data.to_dict(orient='records')
            events.insert_many(final_data_json)

        sequence_number = raw_data['_next_sequence_number']
        time.sleep(60)


def _clean_event(event):
    """
    Run the results of the GET request through the cleaning pipeline.
    Can process multiple events at once.
    """
    return clean_live_data(event)

def _make_prediction(cleaned_events):
    """
    Use the predict method of the previously fit & pickled model
    """
    #model = pickle.load(open('model.pkl', 'br')) # Unpickle the model
    #if len(cleaned_events) == 1:
    #    label_probs = model.predict_proba(cleaned_events.reshape((1, -1)))  # Print the label probability
    #else:
    label_probs = model.predict_proba(cleaned_events)
    return label_probs

if __name__ == "__main__":
    # data = pd.read_json('../data/data.json')
    # clean_train = clean_train_data(data)
    # y = clean_train.pop('label')
    # model = Model()
    # print('------Fitting model------')
    # model.fit(clean_train, y)
    # print('Model has been fit!')

    # We need to add pickle functionality here

    events.delete_many({})
    print('Acquiring data')
    get_predict_save()
