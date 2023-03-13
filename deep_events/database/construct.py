#%%
from extract_yaml import MAIN_PATH
from pathlib import Path
import os
from pymongo import MongoClient
from benedict import benedict


# Initialize database connection
cluster = "mongodb://lebpc13.epfl.ch/"
client = MongoClient(cluster)
coll = client.deep_events['collection']


#%% Get the db.yaml files from the folders
training_path = os.path.join(os.path.dirname(MAIN_PATH), "training_data")
event_list = Path(training_path).rglob("*/db.yaml")
event_list = list(event_list)

#%% Reset the database and add all of the events
coll.delete_many({})
for event in event_list:
    event_dict = benedict(str(event))
    coll.insert_one(event_dict)

#%%
filtered_list = list(coll.find({'contrast': ['bf']}))