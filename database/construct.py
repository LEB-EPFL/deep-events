#%%
from extract_yaml import MAIN_PATH
from pathlib import Path
import os
import datetime

from pymongo import MongoClient
from bson.objectid import ObjectId
from benedict import benedict

cluster = "mongodb://lebpc13.epfl.ch/"
client = MongoClient(cluster)

for db in client.list_database_names():
    print(db)
print(client)
db = client.deep_events
coll = db['collection']


#%%
training_path = os.path.join(os.path.dirname(MAIN_PATH), "training_data")

print(training_path)
event_list = Path(training_path).rglob("*/db.yaml")
event_list = list(event_list)
#%%
coll.delete_many({})
for event in event_list:
    event_dict = benedict(str(event))
    coll.insert_one(event_dict)

#%%
filtered_list = list(coll.find({'microscope': 'zeiss', 'contrast': ['bf']}))
for item in filtered_list:
    print(item)
print(len(filtered_list))


# %%
