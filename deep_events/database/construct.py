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
training_path = os.path.join(MAIN_PATH, "event_data")
print(training_path)
event_list = Path(training_path).rglob("*/event_db.yaml")
event_list = list(event_list)

print(event_list)
#%% Reset the database and add all of the events
coll.delete_many({})
for event in event_list:
    event_dict = benedict(str(event))
    coll.insert_one(event_dict)

#%%
filtered_list = list(coll.find({'microscope': ['isim'], 'cell_line': 'cos7'}))


db_files = []
for item in filtered_list:
    db_files.append(Path(item['event_path']) / "event_db.yaml")

print("\n".join(db_files))