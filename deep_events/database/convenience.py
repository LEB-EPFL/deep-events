from pymongo import MongoClient
from benedict import benedict
from pathlib import Path
import datetime
import re

SETTINGS = benedict(Path(__file__).parents[0] / "settings.yaml")

def get_collection(name):
    cluster = SETTINGS["CLUSTER"]
    print(cluster)
    client = MongoClient(cluster)
    return client.deep_events[name]


def get_cluster():
    return SETTINGS["CLUSTER"]


def handle_repositioning(folder: Path, old_path:str, new_path:str):
    "Replace positions in events. These should be relative in the future ideally"
    "Should only be necessary for manual events anyways."
    collection = get_collection(benedict(folder/'collection.yaml')['collection'])
    manual_events = list(collection.find({"extraction_type": "manual"}))
    paths = [x["original_path"] for x in manual_events]
    print(manual_events[0])
    for event in paths:
        print(event)


def get_latest_folder(parent_folder:Path, pattern:str|tuple = '*'):

    subfolders = [f for f in parent_folder.glob(pattern) if f.is_dir()]
    datetime_format = '%Y%m%d_%H%M'
    subfolders = [f for f in subfolders if f.name.count('_') > 1 and
                  datetime.datetime.strptime("_".join(f.name.split("_")[:2]), datetime_format)]
    subfolders.sort(key=lambda x: datetime.datetime.strptime("_".join(x.name.split("_")[:2]), datetime_format),
                    reverse=True)
    return subfolders if subfolders else None


def get_latest(pattern, folder:Path):
    files = [f for f in folder.glob('*') if f.is_file()]
    files = [f for f in files if pattern in f.name]
    files.sort(reverse=True)
    print(files)
    return folder / files[0]



if __name__ == "__main__":
    handle_repositioning(Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/event_data"), "", "")