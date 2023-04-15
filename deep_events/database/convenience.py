from pymongo import MongoClient
from benedict import benedict
from pathlib import Path
SETTINGS = benedict(Path(__file__).parents[0] / "settings.yaml")

def get_collection(name):
    cluster = SETTINGS["CLUSTER"]
    client = MongoClient(cluster)
    return client.deep_events[name]
