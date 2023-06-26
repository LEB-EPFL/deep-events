from pymongo import MongoClient
from benedict import benedict
from pathlib import Path
SETTINGS = benedict(Path(__file__).parents[0] / "settings.yaml")

def get_collection(name):
    cluster = SETTINGS["CLUSTER"]
    client = MongoClient(cluster)
    return client.deep_events[name]


def get_cluster():
    return SETTINGS["CLUSTER"]


def handle_repositioning(folder: Path, old_path:str, new_path:str):
    "replace positions in events. These should be relative in the future ideally"
    "Should only be necessary for manual events anyways."
    collection = get_collection(benedict(folder/'collection.yaml')['collection'])
    manual_events = list(collection.find({"extraction_type": "manual"}))
    paths = [x["original_path"] for x in manual_events]
    print(manual_events[0])
    for event in paths:
        print(event)


if __name__ == "__main__":
    handle_repositioning(Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/event_data"), "", "")