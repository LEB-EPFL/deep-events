from pathlib import Path
from deep_events.database import get_collection
import pymongo
import yaml


FILEPATH = Path(__file__).parents[0]
with open(FILEPATH / "settings.yaml", "r") as stream:
    COLLECTION = yaml.safe_load(stream)["COLLECTION"]

def test_connection():
    assert isinstance(get_collection(COLLECTION), pymongo.database.Collection)