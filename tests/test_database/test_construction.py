from pathlib import Path
import yaml
from deep_events.database import reconstruct_from_folder, get_collection

FILEPATH = Path(__file__).parents[1]
with open(FILEPATH / "test_database/settings.yaml", "r") as stream:
    COLLECTION = yaml.safe_load(stream)["COLLECTION"]

def test_reconstruct():
    folder = FILEPATH / "data/event_data"
    reconstruct_from_folder(folder, COLLECTION)
    coll = get_collection(COLLECTION)
    filtered_list = list(coll.find({}))
    assert len(filtered_list) == 9
