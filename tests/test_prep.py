from pathlib import Path
import yaml
import shutil
from deep_events.prepare_training import prepare_for_prompt


FILEPATH = Path(__file__).parents[0]
with open(FILEPATH / "test_database/settings.yaml", "r") as stream:
    COLLECTION = yaml.safe_load(stream)["COLLECTION"]

def test_prep_prompt():
    folder = FILEPATH / "data/event_data"
    out_folder = prepare_for_prompt(folder, {"contrast": "fluorescence"}, COLLECTION)
    assert isinstance(out_folder, Path)
    shutil.rmtree(folder.parents[0]/"training_data")