from pathlib import Path
import shutil

from deep_events.gaussians_to_training import extract_events


FILEPATH = Path(__file__).parents[0]


def test_extract_events():
    folder = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl"
    event_folder = folder.parents[0]/"test_event_data"
    extract_events(folder/"GFP_1/db.yaml", events_folder = event_folder)
    extract_events(folder/"db.yaml", events_folder = event_folder)
    db = list(event_folder.rglob("*event_db.yaml"))
    assert len(db) == 1
    gt = list(event_folder.rglob("*ground_truth.tif"))
    assert len(gt) == 1
    im = list(event_folder.rglob("*images.tif"))
    assert len(im) == 1
    shutil.rmtree(folder.parents[0]/"test_event_data")