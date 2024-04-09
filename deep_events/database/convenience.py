from pymongo import MongoClient
from benedict import benedict
from pathlib import Path
import datetime
import re

SETTINGS = benedict(Path(__file__).parents[0] / "settings.yaml")

def get_collection(name):
    cluster = SETTINGS["CLUSTER"]
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


def get_latest_folder(parent_folder:Path):
    subfolders = [f for f in parent_folder.glob('*') if f.is_dir()]
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


def glob_zarr(folder: Path, pattern: str = "*.ome.zarr", ignore = r".*/p\d$"):
    "glob that ignores big subfolder structures in zarr files"
    def recurse(folder: Path, results: list, pattern: str, ignore: str):
        if (not folder.is_dir() and "zarr" in pattern) or re.match(ignore, folder.as_posix()):
            return results
        items = list(folder.glob(pattern))
        subfolders = list(folder.glob("*"))
        for item in items:
            subfolders.remove(item)
        for subfolder in subfolders:
            if subfolder.is_dir():
                results = recurse(subfolder, results, pattern, ignore)
            else:
                continue
        results.extend(items)
        return results

    results = []
    return recurse(folder, results, pattern, ignore)

def clean_dates(folder: Path):
    files = folder.rglob("event_db.yaml")
    for file in files:
        event_dict = benedict(file)
        # if event_dict["date"][:2] != "20":
        #     print(event_dict["date"])
        #     event_dict["date"] = "20" + event_dict["date"]
        #     print(event_dict["date"])
        event_dict["date"] = int(event_dict["date"])
        event_dict.to_yaml(filepath=file)
        
            



if __name__ == "__main__":
    # handle_repositioning(Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/event_data"), "", "")
    import time
    t0 = time.perf_counter()
    folder = Path("Z:/deep_events/data/phaseEDA")
    pattern = "*ome.tif*"
    results = glob_zarr(folder, pattern)
    t1 = time.perf_counter()
    print("\n".join([str(x) for x in results]))
    print("done in ", round((t1 - t0)*1000))
    # results = folder.rglob("**/*.ome.zarr")
    # print("\n".join([str(x) for x in results]))
    # print("rglob", round((time.perf_counter() - t1)*1000))