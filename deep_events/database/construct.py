#%%
from deep_events.database.extract_yaml import MAIN_PATH
from deep_events.database import get_collection
from pathlib import Path
from pymongo import MongoClient
from benedict import benedict

def main(): #pragma: no cover
    reconstruct_from_folder(MAIN_PATH, "mito_events")


def reconstruct_from_folder(folder: Path, collection: str):
    # Initialize database connection
    coll = get_collection(collection)

    #%% Get the db.yaml files from the folders
    if "event_data" in str(folder):
        event_list = list(Path(folder).rglob("*/event_db.yaml"))
        event_dicts = [benedict(str(event)) for event in event_list]
        corrected_event_list = []
        for event_dict, path in zip(event_dicts, event_list):
            event_dict['event_path'] = str(Path(path).parents[0].resolve())
            corrected_event_list.append(event_dict)
        event_list = corrected_event_list
    elif "training_data" in str(folder):
        event_dirs = list(Path(folder).rglob("*/*settings.yaml"))
        event_list = []
        for event in event_dirs:
            db_prompt = event.parents[0] / "db_prompt.yaml"
            model = str(event).replace("settings", "model")
            time = str(event.parts[-1]).replace("_settings.yaml","")
            event = benedict(str(event))
            for key, value in benedict(str(db_prompt)).items():
                event[key] = value
            event["model"] = model
            event["time"] = time
            event_list.append(event)
            print(event)


    #%% Reset the database and add all of the events
    coll.delete_many({})
    for event in event_list:
        coll.insert_one(event)



def retrieve_filtered_list(coll, prompt = {}):
    coll = get_collection(coll)

    # Example for filtering a collection in the database and retrieving the data as a list
    filtered_list = list(coll.find({'microscope': 'isim', 'cell_line': 'cos7'}))
    my_list = [str(item) for item in filtered_list]

    print("\n".join(my_list))
    return my_list