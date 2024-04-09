#%%
from deep_events.database.extract_yaml import MAIN_PATH
from deep_events.database import get_collection, get_cluster
from pathlib import Path
from pymongo import MongoClient
from benedict import benedict
import tifffile

def main(folder = MAIN_PATH): #pragma: no cover
    reconstruct_from_folder(folder, "mito_events")


def reconstruct_from_folder(folder: Path, collection: str):
    # Initialize database connection
    coll = get_collection(collection)
    print(folder)
    # Get the db.yaml files from the folders
    if "event_data" in str(folder):
        event_list = list(Path(folder).rglob("*/event_db.yaml"))
        event_dicts = [benedict(str(event)) for event in event_list]
        corrected_event_list = []
        for event_dict, path in zip(event_dicts, event_list):
            event_dict['event_path'] = str(Path(path).parents[0].resolve())
            event_dict.to_yaml(filepath=event_dict['event_path'] + "/event_db.yaml")
            corrected_event_list.append(event_dict)
        event_list = corrected_event_list
    elif "training_data" in str(folder):
        event_dirs = list(Path(folder).rglob("*/*settings.yaml"))
        event_list = []
        for event in event_dirs:
            db_prompt = event.parents[0] / "db_prompt.yaml"
            model = str(event).replace("settings", "model")
            images_train = event.parents[0] / "train_images_00.tif"

            time = str(event.parts[-1]).replace("_settings.yaml","")
            event = benedict(str(event))
            db_prompt_dict = benedict(str(db_prompt))
            for key, value in db_prompt_dict.items():
                event[key] = value

            #Storing amount of training data
            with tifffile.TiffFile(images_train) as tif:
                event['frames'] = tif.series[0].shape[0]

            event["model"] = model
            event["time"] = time
            event_list.append(dict(event))

    # benedict({"cluster": get_cluster(), "collection": collection}).to_yaml(filepath=folder/"collection.yaml")
    # Reset the database and add all of the events
    coll.delete_many({})
    print(len(event_list))
    for event in event_list:
        coll.insert_one(event)




def retrieve_filtered_list(coll, prompt = {}):
    coll = get_collection(coll)

    # Example for filtering a collection in the database and retrieving the data as a list
    filtered_list = list(coll.find({'microscope': 'isim', 'cell_line': 'cos7'}))
    my_list = [str(item) for item in filtered_list]

    print("\n".join(my_list))
    return my_list

if __name__ == "__main__": #pragma: no cover
    main("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/event_data")
    # retrieve_filtered_list("mito_events")
