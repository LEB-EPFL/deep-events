#%%
from deep_events.database.extract_yaml import MAIN_PATH
from deep_events.database import get_collection, get_cluster
from pathlib import Path
from pymongo import MongoClient
from benedict import benedict
import tifffile
import os
import shutil

def main(folder = MAIN_PATH): #pragma: no cover
    reconstruct_from_folder(folder, "mito_fluo")


def reconstruct_from_folder(folder: Path, collection: str):
    # Initialize database connection
    coll = get_collection(collection)
    print(folder)
    # Get the db.yaml files from the folders
    if "event" in str(folder.name):
        print(folder)
        event_list = list(Path(folder).rglob(r"*event_db.yaml"))
        event_dicts = [benedict(str(event)) for event in event_list]
        corrected_event_list = []
        print(event_list)
        for event_dict, path in zip(event_dicts, event_list):
            event_dict['event_path'] = str(Path(path).parents[0].resolve())
            event_dict.to_yaml(filepath=event_dict['event_path'] + "/event_db.yaml")
            corrected_event_list.append(event_dict)
        event_list = corrected_event_list
    elif "training_data" in str(folder.name):
        event_dirs = list(Path(folder).rglob("*/*settings.yaml"))
        print("in training data")
        event_list = []
        for my_event in event_dirs:
            db_prompt = my_event.parents[0] / "db_prompt.yaml"
            model = str(my_event).replace("settings.yaml", "model.h5")
            images_train = my_event.parents[0] / "train_images_00.tif"

            time = str(my_event.parts[-1]).replace("_settings.yaml","")
            event = benedict(str(my_event))
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


def clean_up_folder(folder):
    if "training_data" not in str(folder):
        print("Not yet implemented for events, only models and training data")
    event_dirs = list(Path(folder).rglob("*/*settings.yaml"))
    for my_event in event_dirs:
        model = str(my_event).replace("settings.yaml", "model.h5")
        if not os.path.exists(model):
            print(f"deleting {my_event.parents[0]}")
            os.remove(my_event)
            if len(list(Path(model).parents[0].glob("*model.h5"))) == 0:
                shutil.rmtree(Path(model).parents[0])


def retrieve_filtered_list(coll, prompt = {}):
    coll = get_collection(coll)

    # Example for filtering a collection in the database and retrieving the data as a list
    filtered_list = list(coll.find({'microscope': 'isim', 'cell_line': 'cos7'}))
    my_list = [str(item) for item in filtered_list]

    print("\n".join(my_list))
    return my_list

if __name__ == "__main__": #pragma: no cover
    # reconstruct_from_folder("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/event_data_pearls",
    #                          'pearl_events')    
    reconstruct_from_folder(Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/single_channel_fluo/event_data"),
                             'mito_fluo')
    # reconstruct_from_folder("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/event_data_ld",
    #                          'ld_events')
    # main("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/training_data")
    # retrieve_filtered_list("mito_events")
