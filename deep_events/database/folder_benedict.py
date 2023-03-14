"""Use the benedict package to locate yaml files that are easy to translate to dictionaries in
folders. Saving is handled by info inside the dicts and entries can be set recursively"""


import os
from benedict import benedict

def handle_folder_dict(func):
    def wrapper(folder_dict, *args, **kwargs):

        if not isinstance(folder_dict, dict):
            folder_dict = get_dict(folder_dict)
        folder_dict = func(folder_dict, *args, **kwargs)
        save_dict(folder_dict)
        return folder_dict
    return wrapper


def get_dict(folder: str):
    """Get the dict fo a folder from the db.yaml file"""
    try:
        folder_dict = benedict(os.path.join(folder, "db.yaml"))
    except ValueError:
        print(f"""Dict not found at {folder}
                  Constructing new""")
        folder_dict = benedict(benedict().
                               to_yaml(filepath=os.path.join(folder, "db.yaml")))
    folder_dict['original_path'] = folder
    folder_dict['original_folder'] = os.path.basename(folder)
    return folder_dict


def save_dict(folder_dict: dict):
    """Save the dict to the original position as stated in itself."""
    if folder_dict['type'] == "original":
        folder_dict.to_yaml(filepath=os.path.join(folder_dict['original_path'],
                                              "db.yaml"))
    elif folder_dict['type'] == "event":
        folder_dict.to_yaml(filepath=os.path.join(folder_dict['event_path'],
                                              "db.yaml"))

def set_dict_entry(my_dict, key, value):
    """Recursively set dict entries for nested dicts"""
    if not isinstance(value, dict):
        my_dict[key] = value
    else:
        for sub_key, sub_value in value.items():
            my_dict = set_dict_entry(my_dict[key], sub_key, sub_value)
    return my_dict