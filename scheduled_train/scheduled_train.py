from pathlib import Path
from benedict import benedict
import traceback
import datetime

# from deep_events.gaussians_to_training import extract_events
from deep_events.prepare_training import prepare_for_prompt
from deep_events.train import distributed_train
from deep_events.database import reconstruct_from_folder
from deep_events.database.prepare_yaml import prepare_all_folder


DEFAULT_SETTINGS = {"nb_filters": 16,
            "first_conv_size": 12,
            "nb_input_channels": 1,
            "batch_size": 16,
            "epochs": 20,
            "n_augmentations": 30,
            'brightness_range': [0.6, 1],
            "poisson": 0,
            "loss": 'binary_crossentropy'}

SETTINGS_FOLDER = Path('C:/Internal/deep_events/scheduled_settings')

def main(settings_file=None):

    if settings_file is None:
        settings_yamls = SETTINGS_FOLDER.glob(r"*.yaml")
    else:
        settings_yamls = [settings_file]
    
    yamls = list(settings_yamls)
    # yamls.reverse()
    print("\n".join([str(x) for x in yamls]))

    for settings_yaml in yamls:
        try:
            settings_dict = benedict(settings_yaml)
            print(settings_dict)
            reconstruct_from_folder(Path(settings_dict["folder"]), settings_dict["collection"])
            folders, settings = gather_settings(settings_dict)
            settings = add_missing_settings(settings)
        except Exception as e:
            handle_error(e, str(settings_yaml) +  "\nMaybe data has moved?")
            continue

        log_file = open(Path(__file__).parents[0] / "scheduled_train.log", "a")
        for folder in folders:
           log_file.write(str(folder) + "\n")
        log_file.close()

        # Start distributed train
        gpus = ['GPU:3/', 'GPU:5/', 'GPU:2/', 'GPU:4/', 'GPU:1/']
        if len(gpus) < len(folders):
            log_file = open(Path(__file__).parents[0] / "scheduled_train.log", "a")
            log_file.write("WARNING, not enough GPUS defined to train all models" + "\n")
            log_file.close()
        print("\n".join([str(x) for x in folders]))
        try:
            distributed_train(folders, gpus, settings)
        except Exception as e:
            handle_error(e, "\n".join([str(x) for x in folders]))
        
        log_file = open(Path(__file__).parents[0] / "scheduled_train.log", "a")
        log_file.write("DONE " + str(datetime.datetime.now()) + "\n")
        log_file.close()


def add_missing_settings(settings_list: list) -> list:
    """Complete incomplete settings dicts using the default settings provided above."""
    for idx, settings in enumerate(settings_list):
        for key, value in DEFAULT_SETTINGS.items():
            if not key in settings:
                settings[key] = value
        settings_list[idx] = settings
    return settings_list


def gather_settings(settings_dict: dict) -> tuple:
    """ Get the settings from yaml files and prepare the data in folders."""
    folders = []
    settings = []
    for key in settings_dict.keys():
        if "db_prompt" in key:
            folder = prepare_for_prompt(Path(settings_dict["folder"]), settings_dict[key],
                                        settings_dict["collection"])
            continue
        if "settings" in key:
            setting = settings_dict[key] if settings_dict[key] != "default" else {}
            folders.append(folder)
            settings.append(setting)
    return folders, settings


def handle_error(e: Exception, message):
    log_file = open(Path(__file__).parents[0] / "scheduled_train_log.txt", "a")
    log_file.write("ERROR \n")
    traceback_str = ''.join(traceback.format_tb(e.__traceback__))
    log_file.write(traceback_str)
    print(e)
    print(traceback_str)
    log_file.write(message + "\n")
    raise(e)


if __name__ == '__main__':
    main()