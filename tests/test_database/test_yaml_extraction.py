from pathlib import Path


from deep_events.database import extract_yaml
from deep_events.database import ome_to_yaml
from deep_events.database import folder_benedict

def test_defaults():
    path = "//lebnas1.epfl.ch/microsc125/deep_events/original_data/220927_mitotrackergreen_cos7_ZEISS_fl"
    extract_yaml.set_defaults(path)
    folder_dict = folder_benedict.get_dict(path)
    assert folder_dict['typically_use'] is True

def test_ome():
    path = Path("//lebnas1.epfl.ch/microsc125/deep_events/original_data/220915_mtstaygold_cos7_ZEISS_fl")
    tif_file = list(path.glob("*.ome.tif"))[0]
    ome_to_yaml.extract_ome(tif_file)
    folder_dict = folder_benedict.get_dict(path)
    assert folder_dict['ome']['channel']['name'] == 'GFP'

def test_foldername():
    path = "//lebnas1.epfl.ch/microsc125/deep_events/original_data/220927_mitotrackergreen_cos7_ZEISS_fl"
    folder_dict = folder_benedict.get_dict(path)
    extract_yaml.extract_foldername(folder_dict, path)
    print(folder_dict)
    assert folder_dict['date'] == '220927'
    assert folder_dict['original_folder'] == "220927_mitotrackergreen_cos7_ZEISS_fl"