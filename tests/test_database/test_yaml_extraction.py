
from database import extract_yaml
from database import ome_to_yaml
from database import folder_benedict

def test_defaults():
    path = "//lebnas1.epfl.ch/microsc125/deep_events/original_data/220927_mitotrackergreen_cos7_ZEISS_fl"
    extract_yaml.set_defaults(path)
    folder_dict = folder_benedict.get_dict(path)
    assert folder_dict['typically_use'] is True

def test_ome():
    path = "//lebnas1.epfl.ch/microsc125/deep_events/original_data/220927_mitotrackergreen_cos7_ZEISS_fl"
    ome_to_yaml.extract_ome(path)
    folder_dict = folder_benedict.get_dict(path)
    assert folder_dict['ome']['channel']['name'] == 'GFP'

def test_foldername():
    path = "//lebnas1.epfl.ch/microsc125/deep_events/original_data/220927_mitotrackergreen_cos7_ZEISS_fl"
    extract_yaml.extract_foldername(path)
    folder_dict = folder_benedict.get_dict(path)
    print(folder_dict)
    assert folder_dict['date'] == '220927'
    assert folder_dict['original_folder'] == "220927_mitotrackergreen_cos7_ZEISS_fl"