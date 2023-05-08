from pathlib import Path
import os

from deep_events.database import extract_yaml
from deep_events.database import ome_to_yaml
from deep_events.database import folder_benedict



FILEPATH = Path(__file__).parents[1]


def test_recursive_folder():
    file = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl/GFP_1/220923_cos7_mitotrackergreen_GFP_1.ome.tif"
    os.remove(Path(file).parents[0] / "db.yaml")
    extract_yaml.recursive_folder(file)
    folder_dict = folder_benedict.get_dict(Path(file).parents[0])
    assert folder_dict['date'] == "220923"


def test_defaults():
    path = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl/GFP_1"
    os.remove(Path(path) / "db.yaml")
    extract_yaml.recursive_folder(Path(path)/'220923_cos7_mitotrackergreen_GFP_1.ome.tif')
    folder_dict = folder_benedict.get_dict(path)
    folder_dict = extract_yaml.set_defaults(folder_dict)
    assert folder_dict['typically_use'] is True


def test_ome():
    path = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl/GFP_1"
    os.remove(Path(path) / "db.yaml")
    extract_yaml.recursive_folder(Path(path)/'220923_cos7_mitotrackergreen_GFP_1.ome.tif')
    tif_file = list(path.glob("*.ome.tif"))[0]
    ome_to_yaml.extract_ome(tif_file)
    folder_dict = folder_benedict.get_dict(path)
    assert folder_dict['ome']['channel']['name'] == 'GFP'

def test_foldername():
    path = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl/GFP_1"
    os.remove(Path(path) / "db.yaml")
    extract_yaml.recursive_folder(Path(path)/'220923_cos7_mitotrackergreen_GFP_1.ome.tif')
    folder_dict = folder_benedict.get_dict(path)
    extract_yaml.extract_foldername(folder_dict, path)
    print(folder_dict)
    assert folder_dict['date'] == '220923'
    assert folder_dict['original_folder'] == "GFP_1"
    assert str(folder_dict['original_path'])[-20:] == '_cos7_ZEISS_fl\\GFP_1'


def test_check_minimal():
    file = FILEPATH / "data/220927_mitotrackergreen_cos7_ZEISS_fl/GFP_1/220923_cos7_mitotrackergreen_GFP_1.ome.tif"
    os.remove(Path(file).parents[0] / "db.yaml")
    extract_yaml.recursive_folder(file)
    extract_yaml.recursive_folder(file)
    folder_dict = folder_benedict.get_dict(Path(file).parents[0])
    folder_dict = extract_yaml.set_defaults(folder_dict)
    assert extract_yaml.check_minimal(Path(file).parents[0])