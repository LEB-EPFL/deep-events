"""Script to translate annotation as lines for pearling to images for training."""


#%% Imports
import numpy as np
import csv
from pathlib import Path
import tifffile
from shapely.geometry.linestring import LineString
from typing import List, Tuple
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from benedict import benedict
from deep_events.database.convenience import glob_zarr

#%% Get tif file
FOLDER = Path('//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data/')

#%% Get data from csv and construct Linestrings
def get_lines_from_csv(csv_file: Path) -> List[List[float]]:
    """Get information from csv file and put into an array.
    Resulting lines columns: index, vertex, frame, z, x, y
    """
    with open(csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip header line
        lines = []
        for row in csv_reader:
            # select relevant columns and reduce frame by 1 (java -> python)
            line = [row[0]] + [row[2]] + [row[3]] + [row[-1]] + [row[-2]]
            lines.append(line)
    return lines

def translate_line_list(lines_info: list) -> List[Tuple[LineString, int]]:
    """Get lines array and put into lines"""
    lines = []
    line = [lines_info[0][-2:]]
    for idx, (row, previous) in enumerate(zip(lines_info[1:], lines_info)):
        if previous[0] == row[0] and idx < len(lines_info) - 2:
            line.append(row[-2:])
        elif idx == len(lines_info) - 2:
            line.append(row[-2:])
            lines.append((LineString(line), row[2]))
        else:
            lines.append((LineString(line), previous[2]))
            line = [row[-2:]]
    return lines

#%% Plot line functions
def plot_line(line: LineString, frame: int, tif: np.ndarray):
    """Plot line in frame of tif"""
    plt.plot(line.xy[1], line.xy[0], 'r')

def draw_line_to_frame(line: LineString, frame: int, tif: np.ndarray, width = 5) -> np.ndarray:
    """Draw line in frame of tif"""
    coordinates = np.array(line.coords, dtype=np.int32)
    coordinates = coordinates.reshape((-1, 1, 2))
    tif[int(float(frame))] = cv2.polylines(tif[int(float(frame))], [coordinates],
                                           False, 1, thickness=width)
    return tif

def get_shape_from_tif(tif_file: List[Path]) -> List[int]:
    """Get shape of tif file"""
    with tifffile.TiffFile(tif_file) as tif:
        try:
            n_frames = tif.imagej_metadata['frames']
        except TypeError:
            n_frames = len(tif.pages)
        shape = tif.pages[0].shape
    return [n_frames] + list(shape)


#%% Main functions
def get_lines(csv_file: Path) -> List[Tuple[LineString, int]]:
    lines = get_lines_from_csv(csv_file)
    lines = translate_line_list(lines)
    return lines

def draw_lines(lines: List[Tuple[LineString, int]], shape: tuple, width: int) -> np.ndarray:
    line_images = np.zeros(shape).astype(np.uint8)
    for line in lines:
        line_images = draw_line_to_frame(line[0], line[1], line_images, width)
    return line_images


# def create_ground_thruth(folder, filename):
#     WIDTH = 7
#     csv_files = Path(folder).rglob(filename)
#     for csv_file in tqdm(csv_files):
#         print(csv_file.parts[-3:-1])
#         # get newest tif file
#         tif_file = max(csv_file.parent.glob('*.ome.tif*'), key=lambda x: x.stat().st_mtime)
#         lines = get_lines(csv_file)
#         shape = get_shape_from_tif(tif_file)
#         print(shape)
#         line_images = draw_lines(lines, shape, WIDTH)
#         folder_dict = benedict(csv_file.parent / "db.yaml")
#         folder_dict['line_width'] = WIDTH
#         folder_dict.to_yaml(filepath=csv_file.parent / "db.yaml")
#         tifffile.imwrite(csv_file.parent / "ground_truth.tif", line_images, dtype=np.uint8)


# %%
def csv_to_lines(FOLDER, WIDTH, csv_pattern):
    csv_files = glob_zarr(Path(FOLDER), csv_pattern + "*.csv")
    for csv_file in tqdm(csv_files):
        # get newest tif file
        tif_file = max(csv_file.parent.glob('*.ome.tif'), key=lambda x: x.stat().st_mtime)
        lines = get_lines(csv_file)
        shape = get_shape_from_tif(tif_file)
        line_images = draw_lines(lines, shape, WIDTH)
        folder_dict = benedict(csv_file.parent / "db.yaml")
        folder_dict['line_width'] = WIDTH
        folder_dict.to_yaml(filepath=csv_file.parent / f"db_{csv_pattern}.yaml")
        tifffile.imwrite(csv_file.parent / f"ground_truth_{csv_pattern}.tif", line_images, dtype=np.uint8)

# %%
        
if __name__ == "__main__":
    WIDTH = 7
    csv_pattern = 'pearls'
    csv_to_lines(FOLDER, WIDTH, csv_pattern)
