"""Functions to detect and separate events from ground truth data.

Some of the functions are from napari-eda-highlight-reel"""

import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm


def basic_scan(data, size=256, threshold=0.7):
    open_events = []
    framenumber = len(data)
    ev_n = 0
    for i in tqdm(range(framenumber)):
        actualist = find_cool_thing_in_frame(data[i],
                                             threshold=threshold, nbh_size = size)
        while actualist:
            new_event = True
            for ev in open_events:
                if all([abs(ev.c_p['x'] - actualist[0]['x'])<size,
                        abs(ev.c_p['y'] - actualist[0]['y'])<size,
                        abs(ev.c_p['z'] - actualist[0]['z'])<size,
                        i - ev.last_frame < 3]):
                    ev.last_frame = i
                    new_event = False
            if new_event:
                open_events.append(EDA_Event('Event ' + str(len(open_events)),
                                             [actualist[0]['x'],
                                              actualist[0]['y'],
                                              actualist[0]['z']],i,ev_n))
                ev_n += 1
            actualist.pop(0)

    for ev in open_events:
        ev.last_frame = ev.last_frame+1
    return open_events


def find_cool_thing_in_frame(frame, threshold: float, nbh_size: int) -> list:
    """Function that takes a 3D frame and takes a list of the positions of the local maxima that are higher than the threshold

    Parameters
    ----------

    frame : numpy.ndarray
        The 3D image to be analyzed
    threshold : float
        Minimal value of the event score a maxima must have to be considered
    nbh_size : int
        Size of the neighbourhood around a local mximum wich it is considered that an other local maximum would be due to noise

    Returns
    -------

    list of dictionaries having at the entries 'x', 'y' and 'z' the x, y nd z coordinate of every event center
    """
    if frame.max() == 0:
        return []
    if len(frame.shape) == 2:                         #To treat 2D images as 3D
        frame = np.expand_dims(frame, axis = 0)
    data_max = ndi.maximum_filter(frame, nbh_size, mode = 'constant', cval = 0)
    maxima = (frame == data_max)
    upper = (frame > threshold)
    maxima[upper == 0] = 0
    labeled, num_objects = ndi.label(maxima)
    slices = ndi.find_objects(labeled)
    Events_centers = []

    for dz,dy,dx in slices:
        evvy = {'x': 0, 'y': 0, 'z': 0}
        evvy['x'] = (dx.start + dx.stop - 1)/2
        evvy['y'] = (dy.start + dy.stop - 1)/2
        evvy['z'] = (dz.start + dz.stop - 1)/2
        Events_centers.append(evvy)
    return Events_centers


class EDA_Event():
    """ Sctucture that represents an interesting event in a 3D video"""
    def __init__(self,name, center_position, first_frame, ID: int = 0):
        self._ID = ID
        self.name = name
        self.c_p = {'x': center_position[0], 'y': center_position[1], 'z': center_position[2]}
        self.first_frame = first_frame-1
        self.last_frame = first_frame


def crop_images(event, imgs, channel=0, size=256):
    dataar=np.zeros((event.last_frame-event.first_frame, size, size))
    if len(imgs.series[0].shape) == 4:
        n_channels = imgs.series[0].shape[1]
    else:
        n_channels = 1

    for index, frame in enumerate(range(event.first_frame + 1, event.last_frame + 1)):
        box = box_from_pos(event.c_p['x'], event.c_p['y'], size)
        box = box_edge_check(box, imgs.pages[0].shape[-2:])
        frame = frame*n_channels+channel
        try:
            dataar[index] = imgs.series[0].pages[frame].asarray()[box[1]:box[3], box[0]:box[2]]
        except IndexError:
            print("frame to extract", frame)
            print("frames in series", len(imgs.series[0].pages))
            print("not continuing")
            print("array before crop", dataar.shape)
            dataar = dataar[:index]
            print("array after crop", dataar.shape)
            break
    return dataar, box


def box_from_pos(x, y, size):
    box = [0, 0, 0, 0]
    box[0] = round(x - size/2)
    box[1] = round(y - size/2)
    box[2] = round(x + size/2)
    box[3] = round(y + size/2)
    return box


def box_edge_check(box, img_size):
    if box[0] < 0:
        box[2] = box[2] - box[0]
        box[0]=0
    if box[1] < 0:
        box[3] = box[3] - box[1]
        box[1] = 0
    if box[2] > img_size[0]:                           #safety conditions in case pics are at the lower edges
        box[0] = img_size[0] - (box[2] - box[0])
        box[2] = img_size[0]
    if box[3] > img_size[1]:
        box[1] = img_size[1] - (box[3] - box[1])
        box[3] = img_size[1]
    return box