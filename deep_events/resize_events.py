"""
In case you have extracted events that are too short, this script will extend them by the needed amount of frames
"""


#%%
from pathlib import Path
from benedict import benedict
import tifffile
import napari
import matplotlib.pyplot as plt
import numpy as np
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
#%%
dbs = list(Path(__file__).parents[0].rglob('event_db.yaml'))

#%%
# dbs = ["//sb-nas1.rcp.epfl.ch\LEB\Scientific_projects\deep_events_WS"
#        "\data\original_data\event_data_ld"
#        "\ev_cos7_zeiss_['brightfield']_679b5b0c1c96a05d2dbc7629\event_db.yaml"]
final_shapes = []
for idx, db in enumerate(dbs[20:]):
    print('INDEX', idx)
    event_dict = benedict(db)
    # if event_dict.get('pre_extended', False):
    #     print(Path(db).parents[0].name, 'was already extended')
    #     continue
    if 'iSIM' in event_dict['original_path'] or 'U2OS' in event_dict['original_path']:
        print('skip', Path(db).parents[0].name, 'iSIM or u2os')
        continue
    orig_path = event_dict['original_path']
    orig_path = orig_path.replace('\\\\lebnas1.epfl.ch\microsc125\deep_events',
                                  '//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS')
    event_dict['original_path'] = orig_path
    orig_event_path = event_dict['event_path']
    orig_event_path = orig_event_path.replace('\\\\lebnas1.epfl.ch\microsc125\deep_events',
                                  '//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS')    
    event_dict['event_path'] = orig_event_path
    print(event_dict['event_path'])
    
    orig_data = tifffile.imread(Path(db).parents[0]/'images.tif')
    orig_gt_data = tifffile.imread(Path(db).parents[0]/'ground_truth.tif')
    if orig_gt_data.max() == 0:
        event_dict['event_content'] = 'none'
    else:
        event_dict['event_content'] = 'contact'    

    box = event_dict['crop_box']
    event_pos = [ box[0] + (box[2]-box[0])/2, box[1] + (box[3]-box[1])/2, 0]
    event = EDAEvent(ID=event_dict['_id'], name=None, center_position=event_pos,
                     first_frame=int(event_dict['frames'][0]+1), last_frame=int(event_dict['frames'][1]-1))

    #prepone event start
    pre_frames = max(0, 18 - (event.last_frame - event.first_frame))
    print('Original GT shape', orig_gt_data.shape)
    print('Original shape', orig_data.shape)
    # if pre_frames == 0:
    #     print('Event long enough, skip')
    #     continue
    old_first = event.first_frame
    print('FIRST FRAME', event.first_frame)
    event.first_frame = event.first_frame - pre_frames
    event.first_frame = 0 if event.first_frame < 0 else event.first_frame
    extension_frames = old_first - event.first_frame
    extension_frames = max(extension_frames, 0)
    if 'zarr' in event_dict['original_folder']:
        print('ZARR')
        print(event_dict['original_path'])
        try:
            reader = Reader(parse_url(event_dict['original_path']))
        except AttributeError:
            event_dict['original_path'] = event_dict['original_path'].replace('test_plin5', 'Willi')
            reader = Reader(parse_url(event_dict['original_path']))
        images = list(reader())[0].data[0]
        del reader
        channel = 0 # always??
        images_crop, box = crop_images_array(event, images, channel)
        extension = np.zeros((int(extension_frames), *orig_gt_data.shape[-2:]))
        gt_crop = np.vstack((extension, orig_gt_data))
        print('extended by', extension_frames)
    else:
        if 'model_pred' in event_dict['original_file']:
            event_dict['model_pred'] = event_dict['original_file']
            ome_tifs = Path(event_dict['original_path']).glob('*ome.tif')
            event_dict['original_file'] = list(ome_tifs)[0]
        orig_file = Path(orig_path) / event_dict['original_file']
        orig_gt = Path(orig_path) / 'ground_truth_ld_mito.tiff'
        if event_dict.get('extraction_type') == 'manual':
            print('extended manual by', extension_frames)
            extension = np.zeros((int(extension_frames), *orig_gt_data.shape[-2:]))
            gt_crop = np.vstack((extension, orig_gt_data))
        else:
            with tifffile.TiffFile(orig_gt) as gaussians:
                gt_crop, box = crop_images_tif(event, gaussians)
                print('extended automatic by', extension_frames)

        with tifffile.TiffFile(orig_file) as images:
            images_crop, box = crop_images_tif(event, images)

    if gt_crop.shape[0] > images_crop.shape[0]:
        print('GT too long')
        gt_crop = gt_crop[(gt_crop.shape[0]-images_crop.shape[0]):]

    print(gt_crop.shape)
    print(images_crop.shape)
    final_shapes.append([gt_crop.shape, images_crop.shape])
    tifffile.imwrite(Path(event_dict['event_path']) / 'ground_truth.tif', gt_crop.astype(np.float16), photometric='minisblack')
    tifffile.imwrite(Path(event_dict['event_path']) / 'images.tif', images_crop.astype(np.float16), photometric='minisblack')

    event_dict['pre_extension_frames'] = event_dict['frames']
    event_dict['frames'][0] = event.first_frame
    event_dict['pre_extended'] = True
    event_dict['pre_extension_frames'] = min(pre_frames, old_first)
    event_dict.to_yaml(filepath=Path(event_dict['event_path'])/'event_db.yaml')

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(orig_data[0])
    axs[2].imshow(orig_gt_data[0])
    axs[1].imshow(images_crop[extension_frames])
    axs[3].imshow(gt_crop[extension_frames])
    plt.tight_layout()
    plt.show()
print('\n'.join([str(x) for x in final_shapes]))

#%%
# from event_extraction.py in the deep_events repository
def crop_images_tif(event, imgs, channel=0, size=256):
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

#this changed from the csv extraction, because I think the napari-plugin defines the box coordinates differently
def crop_images_array(event, imgs, channel=0, size=256):
    dataar=np.zeros((event.last_frame-event.first_frame, size, size))
    if len(imgs.shape) == 4:
        n_channels = imgs.shape[1]
    else:
        n_channels = 1

    for index, frame in enumerate(range(event.first_frame + 1, event.last_frame + 1)):
        box = box_from_pos(event.c_p['x'], event.c_p['y'], size)
        box = box_edge_check(box, imgs.shape[-2:])
        try:
            if len(imgs.shape) == 3:
                dataar[index] = imgs[frame, box[1]:box[3], box[0]:box[2]]
            else:
                dataar[index] = imgs[frame, channel, box[1]:box[3], box[0]:box[2]]    
        except IndexError:
            print("frame to extract", frame)
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

class EDAEvent():
    """ Sctucture that represents an interesting event in a 3D video"""
    def __init__(self,name, center_position, first_frame, last_frame, ID: int = 0):
        self._ID = ID
        self.name = name
        self.c_p = {'x': center_position[0], 'y': center_position[1], 'z': center_position[2]}
        self.first_frame = first_frame-1
        self.last_frame = last_frame
# %%
