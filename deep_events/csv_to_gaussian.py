"""Translate csv files to gaussians"""

from pathlib import Path
import os

folder = "Z:/_Lab members/Juan/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/"




db_files = list(Path(folder).rglob(r'db.yaml'))

# From all the folders with a db.yaml file get the csv file
csv_files = []
for db_file in db_files:
    csv = list(Path(os.path.dirname(db_file)).glob(r'*.csv'))
    if csv:
        csv_files.append(csv[0])

csv_files = [str(x) for x in csv_files]

print("\n".join(csv_files))

for csv in csv_files:

    img_file = sorted(Path(os.path.dirname(csv)).glob(r'*.ome.tif'), key=os.path.getmtime)[-1]
    print(img_file)



    print('scaling by',scale_value)
    print(csv['axis-1'][1])
    csv['axis-1']=csv['axis-1']*(1/scale_value)
    print(csv['axis-1'][1])
    csv['axis-2']=csv['axis-2']*(1/scale_value)

    for i in range(2,7,1):
        sigma=(i,i)
        s=sigma[0]
        in_name=f'{input_name[x]}_sigma{s}.tiff'
        framenum=im.n_frames
        poi(csv,in_name,sigma,size,framenum)