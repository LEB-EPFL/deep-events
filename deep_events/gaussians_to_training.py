from pathlib import Path



folder = "Z:/_Lab members/Juan/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/"
db_files = list(Path(folder).rglob(r'db.yaml'))

for db_file in db_files:
    tif_file = sorted(Path(os.path.dirname(db_file)).glob(r'*.ome.tif'), key=os.path.getmtime)[-1]
    gaussians_file = os.path.join(os.path.dirname(db_file), 'ground_truth.tiff')
