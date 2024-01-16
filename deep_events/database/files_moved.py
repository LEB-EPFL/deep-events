# If files are moved, data in the db files might have to be updated
from pathlib import Path
from benedict import benedict

folder = "//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/event_data"

db_files = list(Path(folder).rglob(r'event_db.yaml'))
print(len(db_files))

for file in db_files:
    db_dict = benedict(file)
    print(db_dict['original_path'])
    # Do modification depending on where things were moved here