Module for deep learning ingestion of new data and training of U-Nets
![training_Artboard 1](https://github.com/LEB-EPFL/deep-events/assets/52414717/6dfd34a1-7700-4a9d-81b5-4ace9f975114)


## Installation
This repository explains how to prepare the acquired microscopy data to train a UNET model, and how to ultimately perform the training. Consider that the tensorflow backend is meant for the 2.10 version, as this is the last version you will be able to run on Windows natively (which you will probably need to run your microscope). If this is unacceptable to you, have a look at the pytorch backend.
The dataflow is described in the figure above and you can see expamples in scheduled_settings/prepare_data.

To start:
- Create and activate a python environment with the packages indicated in the file *requirements.txt*
- Install a MongoDB Database. To do so, follow the instruction at [https://www.mongodb.com/docs/manual/installation/](https://www.mongodb.com/docs/manual/installation/)

## **Necessary before starting training**

- A folder *my_parent_data_folder* containing:
    - **[optional] General Additional Metadata**
    To assign metadata to all original data contained in *my_parent_data_folder*, we can define it in a .yaml file called *db_manual.yaml*. E.g., we can set the cell line and the microscope used:
    
    ```yaml
    # in *db_manual.yaml*
    cell_line:
    - cos7
    microscope:
    - zeiss
    ```
    
    == NOTE ==
    The file *deep_events/database/keys.yaml* contains the keys that are recognised in the *db_manual.yaml* files. Add specific keys and values if necessary.
    
    - **Data Folders**
    Folder containing data, one folder per dataset. Example nomenclature: *my_data_01*. Each folder contains:
        - **Data**
        The .ome.tiff files containing the original multi-channel, multi-frame data. E.g., *my_data_01.ome.tifff.*
        - **[Optional] Additional Metadata**
        To set additional metadata not contained in the ome.tiff file for the data contained in the current folder, we can set it in a file called *db_manual.yaml*. Example: *my_data_01.ome.tifff* is cropped, but this information is not available in the .tiff metadata. In this case, we want *db_manual.yaml* to be:
        
        ```yaml
        # in *db_manual.yaml*
        ome:
          size_x: 458
          size_y: 620
          size_t: 100.0
        ```
        
        - **Annotations**
        A .csv file containing annotations, generated using e.g., Napari. For point-like annotations (e.g., for mitochondrial fission models), the structure of the .csv file is as follows:
            - Index: the index of the point-like annotation
            - axis-0, axis-1, axis-2, axis-3: the frame, ch, y, x of the line annotation
    
    | **index** | **axis-0** | **axis-1** | **axis-2** | **axis-3** |
    | --- | --- | --- | --- | --- |
    | **0** | 1.0 | 1.0 | 139.20894 | 384.0587 |
    | **0** | 1.0 | 1.0 | 153.96388 | 376.01056 |
    | **0** | 1.0 | 1.0 | 167.37746 | 370.6451 |
    

## **Procedure**

This whole procedure to prepare the data for the model training can be found in the script deep-events\deep_events\[main.py](http://main.py/). The following are the different tasks in this script.

1. **Set the desired parameters**

```python
FOLDERS = [Path("my_parent_data_folder")]
csv_file_pattern = 'csv_name'
img_types = [r'*.ome.tif*']
ground_truth_types = [csv_file_pattern + '_points']
```

1. **Generate the necessary metadata** 
Here, we read the .ome.tiff metadata, the optional general metadata contained in *my_parent_data_folder/db_manual.yaml*, and the optional specific metadata contained in any *my_data/db_manual.yaml*  to generate the *db.yaml* file - containing all necessary metadata for further steps. The *db.yaml* file should look something like the following: 

```yaml
augmented: false

# from the general db_manual.yaml
cell_line:
- cos7

date: '230424'

# from the general db_manual.yaml
microscope:
- zeiss

# from the folder-specific db_manual.yaml
ome:
  size_t: 100.0
  size_x: 2048
  size_y: 2048

original_folder: 230424_siCtrl_001
original_path: \\original\\path
scale_csv: true
type: original
typically_use: true
```

1. **Generate the ground truth images**
Here, we take the information from the .csv files and generate the ground truth images that will later be used to train the network. 

```python
for folder in FOLDERS:
    SIGMA_G = 5
    csv_to_gaussian(folder, SIGMA_G, csv_file_pattern)
```

1. **Generate the events folder, with data cropped in space and time**
The event folder will be structured so that it can be used for training

```python
# Generate the event folder to be used to populate the DataBase
for gt_type in ground_truth_types:
settings = {
'img_identifier': "",
'gt_identifier': "ground_truth_" + gt_type,
'db_name': "db.yaml",
'channel_contrast': "",
'label': "",
'add_post_frames': 0,
'auto_negatives': 0,
}

event_folder = f"event_data_{gt_type}"
event_folders = [event_folder]

gaussian_to_training_events(FOLDERS, event_folders, img_types, settings)
```

1. **Create the database needed for actually training**
Here, we update a MongoDB database with the events later used for training. Please refer to the related section in this readme file to check how to create it. The database can be later queried to filter training data as desired. The database is deleted (at least for the automatic annotations) every time we run *reconstruct_from_folder*. 

```python
reconstruct_from_folder(
"[e](https://sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/Smart_Pearling_GT_AK_WS_JCL/Ready_for_training/event_data_pearls_lines)vent_folder", # the event folder just created
'event_data_points' # collection name in the database
)
```

## Training

In order to train, follow these steps:

1. Create a settings .yaml file (e.g., *setting_training_01.yaml*) in the scheduled_settings folder. The file informs the training model where to look for the training data, and which backend to use (either tensorflow or pytorch). Example: 
    
    ```yaml
    # Main settings, if others needed, make new yaml file
    folder: event_folder_data # Where the data is stored
    collection: event_data_points  # What is it called on the MongoDB database?
    backend: torch
    
    # Each db_prompt will make a new folder and get data from the collection defined above
    db_prompt:
    max_n_timepoints: 1
    fps: 1
    train_val_split: 0.1
    settings:
    epochs: 150
    batch_size: 8
    loss: 'soft_focal'
    weight:
    alpha: 0.4
    gamma: 2
    model: 'unet'
    poisson: 0.2
    n_timepoints: 1
    log_dir: logs_ld
    ```
    
2. Set the directory of the yaml file just created in the PowerShell script *deep_events/deep-events/scheduled_train/scheduled_train.ps1*. 
    
    ```python
    # Set your yaml files here
    $BaseDir = "C:\Internal\deep_events\scheduled_settings"
    $YamlFiles = @( '*setting_training_01.yaml*')
    ```
    
3. Set the desired settings for the model, exposed by the uNET. To do so, set DEFAULT_SETTING in *deep-events/scheduled_train/scheduled_train.py.* 
    
    ```python
    DEFAULT_SETTINGS = {
    "nb_filters": 16,  # How many kernels are in the first convolution step. The more filters, the more features in the bottleneck
    "first_conv_size": 12,
    "nb_input_channels": 1,
    "subset_fraction": 1, # allows to train on less data (<1)
    "batch_size": 16, # how many training steps before back_propagation
    "epochs": 20,
    "n_augmentations": 10, # how many time augmemntation is performed
    'brightness_range': [0.6, 1],
    "poisson": 0, # how much noise is added in the augmentation step
    "loss": 'binary_crossentropy', # the loss function of the uNET
    "initial_learning_rate": 4e-4, # How fast the weights are changed. Low: longer training, more accurate. High: high risk to go to local minima
    'subset': 1,
    }
    ```
    
4. Run the PowerShell script:
    
    ```powershell
    >> .\deep_events\deep-events\scheduled_train\scheduled_train.ps1
    ```
    
    As an output, we have:
    
    - the yaml file, contains the settings of the training
    - training images
    - settings where all the model settings are
    - the model
    - the performance scores
    

NOTEs:

- Different training processes can be run on different GPUS. That is, in one GPU we can train one model (with potentially different settings across models). It is not possible to split the load for the training of one model to multiple GPUs.
- To check the GPUs load during training, use the following terminal command:
    
    ```powershell
    >> *nvidia-smi -l 1*
    ```
    
- To monitor the performances of the training with TensorBoard, run:
    
    ```powershell
    >> tensorboard --logdir=my_parent_data_folder\training_data\logs_folder\scalars --port=7779 --samples_per_plugin image
    s=100
    ```
    

This will output a web-link that opens Tensorboard. This can be done also after the training is completed (very useful to compare performances of different models)