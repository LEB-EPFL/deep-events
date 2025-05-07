Module for deep learning ingestion of new data and training of U-Nets
![training_Artboard 1](https://github.com/LEB-EPFL/deep-events/assets/52414717/6dfd34a1-7700-4a9d-81b5-4ace9f975114)


## Installation
You can install this project as a software package. It needs a running MongoDB server that has to be defined in database/settings.yaml. It will be used to store/retrieve metadata and filter the relevant data. Consider that the tensorflow backend is meant for the 2.10 version, as this is the last version you will be able to run on Windows natively (which you will probably need to run your microscope). If this is unacceptable to you, have a look at the pytorch backend.
The dataflow is described in the figure above and you can see expamples in scheduled_settings/prepare_data.