This is the organisation at the moment.

```mermaid
graph TD;
    folder --> original_data;
    ome.tifs(ome.tifs: YYMMDD_cell_stain_contrast_number) --> folder(folder: YYMMDD_stain_cell_microscope_contrast);
    labels.csvs(labels.csvs: labels_number_contrast) --> folder;

    folder2(folder: YYMMDD_stain_cell_microscope_contrast_pos/neg) --> training_data;
    .tiff(image_YYMMDD_cell_contrast_number_event) --> folder2;
    .tiff2(image_YYMMDD_cell_contrast_number_sigmaX_eventgauss) --> folder2;

```

The goal is to make something that is as flat as possible for training_data and have an easy system
to look up the things we want to see. Here is an idea:

```mermaid
graph TD;


```