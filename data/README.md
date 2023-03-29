# TRACK-TBI Pilot Data

This directory contains data from the TRACK-TBI pilot study. 

## Data used for _Data-driven distillation and precision prognosis in traumatic brain injury with interpretable machine learning_

Data used for the manuscript _Data-driven distillation and precision prognosis in traumatic brain injury with interpretable machine learning_ can by found
in the [`data`](https://github.com/BouchardLab/ML_4_prec_prognosis/blob/master/data/data) subdirectory of this directory at 
[`data/preprocessed_data_matrix.csv`](https://github.com/BouchardLab/ML_4_prec_prognosis/blob/master/data/data/preprocessed_data_matrix.csv). 
The file [`data/mask_feature_outcome.txt`](https://github.com/BouchardLab/ML_4_prec_prognosis/blob/master/data/data/mask_feature_outcome.txt) contains a binary mask indicating
which columns in the comma-separated values file were used as _clinical features_ (i.e. value `0`) and _outcome features_ (i.e. value `1`).

The data was filtered to remove incomplete rows and columns before continuing analysis. The script for this filtering can be found at [`subset_selection.R`](https://github.com/BouchardLab/ML_4_prec_prognosis/blob/master/data/subset_selection.R).
This script is written in the [R programming language](https://www.r-project.org/), and therefore must be run with R software environment. This script must be run within this directory. 
Below is an example of how to run this script on a Unix-like system with the R software environment installed.

```bash
$ cd $GIT_REPO_DIR/data
$ R < subset_selection.R
```

The outputs of this script were reformatted to create an HDF5 file to simplify reading data into analysis code. This reformated file can be found at 
[activ/data.h5](https://github.com/BouchardLab/ML_4_prec_prognosis/blob/master/activ/data.h5) and read with any HDF5 library, such as [h5py](https://www.h5py.org/).


## Other files and subdirectories in this directory
- `TRACKTBI_PILOT_FAKE.csv` - a comma-separated values file containing fake data in the same form as the original data. This was used for testing purposes
