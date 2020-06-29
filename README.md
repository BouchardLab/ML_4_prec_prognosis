# activ

## Clustering

### Input

Before you can run the clustering pipeline, you will need to generate an input file. Here is an example code snippet that demonstrates
how to do that:

```Python
import numpy as np
from activ.clustering import write_clustering_input

path = 'path_to_save_input.h5'
X = np.random.random(100).reshape(10,10)
Y = np.random.random(100).reshape(10,10)

write_clustering_input(path, X, Y)
```

The following command calls `activ.clustering`, `activ.readfile`, `activ.utils`. This script is used 
to run a clustering pipeline that clusters response data (i.e. "outcomes") to create labels for each sample. The quality of the
labels are assessed by predicting those labels with the predictor data (i.e. "biomarkers").

```Bash
$ python bin/run_subsample_umap_clustering.py
```

It can be run in parallel with MPI.

```Bash
$ mpirun -n <N_RANKS> python bin/run_subsample_umap_clustering.py
```

The results of this workflow are visualized with the notebook `notebooks/plot_clustering_results.ipynb`. This
notebook pulls functions from `activ.clustering.summarize`

## Latent features (i.e. NMF)

This package contains a subpackage for visualizing NMF result, `activ.nmf.viz`. Examples uses can be found in `notebooks/NMF_viz.ipynb`.

## CCA - Canonical correlation analysis

Within this package, there are a few different implementations of sparse canonical correlation analysis. The most mature
is  _truly-alternating least squares CCA_ [1]. In [1] least squares steps are regularized using an _L2_-norm.
The implementation here regularizes using an _L1_-norm to identify sparse weights. This is implemented in the subpackage
`activ.cca.alscca` in the class `TALSCCA`. Example use can be found in `notebooks/TALS_CCA_CV.ipynb`

## CT Measures

CT analysis is done using executable submodules, rather than executable Python scripts.

1. Reformat data using `python -m activ.ct.convert`
2. Summarize data using `python -m activ.ct.summarize`

Test data for running step 1 can be found in `data/ct/115Label_fake`.

# References
1. Zhiqiang Xu and Ping Lie, [Towards Practical Alternating Least-Squares for CCA](https://papers.nips.cc/paper/9616-towards-practical-alternating-least-squares-for-cca)
