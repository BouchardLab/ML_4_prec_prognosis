FROM continuumio/miniconda3
MAINTAINER  aessiari@lbl.gov

ADD . /workdir
WORKDIR /workdir

RUN apt update && apt-get -y install build-essential
RUN conda install -y -c clawpack -c conda-forge hdf5-parallel
RUN conda install -y -c conda-forge mpi4py
RUN conda install -y -c conda-forge gxx_linux-64
RUN conda install -y -c conda-forge pkg-config
ENV CC=mpicc
ENV HDF5_MPI="ON"
RUN python3 -m pip install --editable . --no-binary=h5py
RUN pip install -r jenkins/requirements-dev.txt

ENTRYPOINT [ "bash" ]
