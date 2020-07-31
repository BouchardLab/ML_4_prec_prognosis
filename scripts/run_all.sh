#!/bin/bash

function log ()
{
    local dir=$1
    echo "`date` - $1"
}

ENV=${1:?"Missing conda environment"};
INPUT=${2:?"Missing input"};
OUTDIR=${3:?"Missing output directory"};
LOG=$OUTDIR/$LOG

source activate $ENV
mkdir -p $OUTDIR
log "Copying $INPUT to $OUTDIR"
cp $INPUT $OUTDIR
INPUT=$OUTDIR/`basename $INPUT`
log "Using $INPUT"

log "Running UoI-NMF"
run_uoinmf.py -b 20 -s 1001 -O 6 -B 8 $INPUT

log "Plotting NMF results"
mkdir -p $OUTDIR/nmf
plot_uoinmf_results.py $INPUT -o $OUTDIR/nmf

log "Running ALS CCA"
run_cca.py $INPUT

log "Plotting CCA results with NMF labelling"
mkdir -p $OUTDIR/cca/nmf_labels
plot_cca_results.py $INPUT -o $OUTDIR/cca/nmf_labels

log "Plotting CCA results with clinical groupings"
mkdir -p $OUTDIR/cca/clinical_labels
plot_cca_results.py $INPUT -o $OUTDIR/cca/clinical_labels
