# SynthSeg External Setup

The CDW wrapper expects the upstream SynthSeg clone at:

```text
external/SynthSeg/
```

Install it on the clinical server with:

```bash
cd external
git clone https://github.com/BBillot/SynthSeg.git
```

The wrapper also supports a custom location via:

```bash
export SYNTHSEG_DIR=/path/to/SynthSeg
```

## Environment

Create a dedicated environment named `cdw_synthseg`. The upstream README
recommends Python 3.8 with TensorFlow/Keras pins:

```bash
conda create -n cdw_synthseg python=3.8 tensorflow-gpu=2.2.0 keras=2.3.1 nibabel matplotlib -c anaconda -c conda-forge
conda activate cdw_synthseg
pip install protobuf==3.20.3 numpy==1.23.5
cd external/SynthSeg
python setup.py install
```

Download any missing upstream model files as instructed by the SynthSeg README
and place them under `external/SynthSeg/models/`.

## Wrapper Contract

The CDW wrapper calls:

```bash
python external/SynthSeg/scripts/commands/SynthSeg_predict.py --i image_nifti.nii.gz --o segmentations_synthseg/synthseg_raw_1mm.nii.gz --robust
```

SynthSeg writes labels at 1 mm isotropic resolution. The wrapper resamples that
multilabel output back to `image_nifti.nii.gz` with nearest-neighbour
interpolation, saves `synthseg_labels_resampled.nii.gz`, then writes per-label
binary masks plus a combined `brain.nii.gz`.
