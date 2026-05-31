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

Download the upstream model files from the SynthSeg model link. The original
README points to a UCL Dropbox/SharePoint folder:

https://liveuclac-my.sharepoint.com/:f:/g/personal/rmappmb_ucl_ac_uk/EtlNnulBSUtAvOP6S99KcAIBYzze7jTPsmFk2_iHqKDjEw

If that link is unavailable, the upstream author later re-uploaded the models
at this MIT SharePoint link:

https://mitprod-my.sharepoint.com/:u:/g/personal/bbillot_mit_edu/Ebqxo6YgUmBJkOML0m8NSXgBrhaHG7iqClFXRXPinS6FGw

Then keep the canonical copies under the repo checkpoint tree:

```text
checkpoints/SynthSeg/
  synthseg_robust_2.0.h5
  synthseg_qc_2.0.h5
  synthseg_2.0.h5              # optional if running without --robust
  synthseg_parc_2.0.h5         # optional if running with parcellation
```

The wrapper links or copies the required files into `external/SynthSeg/models/`
at runtime because the upstream `SynthSeg_predict.py` CLI expects that folder.
You can override paths with:

```bash
export SYNTHSEG_CHECKPOINT=/path/to/synthseg_robust_2.0.h5
export SYNTHSEG_QC_CHECKPOINT=/path/to/synthseg_qc_2.0.h5
export SYNTHSEG_PARC_CHECKPOINT=/path/to/synthseg_parc_2.0.h5
```

## Wrapper Contract

The CDW wrapper calls:

```bash
python external/SynthSeg/scripts/commands/SynthSeg_predict.py --i image_nifti.nii.gz --o segmentations_synthseg/synthseg_raw_1mm.nii.gz --robust
```

For CT head cases, the wrapper adds SynthSeg's CT-specific preprocessing flag:

```bash
python external/SynthSeg/scripts/commands/SynthSeg_predict.py --i image_nifti.nii.gz --o segmentations_synthseg/synthseg_raw_1mm.nii.gz --robust --ct
```

SynthSeg writes labels at 1 mm isotropic resolution. The wrapper resamples that
multilabel output back to `image_nifti.nii.gz` with nearest-neighbour
interpolation, saves `synthseg_labels_resampled.nii.gz`, then writes per-label
binary masks plus a combined `brain.nii.gz`.
