## Data Preparation
Your finetuning datasets should be stored under ```<YOUR MODEL AND DATA DIR>/data```. We support mixed training and sequential evaluation on multiple datasets. Each finetuning dataset should have 3D images and corresponding segmentation masks, as well as textual description of each class. The class definition can be different across different datasets, as long as they are described by the text prompts correctly.

### Data Format
We follow the format of the CVPR 2025 Text-guided 3D Segmentation Challenge [`dataset`](https://huggingface.co/datasets/junma/CVPR-BiomedSegFM). Specifically, each training example is an ```npz``` file containing:
- ```imgs```: image data; shape: (D,H,W); Intensity range: [0, 255]
- ```gts```: ground truth; shape: (D,H,W);
- ```spacing```

#### Folder Structure
```
<YOUR MODEL AND DATA DIR>/data/
├── class_prompts.json
├── DATASET1/
|   └── train/
|       ├── ex1.npz
|       ├── ex1.npz
│       └── ...
├── DATASET2/
|   └── train/
|       ├── ex1.npz
|       ├── ex1.npz
│       └── ...
└── DATASET3/
    └── train/
        ├── ex1.npz
        ├── ex1.npz
        └── ...
```

```class_prompts.json``` is a json file containing the prompts for each datasets dataset, e.g.

```
{
    "DATASET1": { 
        "1": [
            "Liver",
            "Liver in abdominal CT",
            "CT imaging of the liver in the abdomen",
            ...
        ],
        "2": [
            "Pancreas",
            "Pancreas in abdominal CT",
            "CT imaging of the pancreas in the abdomen",
            ...
        ],
        "instance_label": 0
    },
    "DATASET2": {
        "1": [
            "Lesion",
            "Lesion in whole body PET",
            "PET imaging of the lesion in the whole body region",
            ...
        ],
        "instance_label": 1
    },
    "DATASET3": { 
        "1": [
            "Intra-meatal region of vestibular schwannoma",
            "Intra-meatal region of vestibular schwannoma in brain MR",
            "MR imaging of intra-meatal region of vestibular schwannoma in brain",
            ...
        ],
        "2": [
            "Right cochlea",
            "Right cochlea in brain MR",
            "MR imaging of right cochlea in brain",
            ...
        ],
        "instance_label": 0
    }
}
```

```'instance_label'``` indicates the definition for the segmentation masks. 0 denotes semantic segmentation, where each index in ```gts``` corresponds to the index of the text prompts. 1 denotes instance segmentation where each index in ```gts``` corresponds to the instance index (e.g. cell #17 or lesion #2) of the same class, in which case there should only be one class for the dataset.

### Data Processing
BiomedParse v2 uses fractal volumetric encoding to combine multiple slices in a 3D volume into a single RGB image in 2D. Therefore, the model can be trained in 2D completely, and able to do inference in 3D using volumetric context. We provides a simple processing script ```process_2D.py```. Simply put it under your data folder, specify the dataset names in the script, and run it. This will process all the datasets and save the 2D images, which are ready for finetuning. The folder structure after the processing will be

```
<YOUR MODEL AND DATA DIR>/data/
├── process_2D.py
├── class_prompts.json
├── DATASET1/
|   ├── train/
|   |   ├── ex1.npz
|   |   ├── ex1.npz
│   |   └── ...
|   └── processed/
|       ├── train
|       |   ├── slice1.png
|       |   ├── slice2.png
│       |   └── ...
|       ├── train_mask
|       |   ├── slice1.png
|       |   ├── slice2.png
│       |   └── ...
│       └── train.json
└── ...
```

### Config Setup

Datasets are defined using modular configs that allow combining multiple datasets.  
Example configuration:

```yaml
_target_: azureml.acft.image.components.olympus.core.ModuleDatasets
train:
  _target_: torch.utils.data.ConcatDataset
  _partial_: True
  datasets:
    - _target_: src.datasets.biomedparse_dataset.BiomedParseDataset
      root_dir: ${mounts.external}/data/DATASET1/processed
    - _target_: src.datasets.biomedparse_dataset.BiomedParseDataset
      root_dir: ${mounts.external}/data/DATASET2/processed
    - _target_: src.datasets.biomedparse_dataset.BiomedParseDataset
      root_dir: ${mounts.external}/data/DATASET3/processed
```
Put it in ```configs/datamodule/datasets/biomedparse/biomedparse_finetune_dataset.yaml```, and you are ready to run finetuning jobs using the datasets.


### Evaluation
Finetuning evaluation data format follows exactly the same as the main evaluation. Simply create a folder and put images and masks ```npz``` files under ```test/``` and ```test_mask/``` respectively.