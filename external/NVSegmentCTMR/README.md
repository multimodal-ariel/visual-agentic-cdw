# NV-Segment-CTMR Overview

## Description:
NV-Segment-CTMR is a specialized foundation model for 3D medical image segmentation that excels at accurate, adaptable, automatic segmentation across anatomies and modalities, including computed tomography (CT) and magnetic resonance (MR) imaging. NV-Segment-CTMR adapts to varying conditions and anatomical regions, enabling comprehensive automated annotation workflows.

At the core of NV-Segment-CTMR are two automated workflows. Segment Everything enables whole-body exploration, which is crucial for understanding complex diseases affecting multiple organs and for holistic treatment planning. Segment by Class provides detailed sectional views based on specific classes, supporting targeted disease analysis or organ mapping, such as tumor identification in critical organs. 

This model is for research and development only.

## Github links: 
https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main

## Run pipeline:
For running the pipeline, NV-Segment-CTMR requires at least one prompt for segmentation. It supports label prompt, which is the index of the class for automatic segmentation. NV-Segment-CTMR does not support point based interactive segmentation. For interactive model, please refer to [VISTA3D](https://github.com/Project-MONAI/VISTA/tree/main/vista3ds)

Here is a code snippet to showcase how to execute inference with this model.
```python
import os
import tempfile

import torch
from hugging_face_pipeline import HuggingFacePipelineHelper


FILE_PATH = os.path.dirname(__file__)
with tempfile.TemporaryDirectory() as tmp_dir:
    output_dir = os.path.join(tmp_dir, "output_dir")
    pipeline_helper = HuggingFacePipelineHelper("vista3d")
    pipeline = pipeline_helper.init_pipeline(
        os.path.join(FILE_PATH, "vista3d_pretrained_model"),
        device=torch.device("cuda:0"),
    )
    inputs = [
        {
            "image": "/data/Task09_Spleen/imagesTs/spleen_1.nii.gz",
            "label_prompt": [3],
        },
        {
            "image": "/data/Task09_Spleen/imagesTs/spleen_11.nii.gz",
            "modality": 'CT_BODY',
        },
        {
            "image": "/data/Task09_Spleen/imagesTs/spleen_11.nii.gz",
            "modality": 'MRI_BODY'
        },
    ]
    pipeline(inputs, output_dir=output_dir)

```
The inputs defines the image to segment and the prompt for segmentation.
```python 
inputs = {'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'label_prompt':[1]}
```
- The inputs must include the key `image` which contain the absolute path to the nii image file, and includes prompt keys of `label_prompt`.
- The `label_prompt` is a list of length `B`, which can perform `B` foreground objects segmentation, e.g. `[2,3,4,5]`. The full list of label definition is in `metadata.json`.
- If no prompt is provided, user can use `modality` to use predefined class indices. Supported modality includes `CT_BODY`, `MRI_BODY`, `MRI_BRAIN`.  
```
Note: For brain structure segmentation, current model only support standard brain T1 images. The brain T1 images must be preprocessed with skull stripping and normalization. Follow https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing to process the brain images
```

### License/Terms of Use:
NVIDIA OneWay Non-Commercial License for academic research purposes

### Deployment Geography:
Global

### Use Case:
Medical researchers, AI developers, and healthcare institutions are expected to use this system to perform automated medical image segmentation, conduct multi-organ analysis, and accelerate annotation workflows in research applications.

### Release Date:
Huggingface: 10/27/2025 via https://huggingface.co/NVIDIA

## Reference(s):
[1] He, Yufan, et al. "VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging." arXiv preprint arXiv:2406.05285. 2024. https://arxiv.org/abs/2406.05285

## Model Architecture:
**Architecture Type:** Transformer
**Network Architecture:** SAM-like architecture for 3D medical imaging segmentation

This model was developed from scratch using MONAI components.
**Number of model parameters:** 218M

## Input:
**Input Type(s):** Image
**Input Format(s):** Neuroimaging Informatics Technology Initiative (NIfTI)
**Input Parameters:** Three-Dimensional (3D)
**Other Properties Related to Input:** Supports both computed tomography (CT) and magnetic resonance (MR) imaging modalities. It also supports optional class information for targeted segmentation workflows.

### Input Modalities:
- **CT Images:** 3D computed tomography volumes
- **MR Images:** 3D magnetic resonance volumes
- **Class Selection:** Optional class indices for targeted segmentation workflows

## Output:
**Output Type(s):** Image
**Output Format:** Neuroimaging Informatics Technology Initiative (NIfTI)
**Output Parameters:** Three-Dimensional (3D)
**Other Properties Related to Output:** Segmentation masks with up to 345+ anatomical classes, providing comprehensive organ and tissue delineation for medical imaging analysis.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (GPU cores) and software frameworks (CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):**
* MONAI Core v.1.5.0

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Hopper

**Supported Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
0.1 - Initial release version for 3D medical imaging segmentation with multi-modality support

## Training, Testing, and Evaluation Datasets:

### Dataset Overview:
**Total Size:** ~31k
**Total Number of Datasets:** 32 datasets

Public datasets from multiple scanner types were processed to create standardized 3D medical imaging volumes with expert-validated anatomical segmentation masks across diverse anatomical regions and pathological conditions. The data processing pipeline ensured consistent voxel spacing, standardized orientations, and validated anatomical segmentations.

## Training Dataset:
**Data Modality:**
* Image

**Image Training Data Size:**
* Less than a Million Images

**Data Collection Method by dataset:**
* Hybrid: Human, Automatic/Sensors

**Labeling Method by dataset:**
* Hybrid: Human, Automatic/Sensors

## Testing Dataset:
**Data Collection Method by dataset:**
* Hybrid: Human, Automatic/Sensors

**Labeling Method by dataset:**
* Hybrid: Human, Automatic/Sensors

## Evaluation Dataset:
**Data Collection Method by dataset:**
* Hybrid: Human, Automatic/Sensors

**Labeling Method by dataset:**
* Hybrid: Human, Automatic/Sensors

## Inference:
**Acceleration Engine:** PyTorch
**Test Hardware:**
* A100
* H100

## Additional Information:
### Available Anatomical Classes (345+ total):
NV-Segment-CTMR supports comprehensive anatomical segmentation with the following categories:

**Core Organs and Systems:**
- **Abdominal organs:** liver (1), kidney (2), spleen (3), pancreas (4), gallbladder (10), stomach (12), bladder (15), colon (62)
- **Cardiovascular:** heart (115), aorta (6), inferior vena cava (7), superior vena cava (125), portal and splenic veins (17)
- **Respiratory:** lung (20), trachea (57), airway (132), individual lung lobes (28-32)
- **Neurological:** brain (22), spinal cord (121), complete brain structures (214-345)

**Skeletal System:**
- **Spine:** Complete vertebral column from C1-S1 (33-56, 127)
- **Thoracic:** Bilateral ribs 1-12 (63-86), sternum (122), costal cartilages (114)
- **Appendicular:** Bilateral long bones, joints, and extremities (87-96)

**Detailed Brain Segmentation:**
Comprehensive brain parcellation including ventricles, cortical regions, subcortical structures, and specialized brain areas (214-345) based on neuroanatomical atlases.

**Pathological Structures:**
- **Tumors:** lung tumor (23), pancreatic tumor (24), hepatic tumor (26), brain tumor (176)
- **Cancer:** colon cancer primaries (27)
- **Lesions:** bone lesion (128)
- **Cysts:** bilateral kidney cysts (116-117)
- **Note:** We recommend the `NV-Segment-CT` model for better tumor performance. 

**Specialized Regions:**
- **Head and neck:** detailed facial structures, sensory organs, and cranial anatomy (172-213)
- **Cardiac:** heart chambers, major vessels, and cardiac-specific structures (108, 149-155)
- **Reproductive:** prostate zones (118, 147-148), uterocervix (161), gonads (160)

*Complete numerical mapping and deprecated classes available in model documentation.*

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
