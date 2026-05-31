import os

import nibabel as nib
import numpy as np

from tools.base_tool import ToolInput
from tools.synthseg import SynthSegTool


def test_synthseg_dry_run_writes_brain_mask(tmp_path):
    case_path = tmp_path / "case"
    case_path.mkdir()
    image_path = case_path / "image_nifti.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4)), image_path)

    output = SynthSegTool(dry_run=True).run(
        ToolInput(
            image_path=str(image_path),
            case_path=str(case_path),
            modality="MRI",
            anatomy="head",
            target_organs=["brain"],
        )
    )

    mask_path = case_path / "segmentations_synthseg" / "brain.nii.gz"
    assert output.success
    assert output.organs_segmented == ["brain"]
    assert os.path.isfile(mask_path)

    mask = nib.load(mask_path)
    assert mask.shape == (16, 16, 16)
    assert np.allclose(mask.affine, np.eye(4))


def test_synthseg_rejects_non_head_mri(tmp_path):
    case_path = tmp_path / "case"
    case_path.mkdir()
    image_path = case_path / "image_nifti.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4)), image_path)

    output = SynthSegTool(dry_run=False, synthseg_dir=str(tmp_path)).run(
        ToolInput(
            image_path=str(image_path),
            case_path=str(case_path),
            modality="MRI",
            anatomy="abdomen",
            target_organs=["brain"],
        )
    )

    assert not output.success
    assert "restricted to brain/head MRI" in output.error
