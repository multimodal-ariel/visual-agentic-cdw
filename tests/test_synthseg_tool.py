import os

import nibabel as nib
import numpy as np

from tools.base_tool import ToolInput
from tools.synthseg import SynthSegTool
from orchestrator.pipeline import CasePipeline


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
    assert "restricted to brain/head MRI or CT" in output.error


def test_synthseg_accepts_head_ct_in_dry_run(tmp_path):
    case_path = tmp_path / "case"
    case_path.mkdir()
    image_path = case_path / "image_nifti.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4)), image_path)

    output = SynthSegTool(dry_run=True).run(
        ToolInput(
            image_path=str(image_path),
            case_path=str(case_path),
            modality="CT",
            anatomy="head",
            target_organs=["brain"],
        )
    )

    assert output.success
    assert output.organs_segmented == ["brain"]


def test_synthseg_head_ct_passes_ct_flag(tmp_path, monkeypatch):
    case_path = tmp_path / "case"
    case_path.mkdir()
    image_path = case_path / "image_nifti.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4)), image_path)

    synthseg_dir = tmp_path / "SynthSeg"
    runner = synthseg_dir / "scripts" / "commands" / "SynthSeg_predict.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# fake runner\n")
    labels_dir = synthseg_dir / "data" / "labels_classes_priors"
    labels_dir.mkdir(parents=True)
    for label_file in SynthSegTool.REQUIRED_LABEL_FILES:
        np.save(labels_dir / label_file, np.array([0, 1], dtype=np.int16))

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    seg_ckpt = ckpt_dir / "synthseg_robust_2.0.h5"
    qc_ckpt = ckpt_dir / "synthseg_qc_2.0.h5"
    seg_ckpt.write_text("fake")
    qc_ckpt.write_text("fake")

    captured = {}

    def fake_run_in_env(self, cmd, cwd, gpu_id, timeout):
        captured["cmd"] = cmd
        raw_seg = cmd[cmd.index("--o") + 1]
        labels = np.zeros((16, 16, 16), dtype=np.int16)
        labels[4:12, 4:12, 4:12] = 2
        nib.save(nib.Nifti1Image(labels, np.eye(4)), raw_seg)

    monkeypatch.setattr(SynthSegTool, "_run_in_env", fake_run_in_env)

    output = SynthSegTool(
        dry_run=False,
        synthseg_dir=str(synthseg_dir),
        segmentation_checkpoint=str(seg_ckpt),
        qc_checkpoint=str(qc_ckpt),
    ).run(
        ToolInput(
            image_path=str(image_path),
            case_path=str(case_path),
            modality="CT",
            anatomy="head",
            target_organs=["brain"],
        )
    )

    assert output.success
    assert "--ct" in captured["cmd"]
    assert "brain" in output.organs_segmented


def test_no_llm_registry_selects_synthseg_for_head_ct_mri_only():
    pipeline = CasePipeline(no_llm=True, dry_run=True)

    assert "SynthSeg" in pipeline._all_compatible_tools("CT", "head")
    assert "SynthSeg" in pipeline._all_compatible_tools("MRI", "head")
    assert pipeline._all_compatible_tools("CT", "unknown") == []
    assert pipeline._all_compatible_tools("MRI", "unknown") == []
    assert "SynthSeg" not in pipeline._all_compatible_tools("CT", "abdomen")


def test_no_llm_metadata_skips_known_modality_unknown_anatomy():
    pipeline = CasePipeline(no_llm=True, dry_run=True)

    meta = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/LUPUS/PATIENT/20240101/CT_OUTSIDE_FILM/AXIAL"
    )

    assert meta["modality"] == "CT"
    assert meta["anatomy"] == "unknown"
    assert meta["is_diagnostic"] is False
    assert "Could not infer anatomy" in meta["skip_reason"]


def test_no_llm_metadata_maps_brain_and_head_synonyms_to_head():
    pipeline = CasePipeline(no_llm=True, dry_run=True)

    brain = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/CONTR/PATIENT/20240101/"
        "MRI_BRAIN_W_WO_CONTRAST/t1_mprage_tra_p2_iso_1_0_POST"
    )
    head = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/CONTR/PATIENT/20240101/"
        "CTA_HEAD_W_WO_CONTRAST/CTA_HEAD__1_0__Hv45"
    )

    assert brain["modality"] == "MRI"
    assert brain["anatomy"] == "head"
    assert brain["is_diagnostic"] is True
    assert head["modality"] == "CT"
    assert head["anatomy"] == "head"
    assert head["is_diagnostic"] is True


def test_no_llm_metadata_guards_neuro_head_inference():
    pipeline = CasePipeline(no_llm=True, dry_run=True)

    neuro_brain = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/CONTR/PATIENT/20240101/"
        "MRI_NEURO_OUTSIDE_FILM_FOR_CONTINUED_CARE/FLAIR_AX"
    )
    neuro_spine = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/CONTR/PATIENT/20240101/"
        "MRI_NEURO_OUTSIDE_FILM_FOR_CONTINUED_CARE/Ax_T1__C_CERVICAL"
    )
    brainlab_sinus = pipeline._metadata_from_path(
        "/data/soumitri/segmentations_3d/CONTR/PATIENT/20240101/"
        "CT_MAXILLOFACIAL_WO_CONTRAST/BrainLab_Sinus__2_0__H70h"
    )

    assert neuro_brain["anatomy"] == "head"
    assert neuro_brain["is_diagnostic"] is True
    assert neuro_spine["anatomy"] == "spine"
    assert "SynthSeg" not in pipeline._all_compatible_tools(
        neuro_spine["modality"],
        neuro_spine["anatomy"],
    )
    assert brainlab_sinus["anatomy"] == "unknown"
    assert brainlab_sinus["is_diagnostic"] is False
