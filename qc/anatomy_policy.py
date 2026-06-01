"""Anatomy-aware QC/radiomics target policy.

This module centralizes hard-coded clinical rules that decide which masks are
meaningful for a case. The runner should not know that cardiac means heart-only
or that SynthSeg head cases use brain-only masks; it should ask this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MEDSEGQC_ORGANS = ("liver", "kidney_left", "kidney_right", "spleen")

SUPPORTED_CT_ANATOMIES = {
    "chest",
    "abdomen",
    "pelvis",
    "abdomen_pelvis",
    "chest_abdomen_pelvis",
    "whole_body",
    "cardiac",
    "head",
}

SUPPORTED_MRI_ANATOMIES = {
    "abdomen",
    "abdomen_pelvis",
    "cardiac",
    "spine",
    "pelvis",
    "head",
}

TARGET_ORGAN_OVERRIDES = {
    # Non-abdominal anatomies should not inherit broad body-organ lists.
    "cardiac": ["heart"],
    "spine": ["spine"],
    "pelvis": ["prostate"],
    "head": ["brain"],
}

LUNG_ELIGIBLE_CT_ANATOMIES = {
    "chest",
    "abdomen",
    "abdomen_pelvis",
    "chest_abdomen_pelvis",
    "whole_body",
}


@dataclass(frozen=True)
class AnatomyQCTargets:
    metadata: dict[str, Any]
    target_organs: list[str]
    skip_reason: str = ""

    @property
    def modality(self) -> str:
        return str(self.metadata.get("modality", "UNKNOWN")).upper()

    @property
    def anatomy(self) -> str:
        return str(self.metadata.get("anatomy", "unknown")).lower()


def is_ct_like_modality(modality: str) -> bool:
    return modality.upper() in {"CT", "PET_CT"}


def is_lung_shortcut_eligible(modality: str, anatomy: str) -> bool:
    return is_ct_like_modality(modality) and anatomy.lower() in LUNG_ELIGIBLE_CT_ANATOMIES


def medsegqc_supported_organs(target_organs: list[str]) -> list[str]:
    return [organ for organ in target_organs if organ in MEDSEGQC_ORGANS]


def target_override_for_anatomy(anatomy: str) -> list[str]:
    return list(TARGET_ORGAN_OVERRIDES.get(str(anatomy).lower(), []))


def needs_reference_organs(metadata: dict[str, Any]) -> bool:
    modality = str(metadata.get("modality", "UNKNOWN")).upper()
    anatomy = str(metadata.get("anatomy", "unknown")).lower()
    if not metadata.get("is_diagnostic", True):
        return False
    if target_override_for_anatomy(anatomy):
        return False
    if is_ct_like_modality(modality):
        return anatomy in SUPPORTED_CT_ANATOMIES
    if modality == "MRI":
        return anatomy in SUPPORTED_MRI_ANATOMIES
    return False


def resolve_qc_targets(metadata: dict[str, Any], expected_organs: list[str]) -> AnatomyQCTargets:
    """Return the anatomy-relevant QC/radiomics targets for one case.

    ``expected_organs`` comes from the general organ reference for broad body
    regions. Overrides here intentionally narrow non-abdominal FOVs to the only
    meaningful structures we currently trust.
    """

    modality = str(metadata.get("modality", "UNKNOWN")).upper()
    anatomy = str(metadata.get("anatomy", "unknown")).lower()

    if not metadata.get("is_diagnostic", True):
        return AnatomyQCTargets(
            metadata=metadata,
            target_organs=[],
            skip_reason=metadata.get("skip_reason", "non-diagnostic case"),
        )

    if is_ct_like_modality(modality):
        if anatomy not in SUPPORTED_CT_ANATOMIES:
            return AnatomyQCTargets(
                metadata=metadata,
                target_organs=[],
                skip_reason=f"unsupported CT anatomy: {anatomy}",
            )
    elif modality == "MRI":
        if anatomy not in SUPPORTED_MRI_ANATOMIES:
            allowed = ", ".join(sorted(SUPPORTED_MRI_ANATOMIES))
            return AnatomyQCTargets(
                metadata=metadata,
                target_organs=[],
                skip_reason=f"unsupported MRI anatomy: {anatomy} (allowed: {allowed})",
            )
    else:
        return AnatomyQCTargets(
            metadata=metadata,
            target_organs=[],
            skip_reason=f"unsupported modality: {modality}",
        )

    organs = list(TARGET_ORGAN_OVERRIDES.get(anatomy, expected_organs))
    if not organs:
        return AnatomyQCTargets(
            metadata=metadata,
            target_organs=[],
            skip_reason=f"no expected organs for anatomy: {anatomy}",
        )
    return AnatomyQCTargets(metadata=metadata, target_organs=organs)
