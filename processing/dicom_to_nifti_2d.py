import os
import json
import numpy as np
import pydicom
import nibabel as nib
from nibabel.nifti1 import Nifti1Image, Nifti1Extension
from pydicom import dcmread

# utility functions for metadata handling across dicom and nifti

def export_nifti_metadata_to_json(nifti_path, json_out_path):
    img = nib.load(nifti_path)
    header = img.header

    # Convert to a regular dictionary with proper data types
    metadata = {k: header[k].tolist() if hasattr(header[k], "tolist") else str(header[k]) 
                for k in header.keys()}

    # Add extra metadata
    metadata["data_shape"] = header.get_data_shape()
    metadata["zooms"] = header.get_zooms()
    metadata["datatype"] = str(header.get_data_dtype())

    with open(json_out_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def extract_dicom_metadata(dicom_dir, output_json_path):
    metadata = {}
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                for elem in ds:
                    if elem.keyword and elem.keyword not in metadata:
                        metadata[elem.keyword] = str(elem.value)
                break  # just take the first file
    with open(output_json_path, "w") as out:
        json.dump(metadata, out, indent=2)

def compare_metadata(json1, json2):
    with open(json1) as f1, open(json2) as f2:
        meta1 = json.load(f1)
        meta2 = json.load(f2)

    common_keys = set(meta1.keys()) & set(meta2.keys())

    print("Common fields with differences:")
    for key in sorted(common_keys):
        if meta1[key] != meta2[key]:
            print(f"{key}:")
            print(f"  DICOM: {meta1[key]}")
            print(f"  NIfTI: {meta2[key]}\n")

    only_in_dicom = set(meta1.keys()) - set(meta2.keys())
    only_in_nifti = set(meta2.keys()) - set(meta1.keys())

    print("\nOnly in DICOM metadata:", only_in_dicom)
    print("Only in NIfTI metadata:", only_in_nifti)


# dicom to nifti conversion codes

def extract_selected_dicom_metadata(dicom):
    selected_keys = ["AccessionNumber", "Modality", "BodyPartExamined"] # can be changed afterwards
    metadata = {}
    for key in selected_keys:
        if hasattr(dicom, key):
            val = getattr(dicom, key)
            # Convert sequences and complex objects to strings
            metadata[key] = str(val)
    return metadata

def convert_dicom_to_nifti_2d(dicom_path_or_dir):
    # Load DICOM
    if os.path.isdir(dicom_path_or_dir):
        files = [f for f in os.listdir(dicom_path_or_dir) if f.lower().endswith(".dcm")]
        if not files:
            raise FileNotFoundError("No DICOM files found.")
        dcm = dcmread(os.path.join(dicom_path_or_dir, files[0]))
    else:
        dcm = dcmread(dicom_path_or_dir)

    # Extract image
    image = dcm.pixel_array.astype(np.float32)
    volume = np.expand_dims(image, axis=0)  # (1, H, W)

    # Spacing
    pixel_spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
    slice_thickness = getattr(dcm, "SliceThickness", 1.0)
    spacing = [float(slice_thickness)] + [float(s) for s in pixel_spacing] if slice_thickness is not None and pixel_spacing is not None else [1.0, 1.0, 1.0]

    # Affine
    origin = np.array(getattr(dcm, "ImagePositionPatient", [0, 0, 0]))
    orient = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    row_cos = np.array(orient[:3])
    col_cos = np.array(orient[3:])
    normal = np.cross(row_cos, col_cos)

    affine = np.eye(4)
    affine[:3, 0] = row_cos * spacing[2]
    affine[:3, 1] = col_cos * spacing[1]
    affine[:3, 2] = normal * spacing[0]
    affine[:3, 3] = origin

    # DCM metadata
    metadata = extract_selected_dicom_metadata(dcm)
    meta_json = json.dumps(metadata, indent=2)
    ext = Nifti1Extension(6, meta_json.encode("utf-8"))

    return volume, affine, ext

def save_nifti_with_metadata(volume, affine, ext, output_path):
    nifti_img = Nifti1Image(volume, affine)
    nifti_img.header.extensions.append(ext)
    nib.save(nifti_img, output_path)
    print(f"✅ Saved NIfTI to: {output_path}")

def save_nifti(volume, affine, output_path):
    nifti_img = Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)
    print(f"✅ Saved NIfTI to: {output_path}")

# ========== USAGE ==========
if __name__ == "__main__":

    OUTPUT_DIR = "/data/soumitri/segmentations_2d"
    # json_2d_list = "/home/soumitri/data_paths_linux/2d_scans_list.json"
    # with open(json_2d_list, "r") as f:
    #     data = json.load(f)
    # # Extract the filepaths
    # dicom_dir_list = [item[0] for item in data]

    json_2d_list = "/data/soumitri/new_data_paths/2d_scans_list.json"
    with open(json_2d_list, "r") as f:
            data = json.load(f)
    
    # Extract the filepaths
    data = [item for item in data if "XR_" in item[0]]
    remaining_to_convert = [item[0] for item in data if not os.path.exists(item[0].replace("/data/RAD", "/data/soumitri/segmentations_2d"))]
    print(f"\n📊 No. of XR images remaining to convert: {len(remaining_to_convert)} / {len(data)}")
    dicom_dir_list = remaining_to_convert

    dicom_to_nifti_errors = []
    for dicom_path in dicom_dir_list:
        output_path = dicom_path.replace("/data/RAD", OUTPUT_DIR)
        os.makedirs(output_path, exist_ok=True)
        output_nifti_path = os.path.join(output_path, "image_nifti.nii.gz")

        # check if the file already exists, if so, skip
        if os.path.exists(output_nifti_path):
            print(f"✅✅ [{output_nifti_path}] already exists i.e. has been converted previously! Skipping...")
            continue
        try:
            volume, affine, ext = convert_dicom_to_nifti_2d(dicom_path)
            save_nifti_with_metadata(volume, affine, ext, output_nifti_path)
            print(f"✅ DICOM conversion successful: {dicom_path} ===> {output_nifti_path}")
        except Exception as e:
            dicom_to_nifti_errors.append({'dicom_path': dicom_path, 'exception': str(e)})
            print(f"⚠️ Could not convert {dicom_path} to NIfTI: {e}")
    
    with open(".jsons/dicom_to_nifti_2d_errors_v3.json", "w", encoding="utf-8") as f:
        json.dump(dicom_to_nifti_errors, f, indent=2)
        
    
