import os
import sys
import math
import json
import torch
import nibabel as nib
import pydicom
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats

from inference_utils.processing_utils import read_nifti, read_dicom
from semantics_biomedparse import SEMANTIC_CLASSES, MODALITIES

def find_present_words(word_list, text):
    return [w for w in word_list if w.upper() in text.upper()]
    

def main(filelist, model, dtype):

    seg_errors = []
    for ii, filepath in enumerate(filelist):
        if "Protocol" in filepath:
            print(f"⚠️ Protocol detected in {filepath} -- SKIPPING...")
            continue
        if dtype == 'nifti':
            filepath = filepath.replace('/data/RAD/LUPUS', '/home/soumitri/segmentations_2d')
            
        # [CHECK IF SAME FOR 2D or NOT!!!!!!] 
        # get nested directory structure of original data
        filepath_subdirs = filepath.split("/")
        patient_id, timestamp = filepath_subdirs[4], filepath_subdirs[5] # hard-coded
        scan_name = filepath.split(timestamp+"/")[-1] # appendage to timestamp is the nested subdir of dicom folder structure
        print(OUTPUT_DIR, "|", patient_id, "|", timestamp, "|", scan_name)
        output_path = os.path.join(OUTPUT_DIR, patient_id, timestamp, scan_name)

        if dtype == "nifti":
            image_path = os.path.join(filepath, "image_nifti.nii.gz")
        else:
            raise NotImplementedError("only supports nifti inputs currently")

        # match anatomy of query image with pre-defined semantics
        image_anatomies = find_present_words(word_list=SEMANTIC_CLASSES.keys(), text=scan_name)
        image_modalities = find_present_words(word_list=MODALITIES, text=scan_name)

        try:
            image = read_nifti(
                image_path, is_CT="CT" in image_modalities,
                slice_idx=0, HW_index=(1,2)
            )

            # frame text prompts
            prompts = []
            for anatomy in image_anatomies:
                classes = SEMANTIC_CLASSES[anatomy]
                prompts.extend(classes)
            prompts = prompts + SEMANTIC_CLASSES["default"]

            # add modality + anatomy information in prompt
            anatomies_combined, modalities_combined = " ".join(image_anatomies), " ".join(image_modalities)
            prompts = [f"{modalities_combined} scan of {anatomies_combined} showing {p}" for p in prompts]

            # save text prompt file
            with open(os.path.join(output_path, "text_prompts.json"), "w") as f:
                json.dump(prompts, f)

            with torch.no_grad():
                # this is adapted verbatim from original inference code -- may need some tweaks
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(prompts, is_eval=True)
                # pass image into model
                pred_logits, pred_masks = interactive_infer_image(
                    model=model, 
                    image=image, 
                    prompts=prompts,
                )
                        
            # save predictions
            np.savez_compressed(os.path.join(output_path, f"segmentations_biomedparse.npz"), logits=pred_logits, preds=pred_masks, prompts=prompts)
            print(f"✅ Segmentation done [{ii+1}/{len(filelist)}]: {output_path}/segmentations_biomedparse.npz")

        except Exception as e:
            print(f"❌ Error during BiomedParse pass: {e}")
            seg_errors.append({
                "filepath": filepath,
                "error": str(e)
            })
            with open("../.logs/biomedparse_2d_seg_errors.json", "w") as f:
                json.dump(seg_errors, f)
            continue


if __name__ == "__main__":

    # ========== CONFIGURATION ==========
    # run: CUDA_VISIBLE_DEVICES=XX python get_segmentations_from_prompts.py XX
    
    GPU_ID = int(sys.argv[1])
    dtype = "nifti" # str(sys.argv[2]) # dicom or nifti
    
    DEVICE = "cuda"
    OUTPUT_DIR = "/home/soumitri/segmentations_2d"
    json_2d_list = "/home/soumitri/data_paths_linux/2d_scans_list.json"
    with open(json_2d_list, "r") as f:
        data = json.load(f)

    # Load model from pretrained weights
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    pretrained_pth = 'pretrained/biomedparse_v1.pt'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Extract the filepaths
    filelist = [item[0] for item in data]
    tot_len = len(filelist)
    # chunk based on GPU ID
    chunk_size = math.ceil(tot_len / 8)
    # Compute start and end indices for this GPU
    start_idx = GPU_ID * chunk_size
    end_idx = min(start_idx + chunk_size, tot_len)
    filelist = filelist[start_idx : end_idx]
    print(f"\n📊 No. of images to segment on GPU {GPU_ID}: {len(filelist)} / {tot_len} -- chunked from {start_idx} to {end_idx}")

    main(filelist, model, dtype)