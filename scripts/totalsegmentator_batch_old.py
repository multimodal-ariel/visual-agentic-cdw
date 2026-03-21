import os
import sys
import math
import json
import subprocess
import concurrent.futures
import time
from datetime import datetime

# ========== CONFIGURATION ==========
# CRITICAL: Controls how many TotalSegmentator instances run at once.
# Set to 1 for standard GPUs (start here).
# Set to 2 or 3 ONLY if you have >24GB VRAM (e.g., A6000, A100).
MAX_WORKERS = 3

# ========== STEP 1: WORKER FUNCTION ==========
def process_single_patient(dicom_path, output_dir_root, device_str, dtype):
    """
    Worker function to process a single patient.
    Handles path parsing and subprocess calls.
    """
    try:
        # # 1. Parse Paths
        # # Assuming structure: /data/.../PatientID/Timestamp/ScanName
        # parts = dicom_path.strip("/").split("/")
        # # Adjust indices [-3] [-2] [-1] based on your actual depth if standard split fails
        # patient_id = parts[-3] 
        # timestamp = parts[-2]
        # scan_name = parts[-1]

        # output_path = os.path.join(output_dir_root, patient_id, timestamp, scan_name)

        
        # NIfTI Path Logic
        # input_file = dicom_path
        if dtype == 'nifti':
            # Mirroring your original path replacement logic
            input_file = dicom_path.replace('/data/RAD', output_dir_root)
            input_file = os.path.join(input_file, "image_nifti.nii.gz")

        if not os.path.exists(input_file):
            return f"⚠️ {input_file} does not exist. Refer to dicom2nifti conversion."
        
        output_path = input_file.replace('image_nifti.nii.gz', "segmentations_combined.nii.gz")

        # 2. Check if already processed (Optional resume capability)
        final_seg_dir = output_path.replace("segmentations_combined.nii.gz", "segmentations_separate")
        if os.path.exists(final_seg_dir) and len(os.listdir(final_seg_dir)) > 0:
            return f"⏩ Skipped (Exists): {input_file}" # {patient_id}/{timestamp}"

        # 3. Define Commands
        # We run both commands sequentially for THIS patient, but in parallel with OTHER patients
        
        # Task A: Multi-label (--ml)
        # cmd_combined = [
        #     "TotalSegmentator", "-i", input_file, "-o", output_path,
        #     "-ot", "nifti", "--ml", "--statistics", "--device", device_str
        # ]
        
        # Task B: Standard (Separate organs)
        # Note: Writing to same output dir might overwrite? Ensure logic is intended.
        cmd_separate = [
            "TotalSegmentator", "-i", input_file, "-o", final_seg_dir,
            "-ot", "nifti", "--statistics", "--device", device_str
        ]

        # 4. Execute
        # Run Combined
        # subprocess.run(cmd_combined, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Run Separate
        subprocess.run(cmd_separate, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # return f"✅ Success: {input_file} --> {output_path}"
        
        return f"\n✅ Success: {input_file} --> {final_seg_dir}"

    except subprocess.CalledProcessError as e:
        return f"❌ Subprocess Failed: {dicom_path} | {e}"
    except Exception as e:
        return f"⚠️ Error: {dicom_path} | {e}"


# ========== STEP 2: MAIN WORKFLOW ==========
def main(dicom_dir_list, gpu_id, dtype, output_dir):
    
    device_str = f"gpu:{gpu_id}"
    total_scans = len(dicom_dir_list)
    print(f"🚀 Starting processing on GPU {gpu_id} | Workers: {MAX_WORKERS} | Scans: {total_scans}")

    # ProcessPoolExecutor allows multiple CPU processes to launch commands
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # Submit all jobs
        # We map the function to the arguments
        futures = {
            executor.submit(process_single_patient, path, output_dir, device_str, dtype): path 
            for path in dicom_dir_list
        }

        # Monitor Progress
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            print(f"[{completed}/{total_scans}] {result}")

if __name__ == "__main__":
    
    # Arg parsing
    if len(sys.argv) < 3:
        print("Usage: python script.py <GPU_ID> <dtype>")
        sys.exit(1)

    GPU_ID = int(sys.argv[1])
    DTYPE = str(sys.argv[2]) # 'dicom' or 'nifti'
    
    OUTPUT_DIR = "/data/soumitri/segmentations_3d"
    JSON_LIST = "/data/soumitri/new_data_paths/reconvert_cases.json"
    # JSON_LIST = "/data/soumitri/new_data_paths/3d_scans_list.json"

    # Load List
    with open(JSON_LIST, "r") as f:
        data = json.load(f)
    
    # Handle list of tuples or list of strings
    full_list = []
    for item in data:
        full_list.append(item[0] if isinstance(item, (list, tuple)) else item)

    # Chunking logic for multi-GPU setups
    tot_len = len(full_list)
    chunk_size = math.ceil(tot_len / 8) # Assuming 8 GPUs total available
    start_idx = GPU_ID * chunk_size
    end_idx = min(start_idx + chunk_size, tot_len)
    
    my_chunk = full_list[start_idx : end_idx]
    
    print(f"\n📂 Loaded {len(my_chunk)} scans for this node:{GPU_ID} (Index {start_idx}-{end_idx})")

    start_time = time.time()
    main(my_chunk, GPU_ID, DTYPE, OUTPUT_DIR)
    end_time = time.time()
    
    print(f"\n🏁 Done in {(end_time - start_time)/60:.2f} minutes.")