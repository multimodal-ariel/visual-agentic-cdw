import numpy as np
import os
import json
from tqdm import tqdm
import cv2

datasets = ['DATASET1', 'DATASET2', 'DATASET3']

prompt_data = json.load(open("class_prompts.json"))

size = 512

def get_axis(data):
    # get the axis to slice the 3D volume
    
    img = data['imgs']
    spacing = data['spacing']
    
    shape = img.shape
    # get shape difference between the axes
    diff_ratio = [2*abs(shape[1]-shape[2])/(shape[1]+shape[2]),
            2*abs(shape[0]-shape[2])/(shape[0]+shape[2]),
            2*abs(shape[0]-shape[1])/(shape[0]+shape[1])]
    
    if diff_ratio[0] < 0.5:
        valid_axis = [0]
    else:
        min_axis = np.argmin(shape)
    
        print(f'Adding axis {min_axis} to valid axis', shape, spacing)
        valid_axis = [min_axis]
    
    # check if the volume is nearly cubic
    if (spacing.max() - spacing.min())/spacing.max() > 0.1:
        return valid_axis
    
    # check if the volume is nearly isotropic
    if max(diff_ratio) < 0.5:
        # use all axes
        valid_axis = [0, 1, 2]
        
    return valid_axis
    

def process_3D_volume(image_path, target_path, size):
    try:
        # load the 3D volume
        data = np.load(image_path, allow_pickle=True)
    except Exception as e:
        try:
            data = np.load(image_path, allow_pickle=False)
        except Exception as e:
            print(f'Error loading {image_path}: {e}')
            return []
        
    file_name = os.path.basename(image_path)
    data_name = image_path.split('/')[-3]
    instance_label = prompt_data[data_name]['instance_label']
    
    valid_axis = get_axis(data)
    
    annotations = []
    annotations1 = []
    annotations2 = []
    
    for axis in valid_axis:
        if axis != 0:
            print(f'Processing {file_name} on axis {axis}...')
            # create target directory
            save_path = target_path + f'_view{axis}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                os.makedirs(os.path.join(save_path, 'train'))
                os.makedirs(os.path.join(save_path, 'train_mask'))
        else:
            save_path = target_path
        # move the corresponding axis to the first axis
        imgs = np.moveaxis(data['imgs'], axis, 0)
        gts = np.moveaxis(data['gts'], axis, 0)
        
        n_slices = imgs.shape[0]
        # pad to square with equal padding on both sides
        imgs = pad_and_resize(imgs.astype(np.uint8), size)
        gts = pad_and_resize(gts.astype(np.uint8), size, is_gt=True)
        
        for i in range(n_slices):
            if axis == 0:
                img_name = f'{file_name[:-4]}_{i:03d}.png'
            else:
                img_name = f'{file_name[:-4]}_view{axis}_{i:03d}.png'
                
            # class ids in the gt
            if instance_label:
                class_ids = [1] if gts[i].max() > 0 else []
            else:
                class_ids = np.unique(gts[i])[1:]
                class_ids = [int(x) for x in class_ids]
            
            ann = {"mask_file": img_name, "file_name": img_name, "split": "train", 
                'class_ids': class_ids, 'instance_label': instance_label}
            
            if axis == 0:
                annotations.append(ann)
            elif axis == 1:
                annotations1.append(ann)
            elif axis == 2:
                annotations2.append(ann)
            else:
                raise ValueError(f'Invalid axis {axis}')
            
            # save slice with neighboring slices as RGB
            i1 = i-1 if i > 0 else i+1
            i2 = i+1 if i < n_slices-1 else i-1
            if n_slices == 1:
                i1 = i
                i2 = i
            img = imgs[np.array([i, i1, i2])]
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(os.path.join(save_path, 'train', img_name), img.astype(np.uint8))
            
            # save gt 
            cv2.imwrite(os.path.join(save_path, 'train_mask', img_name), gts[i].astype(np.uint8))
            
    return annotations, annotations1, annotations2
        
    
def pad_and_resize(vol, size, is_gt=False):
    # pad to square with equal padding on both sides
    # vols is a 3D numpy arrays of the shape (D, H, W)
    shape = vol.shape[1:]
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (0,0), (pad, pad))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((0,0), (pad, pad), (0,0))
    else:
        pad_width = None
    
    if pad_width is not None:
        vol = np.pad(vol, pad_width, mode='constant', constant_values=0)
    
    n_slices = vol.shape[0]
    # resize to 512x512
    resized_vol = np.zeros((n_slices, size, size))
    for i in range(n_slices):
        if not is_gt:
            resized_vol[i] = cv2.resize(vol[i], (size, size), interpolation=cv2.INTER_CUBIC)
        else:
            resized_vol[i] = cv2.resize(vol[i], (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_vol



for data_name in datasets:
    print(f'Processing {data_name}...')
    target_path = os.path.join(data_name, 'processed')
    # if os.path.exists(os.path.join(target_path, 'train.json')):
    #     print(f'{data_name} already processed.')
    #     continue
    # check data_name exists in prompt_data
    if data_name not in prompt_data:
        print(f'{data_name} not in prompt_data.')
        continue
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        os.makedirs(os.path.join(target_path, 'train'))
        os.makedirs(os.path.join(target_path, 'train_mask'))
    
    annotations = []
    annotations1 = []
    annotations2 = []
    # create target directory
    for file in tqdm(os.listdir(os.path.join(data_name, 'train'))):
        image_path = os.path.join(data_name, 'train', file)

        anns, anns1, anns2 = process_3D_volume(image_path, target_path, size)
        annotations += anns
        annotations1 += anns1
        annotations2 += anns2
        
    data = {'class_prompts': prompt_data[data_name], 'annotations': annotations}
    with open(os.path.join(target_path, 'train.json'), 'w') as f:
        json.dump(data, f, indent=4)
        
    if len(annotations1) > 0:
        data = {'class_prompts': prompt_data[data_name], 'annotations': annotations1}
        with open(os.path.join(target_path+'_view1', 'train.json'), 'w') as f:
            json.dump(data, f, indent=4)
    if len(annotations2) > 0:
        data = {'class_prompts': prompt_data[data_name], 'annotations': annotations2}
        with open(os.path.join(target_path+'_view2', 'train.json'), 'w') as f:
            json.dump(data, f, indent=4)