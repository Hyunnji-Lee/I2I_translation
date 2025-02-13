'''
This file organize followed data path. And all file saves .png file form.

------------------------------------
<original path>
base_dir : data\Task2\brain

data\Task2\brain\2BA001
                        \cbct.nii.gz
                        \ct.nii.gz
                        \mask.nii.gz
data\Task2\brain\2BA002
......
data\Task2\brain\2BC090
data\Task2\brain\overview

-------------------------------------
<revised path after running this py file>
base_dir : data\Task2\brain_data

data\Task2\brain_data\train\A
data\Task2\brain_data\train\B
data\Task2\brain_data\val\A
data\Task2\brain_data\val\B
data\Task2\brain_data\test\A
data\Task2\brain_data\test\B
--------------------------------------

cbct : input(reference), ct : ground truth
'''
#%%
import os
import shutil
import numpy as np
import nibabel as nib
import random

base_dir = "data/Task2/brain" 

# Define the paths for new directories
result_dir = 'data/Task2/brain_data_npy_sub'

dir_save_train = os.path.join(result_dir, 'train')
dir_save_val = os.path.join(result_dir, 'val')
dir_save_test = os.path.join(result_dir, 'test')

# Create new directories if they don't exist
for dir_path in [dir_save_train, dir_save_val, dir_save_test]:
    os.makedirs(dir_path, exist_ok=True)

# all data
patient_list = os.listdir(base_dir)
patient_list = [x for x in patient_list if '2B' in x]

# set train/val/test indices (0.8:0.1:0.1)
split_ratio = (0.8, 0.1, 0.1)
random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)

indices = np.arange(len(patient_list))
np.random.shuffle(indices)

train_size = int(len(indices) * split_ratio[0])
val_size = int(len(indices) * split_ratio[1])

train_indices = indices[:train_size]
val_indices = indices[train_size: (train_size + val_size)]
test_indices = indices[train_size + val_size:]

all_mode = [train_indices, val_indices, test_indices]
mode_list = ['train','val','test']

# Traverse the existing directory structure
for n, mode in enumerate(all_mode):
    mode_state = mode_list[n]
    for indices in mode:
        patient = patient_list[indices]
        file_path = os.path.join(base_dir, patient)
        for case in os.listdir(file_path):
            if "cbct" in case.lower() and case.endswith(".nii.gz"):
                try:
                    nii_cbct = nib.load(os.path.join(file_path, 'cbct.nii.gz')).get_fdata()
                    center_z = nii_cbct.shape[2] // 2
                    slice_range = range(center_z -20, center_z+20)

                    for z in slice_range:
                        save_path = os.path.join(result_dir, mode_list[n],'A')
                        os.makedirs(save_path, exist_ok=True)
                        np.save(os.path.join(save_path, f'cbct_{patient}_{z}.npy'), nii_cbct[:, :, z])
                except Exception as e:
                    print(f"Error processing CBCT for patient {patient}: {e}")

            elif "ct" in case.lower() and case.endswith(".nii.gz"):
                try:
                    nii_ct = nib.load(os.path.join(file_path, 'ct.nii.gz')).get_fdata()
                    center_z = nii_ct.shape[2] // 2
                    slice_range = range(center_z -20, center_z+20)

                    for z in slice_range:
                        save_path = os.path.join(result_dir, mode_list[n],'B')
                        os.makedirs(save_path, exist_ok=True)
                        np.save(os.path.join(save_path, f'ct_{patient}_{z}.npy'), nii_ct[:, :, z])
                except Exception as e:
                    print(f"Error processing CT for patient {patient}: {e}")

            elif mode_state == 'test': # save mask only in test mode
                if "mask" in case.lower() and case.endswith(".nii.gz"):
                    try:
                        nii_mask = nib.load(os.path.join(file_path, case)).get_fdata()
                        center_z = nii_mask.shape[2] // 2
                        slice_range = range(center_z -20, center_z+20)

                        save_path = os.path.join(result_dir, mode_list[n], 'mask')
                        os.makedirs(save_path, exist_ok=True)

                        for z in slice_range:
                            np.save(os.path.join(save_path, f'mask_{patient}_{z}.npy'), nii_mask[:, :, z])

                    except Exception as e:
                        print(f"Error processing CT for patient {patient}: {e}")
#%% convert to png
import os
import shutil
import numpy as np
import nibabel as nib
import random
import PIL.Image as Image

base_dir = "data/Task2/brain" 

# Define the paths for new directories
result_dir = 'data/Task2/brain_data'

dir_save_train = os.path.join(result_dir, 'train')
dir_save_val = os.path.join(result_dir, 'val')
dir_save_test = os.path.join(result_dir, 'test')

# Create new directories if they don't exist
for dir_path in [dir_save_train, dir_save_val, dir_save_test]:
    os.makedirs(dir_path, exist_ok=True)

# all data
patient_list = os.listdir(base_dir)
patient_list.remove('overview')

# set train/val/test indices (0.8:0.1:0.1)
split_ratio = (0.8, 0.1, 0.1)
random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)

indices = np.arange(len(patient_list))
np.random.shuffle(indices)

train_size = int(len(indices) * split_ratio[0])
val_size = int(len(indices) * split_ratio[1])

train_indices = indices[:train_size]
val_indices = indices[train_size: (train_size + val_size)]
test_indices = indices[train_size + val_size:]

all_mode = [train_indices, val_indices, test_indices]
mode_list = ['train','val','test']

# Traverse the existing directory structure
for n, mode in enumerate(all_mode):
    for indices in mode:
        patient = patient_list[indices]
        file_path = os.path.join(base_dir, patient)
        
        # set max, min value for normalization (CBCT)
        if '2BA' in patient:
            max_cbct = 3000
            min_cbct = 0
        elif '2BB' in patient:
            max_cbct = 2000
            min_cbct = -1000
        elif '2BC' in patient:
            max_cbct = 3000
            min_cbct = -1024
        else: 
            print(f'error in {patient}')

        # set max, min value for normalization (CT)            
        max_ct = 3000
        min_ct = -1024
        
        for case in os.listdir(file_path):
            if "cbct" in case.lower() and case.endswith(".nii.gz"):
                try:
                    nii_cbct = nib.load(os.path.join(file_path, 'cbct.nii.gz')).get_fdata()
                    center_z = nii_cbct.shape[2] // 2
                    slice_range = range(center_z -20, center_z + 20)
                    
                    for z in slice_range:
                        save_path = os.path.join(result_dir, mode_list[n],'A')
                        os.makedirs(save_path, exist_ok=True)
                        image = (nii_cbct[:, :, z] - min_cbct) / (max_cbct - min_cbct)
                        img = (image * 256).astype(np.uint8)
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil.save(os.path.join(save_path, f'cbct_{patient}_{z}.png'))
    
                except Exception as e:
                    print(f"Error processing CBCT for patient {patient}: {e}")

            elif "ct" in case.lower() and case.endswith(".nii.gz"):
                try:
                    nii_ct = nib.load(os.path.join(file_path, 'ct.nii.gz')).get_fdata()
                    center_z = nii_ct.shape[2] // 2
                    for z in slice_range:
                        save_path = os.path.join(result_dir, mode_list[n],'B')
                        os.makedirs(save_path, exist_ok=True)
                        image = (nii_ct[:, :, z] - min_ct) / (max_cbct - min_ct)
                        img = (image * 256).astype(np.uint8)
                        img_pil = Image.fromarray(img.astype(np.uint8))
                        img_pil.save(os.path.join(save_path, f'ct_{patient}_{z}.png'))        
                                        
                except Exception as e:
                    print(f"Error processing CT for patient {patient}: {e}")


# %%
import PIL.Image as Image
import numpy as np

path_cbct = 'data/Task2/brain_data/train/A/cbct_2BA001_89.png'
path_ct = 'data/Task2/brain_data/train/B/ct_2BA001_89.png'

img_cbct = Image.open(path_cbct)
img_ct = Image.open(path_ct)

print(np.max(img_cbct), np.min(img_cbct))
print(np.max(img_ct), np.min(img_ct))

# %% mask convert to png
import os
import shutil
import numpy as np
import nibabel as nib
import random
import PIL.Image as Image

base_dir = "data/Task2/brain" 

# Define the paths for new directories
result_dir = 'data/Task2/brain_data'

dir_save_train = os.path.join(result_dir, 'train')
dir_save_val = os.path.join(result_dir, 'val')
dir_save_test = os.path.join(result_dir, 'test')

# Create new directories if they don't exist
for dir_path in [dir_save_train, dir_save_val, dir_save_test]:
    os.makedirs(dir_path, exist_ok=True)

# all data
patient_list = os.listdir(base_dir)
patient_list.remove('overview')

# set train/val/test indices (0.8:0.1:0.1)
split_ratio = (0.8, 0.1, 0.1)
random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)

indices = np.arange(len(patient_list))
np.random.shuffle(indices)

train_size = int(len(indices) * split_ratio[0])
val_size = int(len(indices) * split_ratio[1])

# train_indices = indices[:train_size]
# val_indices = indices[train_size: (train_size + val_size)]
test_indices = indices[train_size + val_size:]

# all_mode = [train_indices, val_indices, test_indices]
# mode_list = ['train','val','test']

for indices in test_indices:
    patient = patient_list[indices]
    file_path = os.path.join(base_dir, patient)
    
    for case in os.listdir(file_path):
            if "mask" in case.lower() and case.endswith(".nii.gz"):
                try:
                    nii_mask = nib.load(os.path.join(file_path, case)).get_fdata()
                    nii_mask = (nii_mask > 0).astype(np.uint8)  # 0 or 1
                    
                    center_z = nii_mask.shape[2] // 2
                    slice_range = range(center_z - 20, center_z + 20)
                    
                    save_path = os.path.join(result_dir, 'test', 'mask')
                    os.makedirs(save_path, exist_ok=True)
                    
                    for z in slice_range:
                        mask_slice = (nii_mask[:, :, z] * 255).astype(np.uint8) # 0 or 255
                        img_pil = Image.fromarray(mask_slice)
                        img_pil.save(os.path.join(save_path, f'mask_{patient}_{z}.png'))
                
                except Exception as e:
                    print(f"Error processing mask for patient {patient}: {e}")
                    
# %%
