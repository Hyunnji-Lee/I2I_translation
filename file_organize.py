'''
데이터 정리 필요 
data/Task2/brain_data
                    /cbct
                    /ct
                    /mask
# data/Task2/3d_brain
#                     /cbct
#                     /ct
#                     /mask             

'''
#%% brain 데이터 정리

import os
import shutil
import numpy as np
import nibabel as nib

base_dir = "data/Task2/brain" 

# dir_save_train = os.path.join(base_dir, 'train')
# dir_save_val = os.path.join(base_dir, 'val')
# dir_save_test = os.path.join(base_dir, 'test')

# Define the paths for new directories
result_dir = 'data/Task2/brain_data'
cbct_dir = os.path.join(result_dir, "cbct")
ct_dir = os.path.join(result_dir, "ct")
mask_dir = os.path.join(result_dir, "mask")

# Create new directories if they don't exist
for dir_path in [cbct_dir, ct_dir, mask_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Traverse the existing directory structure
for patient in os.listdir(base_dir):
    case_list =  os.listdir(os.path.join(base_dir, patient))
    for case in case_list:
        file_path = os.path.join(base_dir, patient, case)
        if "cbct" in case.lower():
            nii_cbct= nib.load(os.path.join(base_dir, patient, 'cbct.nii.gz'))
            np.save(os.path.join(cbct_dir,f'{patient}.npy'), nii_cbct)
        elif "ct" in case.lower():
            nii_ct= nib.load(os.path.join(base_dir, patient, 'ct.nii.gz'))
            np.save(os.path.join(ct_dir,f'{patient}.npy'), nii_ct)        
        elif "mask" in case.lower():
            nii_mask= nib.load(os.path.join(base_dir, patient, 'mask.nii.gz'))
            np.save(os.path.join(mask_dir,f'{patient}.npy'), nii_mask)
    
print("Files have been reorganized successfully.")

#%% 
'''
brain 데이터 정리 - 2d(z축)
-> save .np
'''
import os
import shutil
import numpy as np
import nibabel as nib

base_dir = "data/Task2/brain" 

# Define the paths for new directories
result_dir = 'data/Task2/brain_data'
cbct_dir = os.path.join(result_dir, "cbct")
ct_dir = os.path.join(result_dir, "ct")
mask_dir = os.path.join(result_dir, "mask")

# Create new directories if they don't exist
for dir_path in [cbct_dir, ct_dir, mask_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Traverse the existing directory structure
for patient in os.listdir(base_dir):
    case_list =  os.listdir(os.path.join(base_dir, patient))
    for case in case_list:
        file_path = os.path.join(base_dir, patient, case)
        if "cbct" in case.lower():
            nii_cbct= nib.load(os.path.join(base_dir, patient, 'cbct.nii.gz'))
            np_cbct = nii_cbct.get_fdata()
            
            z_slices = np_cbct.shape[2]  # Z축 슬라이스 수
            for z in range(z_slices):
                slice_data = np_cbct[:, :, z]  # Z축 슬라이스
                save_path = os.path.join(cbct_dir,f'{patient}_{z:03d}.npy')
                np.save(save_path, slice_data)

        elif "ct" in case.lower():
            nii_ct= nib.load(os.path.join(base_dir, patient, 'ct.nii.gz'))
            np_ct = nii_ct.get_fdata()
            
            z_slices = np_ct.shape[2]  # Z축 슬라이스 수
            for z in range(z_slices):
                slice_data = np_ct[:, :, z]  # Z축 슬라이스
                save_path = os.path.join(ct_dir,f'{patient}_{z:03d}.npy')
                np.save(save_path, slice_data)
                
        elif "mask" in case.lower():
            nii_mask= nib.load(os.path.join(base_dir, patient, 'mask.nii.gz'))
            np_mask = nii_mask.get_fdata()

            z_slices = np_mask.shape[2]  # Z축 슬라이스 수
            for z in range(z_slices):
                slice_data = np_mask[:, :, z]  # Z축 슬라이스
                save_path = os.path.join(mask_dir,f'{patient}_{z:03d}.npy')
                np.save(save_path, slice_data)            
    
print("Files have been reorganized successfully.")

# %% 

