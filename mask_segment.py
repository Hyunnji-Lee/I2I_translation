'''
To make mask to fit the CBCT image only test dataset
'''
#%% Make segmentation mask to fit the CBCT image
import os
import shutil
import random

from dataset.pre_process_tools import segment_small, segment_fit

base_dir = "data/Task2/brain" 
mask_dir = "data/Task2/brain_mask_test"
case_list = [case for case in os.listdir(base_dir) if case.startswith('2B')]

random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)

indices = np.arange(len(case_list))
np.random.shuffle(indices)

test_size = int(len(indices) * 0.1)

test_indices = indices[-test_size:]

case_list_test = [case_list[i] for i in test_indices]

print(len(case_list_test))
#%%
if os.path.exists(mask_dir):
    # os.remove(mask_dir)
    shutil.rmtree(mask_dir)
os.makedirs(mask_dir, exist_ok=True)

for case in case_list_test:
    try:
        segment_small(os.path.join(base_dir, case,'cbct.nii.gz'), 
                os.path.join(mask_dir,f'{case}_mask_CBCT.nii.gz'))
    except Exception as e:
        print(f"Error in {case}")
        print(e)

print(len(os.listdir(mask_dir)))
#%% Save mask slices only test dataset
npy_dir = "data/Task2/brain_data_npy_sub"
case_name = []

name = os.listdir(os.path.join(npy_dir, 'test','A'))
for j in name:
    case_name.append(j[5:-4])
case_set = set([case[:6] for case in case_name])

#%%
import nibabel as nib
import numpy as np

mask_list = os.listdir(mask_dir)
            
if os.path.exists(os.path.join(npy_dir, 'test','mask_fit')):
    # os.remove(mask_dir)
    shutil.rmtree(os.path.join(npy_dir, 'test','mask_fit'))
os.makedirs(os.path.join(npy_dir, 'test','mask_fit'), exist_ok=True)

error = []
for mask in mask_list:
    if mask[:6] in case_set:
        print(mask[:6])
        mask_nii = nib.load(os.path.join('data/Task2/brain_mask_test', mask))
        mask_data = mask_nii.get_fdata()
        mask_z = mask_data.shape[2] //2
        mask_range = range(mask_z-20, mask_z+20)
        
        for z in mask_range:
            mask_slice = mask_data[:,:,z]
            np.save(os.path.join(npy_dir, 'test','mask_fit',f'mask_{mask[:6]}_{z}.npy'), mask_slice)
    else:
        error.append(mask[:6])
        
print(len(os.listdir(os.path.join(npy_dir, 'test','mask_fit'))))

#%% plot
import matplotlib.pyplot as plt
import os
import numpy as np

cbct_dir = "data/Task2/brain_data_npy_sub/test/A"
mask_fit_dir = "data/Task2/brain_data_npy_sub/test/mask_fit"

cbct = sorted(os.listdir(cbct_dir))
mask = sorted(os.listdir(mask_fit_dir))

nn = 0
for cc, mm in zip(cbct, mask):
    nn +=1

    np_cbct = np.load(os.path.join(cbct_dir, cc))
    np_mask = np.load(os.path.join(mask_fit_dir, mm))
    # print(np_cbct.shape, np_mask.shape)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np_cbct, cmap='gray')
    plt.title(cc)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(np_mask, cmap='gray')
    plt.title(mm)
    plt.axis('off')
    plt.show()
    if nn>30:
        break
# %%
