# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
from nibabel.spatialimages import SpatialImage
import cv2

ROOT_DIR = os.path.abspath(os.curdir)


def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s

    return ct_scan



def load_ct_mask(datasetDir, sub_n, window_specs):
    ct_dir_subj = Path(datasetDir, 'ct_scans', "{0:0=3d}.nii".format(int(
        sub_n)))
    ct_scan_nifti: SpatialImage = nib.load(str(ct_dir_subj))
    ct_scan = np.asanyarray(ct_scan_nifti.dataobj)
    ct_scan = window_ct(ct_scan, window_specs[0],
                        window_specs[1])  #
    # Convert the CT scans using a brain window
    # Loading the masks
    masks_dir_subj = Path(datasetDir, 'masks', "{0:0=3d}.nii".format(sub_n))
    masks_nifti = nib.load(str(masks_dir_subj))
    masks = np.asanyarray(masks_nifti.dataobj)
    return ct_scan, masks


if __name__ == '__main__':
    numSubj = 82
    new_size = (512, 512)
    window_specs=[40,120] #Brain window
    currentDir = os.path.dirname(os.path.dirname(os.getcwd()))
    datasetDir = currentDir / Path('data/brain-hemorrhage')

    # Reading labels
    hemorrhage_diagnosis_df = pd.read_csv(
        Path(datasetDir, 'hemorrhage_diagnosis_raw_ct.csv'))
    hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values

    # reading images
    train_path = Path(datasetDir, 'train')
    image_path = train_path / 'images'
    label_path = train_path / 'masks'
    if not train_path.exists():
        train_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()

    counterI = 0
    for sNo in range(0+49, numSubj+49):
        if sNo>58 and sNo<66: #no raw data were available for these subjects
            next
        else:
            # Loading the CT scan and masks
            ct_scan, masks = load_ct_mask(datasetDir, sNo, window_specs)

            ct_dir_subj = Path(datasetDir,'ct_scans', "{0:0=3d}.nii".format(sNo))

            idx = hemorrhage_diagnosis_array[:, 0] == sNo
            sliceNos = hemorrhage_diagnosis_array[idx, 1]
            NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]
            if sliceNos.size!=ct_scan.shape[2]:
                print('Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

            for sliceI in range(0, sliceNos.size):
                # Saving the a given CT slice
                ct_scan_slice = ct_scan[:, :, sliceI]
                x = Image.fromarray(ct_scan_slice).resize(new_size)
                x.convert("L").save(image_path / (str(counterI) + '.png'))

                # Saving the mask for a given slice
                mask_slice = cv2.resize(masks[:, :, sliceI], new_size)
                x = Image.fromarray(mask_slice)
                x.convert("L").save(label_path / (str(counterI) + '.png'))
                counterI = counterI+1