import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
import os
import scipy.io as sio
import pandas as pd

sys.path.append('/gpfs/scratch/pa2297/CCL-Synthetis/')

# Assuming utils and Datagen are your local modules or packages
from utils.model_utils import modelObj
from Datagen.h5_pretrain_Synth_Data_Generator import DataLoaderObj

import Synthesis.synth_config as cfg

#-----------------------------------------------------------------
# Load model
model_ = modelObj(cfg)

# Load the trained weights
weights = {
    'Baseline': 
    '/gpfs/scratch/pa2297/Training/t1-t2-flair/ZmeanOff/brats_t1_t2_flair_ReLUloss1/rand_deform/finetune_Hybride _baseline/t1_t2_flair_ReLUloss1_tr1_Hybride/tr_comb_Exp1_LR_ft_0.01/weights_20Final.hdf5',
    'Full_Decoder': '/gpfs/scratch/pa2297/Training/t1-t2-flair/ZmeanOff/brats_t1_t2_flair_ReLUloss1/rand_deform/finetune_Hybride _CL_full_dec_warm/t1_t2_flair_ReLUloss1_tr1_Hybride/tr_comb_Exp1_LR_ft_0.01/weights_20Final.hdf5',
    'Partial_Decoder': '/gpfs/scratch/pa2297/Training/t1-t2-flair/ZmeanOff/brats_t1_t2_flair_ReLUloss1/rand_deform/finetune_Hybride _CL_partial_dec_warm/t1_t2_flair_ReLUloss1_tr1_Hybride/tr_comb_Exp1_LR_ft_0.01/weights_20Final.hdf5'
}

models = {
    'Baseline': model_.synth_unet(act_name='relu'),
    'Full_Decoder': model_.synth_unet(act_name='relu'),
    'Partial_Decoder': model_.synth_unet(act_name='relu')
}

sys.path.append('/gpfs/scratch/pa2297/multi-contrast-contrastive-learning/')

from utils.utils import myCrop3D
from utils.utils import contrastStretch

def normalize_img_zmean(img, mask):
    ''' Zero mean unit standard deviation normalization based on a mask'''
    mask_signal = img[mask>0]
    mean_ = mask_signal.mean()
    std_ = mask_signal.std()
    img = (img - mean_ )/ std_
    return img, mean_, std_

def normalize_img(img):
    img = (img - img.min())/(img.max()-img.min())
    return img

def load_subject(datadir, subName):
    data_suffix = ['_t1ce.nii.gz', '_t2.nii.gz', '_t1.nii.gz', '_flair.nii.gz']
    sub_img = []
    mask = None
    for suffix in data_suffix:
        img_path = f"{datadir}{subName}/{subName}{suffix}"
        img_data = nib.load(img_path).get_fdata()
        img_data = np.rot90(img_data, -1)
        img_data = myCrop3D(img_data, (192,192))

        if mask is None:  
            mask = np.zeros(img_data.shape)
            mask[img_data > 0] = 1
        
        img_data = contrastStretch(img_data, mask, 0.01, 99.9)
        img_data, mean_, std_ = normalize_img_zmean(img_data, mask)
        # img_data = normalize_img(img_data)
        sub_img.append(img_data)
    
    sub_img = np.stack(sub_img, axis=-1)
    sub_img = np.transpose(sub_img, (2,0,1,3))  # Adjust dimensions as needed
    sub_img = sub_img[40:120]  # Assuming your volume z-axis slice range
    

    return sub_img

#-----------------------------------------------------------------

def get_data(img, contrast_idx, target_contrast_idx):
    """Returns tuple (input, target) correspond to sample #idx."""
    x_train = generate_X(img, contrast_idx)
    y_train = generate_Y(img, target_contrast_idx)
    return tf.identity(x_train), tf.identity(y_train)
        
def generate_X(img, contrast_idx):    
    X = img[..., contrast_idx]
    return X
    
def generate_Y(img, target_contrast_idx):    
    Y = img[..., target_contrast_idx] 
    return Y

#-----------------------------------------------------------------
# Load weights
for key, model in models.items():
    model.load_weights(weights[key])

datadir = "/gpfs/scratch/pa2297/Dataset/BraTS2021_Test/"
subjects = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f)) and f.startswith("BraTS2021_")]

# Path to save predictions
output_dir = "/gpfs/scratch/pa2297/CCL-Predictions/relu-zmeanoff/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a function to process all subjects
def process_all_subjects(datadir, subjects, models, cfg):
    predictions = {}
    for subName in subjects:
        img = load_subject(datadir, subName)
        x_train, _ = get_data(img, cfg.contrast_idx, cfg.target_contrast_idx)
        
        subject_predictions = {}
        for model_name, model in models.items():
            subject_predictions[model_name] = model.predict(x_train)
        
        predictions[subName] = subject_predictions
        print(f"Processed {subName}")
        
    # Save each subject's predictions as a separate .mat file
        filename = f"{output_dir}{subName}_predictions.mat"
        sio.savemat(filename, {'predictions': subject_predictions})
        print(f"Predictions for {subName} saved to {filename}")

process_all_subjects(datadir, subjects, models, cfg)