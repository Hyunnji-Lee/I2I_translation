#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
from skimage.transform import resize
import PIL.Image as Image
import numpy as np

class ImageMetrics():
    def __init__(self):
        # Use fixed wide dynamic range
        self.dynamic_range = [-1024., 3000.]

    def score_patient(self, ground_truth_path, predicted_path, mask_path):
        # Load images(png)
#         gt_array = np.array(Image.open(ground_truth_path).convert("L")) #gray scale image
#         pred_array = np.array(Image.open(predicted_path).convert("L"))
#         mask_array = np.array(Image.open(mask_path).convert("L"))
        
        # Load images(npy)
        gt_array = np.load(ground_truth_path)
        pred_array = np.load(predicted_path)
        mask_array = np.load(mask_path)
        
        if gt_array.ndim == 3:
            gt_array = gt_array[:,:,0]
        if pred_array.ndim == 3:
            pred_array = pred_array[:,:,0]
        if mask_array.ndim == 3:
            mask_array = mask_array[:,:,0]

        # confirm mask shape
        if mask_array.shape != gt_array.shape:
            mask_array = resize(mask_array, gt_array.shape)
                
        # Calculate image metrics
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)

        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)

        ssim_value = self.ssim(gt_array,
                               pred_array,
                               mask_array)

#         cnr_value = self.cnr(pred_array,
#                              mask_array)

        return {
            'mae': mae_value,
            'ssim': ssim_value,
            'psnr': psnr_value
        }

    def mae(self,
            gt: np.ndarray,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        mae : float
            mean absolute error.

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0.5, 1., 0.)

        mae_value = np.sum(np.abs(gt*mask - pred*mask))/mask.sum()
        return float(mae_value)


    def psnr(self,
             gt: np.ndarray,
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.

        Returns
        -------
        psnr : float
            Peak signal to noise ratio..

        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0.5, 1., 0.)

        if use_population_range:
            dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]

            # Clip gt and pred to the dynamic range
            gt = np.where(gt < self.dynamic_range[0], self.dynamic_range[0], gt)
            gt = np.where(gt > self.dynamic_range[1], self.dynamic_range[1], gt)
            pred = np.where(pred < self.dynamic_range[0], self.dynamic_range[0], pred)
            pred = np.where(pred > self.dynamic_range[1], self.dynamic_range[1], pred)
        else:
            dynamic_range = gt.max()-gt.min()

        # apply mask
        gt = gt[mask==1]
        pred = pred[mask==1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)


    def ssim(self,
              gt: np.ndarray,
              pred: np.ndarray,
              mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index Metric (SSIM)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        ssim : float
            structural similarity index metric.

        """
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            #binarize mask
            mask = np.where(mask>0.5, 1., 0.)

            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return ssim_value_masked
        else:
            return ssim_value_full



    def cnr(self,
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Contrast-to-Noise Ratio (CNR)

        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).

        Returns
        -------
        cnr : float
            Contrast-to-noise ratio.
            absolute difference of expectation of two signals / sqrt(sum of square of standard deviation of two signals)
        """
        if mask is None:
            mask = np.ones(pred.shape)
        else:
            # binarize mask
            mask = np.where(mask > 0, 1., 0.)

        pred_masked = pred[mask == 1]
        pred_unmasked = pred[mask == 0]

        # Compute the signal mean
        mean_masked = np.mean(pred_masked)
        mean_unmasked = np.mean(pred_unmasked)

        # Compute the standard deviation
        sigma_masked = np.std(pred_masked)
        siama_unmasked = np.std(pred_unmasked)
        sigma = np.sqrt(sigma_masked**2 + siama_unmasked**2)

        # Calculate CNR
        cnr_value = np.abs(mean_masked-mean_unmasked) / sigma
        return float(cnr_value)