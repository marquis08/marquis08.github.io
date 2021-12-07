---
date: 2021-12-08 02:23
title: "Heart Datathon 2021"
categories: Competition Medical Segmentation
tags: Competition Medical Segmentation Ultrasound
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Detection Metrics
## Dice Coefficient
- $$Dice = \frac{2 |A∩B|}{(|A|+|B|)}  = \frac{2 TP}{(2 TP + FP + FN)}$$

## Jaccard Index and IoU
- $$JI = \frac{DICE}{2-DICE} = \frac{|A \cap B|}{|A|+|B|-|A\cap B|}$$

# Review
## EDA
- Train: 1600 (A2C: 800, A4C: 800)
- Validation: 200 (A2C: 100, A4C: 100)
- Too small data
  - Sensitive to apply augmentation
- Ultrasound Image
  - Masks are not strictly following the boundaries of the object.

## Worked
- Dehaze Pre-Processing
- CRF Post-Processing
- ReduceLROnPlateau
- HRnet(not used)
  - Limited with model size rule.
  - HRNet + OCR $\approx$ HRNet Unet
- Ensemble(mean)

## Not Worked
- more augmentation
  - Hflip, Cutout, CenterCrop
- AdamP
- CosineAnnealingLR

## Losses
- Used
  - Tversky
  - DiceLoss
  - LogCosh
  - BCELoss
  - BCE_DICE_Combo (final)
- Still doubt about selecting which loss to use.


# Appendix
## Reference
> Competition info: <http://www.hdaidatathon.com/>  
> 
> Competition github: <https://github.com/DatathonInfo/H.D.A.I.2021>  
> 
> similar competition: <https://www.kaggle.com/c/understanding_cloud_organization/discussion/114093>  
> 
> pytorch-goodies: <https://github.com/kevinzakka/pytorch-goodies>  
> 
> Loss Function Library - Keras & PyTorch: <https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch>  
> 
> torchmetrics: <https://torchmetrics.readthedocs.io/en/latest/references/modules.html?highlight=iou#iou>  
> 
> Medical_Img_Seg_and_Enhancement: <https://github.com/GeekyGeek3371/Medical_Img_Seg_and_Enhancement>  
> 
> Kaggle Ultrasound Nerve Segmentation: <http://fhtagn.net/prog/2016/08/19/kaggle-uns.html>
> 
> semantic segmentation Loss Survey: <https://arxiv.org/pdf/2006.14822.pdf>
> 
> Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography: <https://arxiv.org/pdf/1908.06948.pdf>
> 
> SegLoss: <https://github.com/JunMa11/SegLoss>
> 
> CRF: <https://github.com/lucasb-eyer/pydensecrf>