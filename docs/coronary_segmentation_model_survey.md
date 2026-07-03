# Coronary Segmentation Model Survey

This file tracks candidate open-source repositories for coronary artery segmentation
or plaque analysis from CCTA.

## Candidate repositories

### 1. MM-DHM/nnUNet-Coronary-CTA-Segmentation

- Repository: https://github.com/MM-DHM/nnUNet-Coronary-CTA-Segmentation
- Role in OpenPlaque: first candidate for automated coronary segmentation.
- Notes: repository describes use of pretrained weights and nnU-Net-style transfer learning.
- OpenPlaque status: integration scaffold only.

### 2. MIC-DKFZ/nnUNet

- Repository: https://github.com/MIC-DKFZ/nnUNet
- Role in OpenPlaque: standard segmentation framework used by many biomedical imaging projects.
- Notes: useful framework for inference/training if coronary model weights are compatible.
- OpenPlaque status: framework dependency, not coronary-specific by itself.

### 3. qianjinmingliang/Coronary-Artery-segmentation-with-LCTUnet

- Repository: https://github.com/qianjinmingliang/Coronary-Artery-segmentation-with-LCTUnet
- Associated publication: “Automatic coronary artery segmentation of CCTA images using UNet with a local contextual transformer.”
- Role in OpenPlaque: alternative coronary artery segmentation model to evaluate.
- OpenPlaque status: integration scaffold only.

### 4. RoelvH97/FanCNN

- Repository: https://github.com/RoelvH97/FanCNN
- Associated publication: “Automatic Coronary Artery Plaque Quantification and CAD-RADS Prediction Using Mesh Priors.”
- Role in OpenPlaque: plaque/lumen segmentation after coronary centerlines are available.
- Notes: centerline inputs are a major prerequisite.
- OpenPlaque status: future integration target.

## Strategy

OpenPlaque will not vendor third-party code. Instead, notebooks will clone third-party
repositories into `/content/third_party` during Colab sessions and call their documented
inference pipelines.

## Immediate objective

1. Export Series 7 BestDiast CTA as NIfTI.
2. Clone candidate model repositories.
3. Determine which repository can run inference in Colab with available pretrained weights.
4. Visualize any generated coronary mask over the CTA.
5. Save masks to Google Drive for review in 3D Slicer.
