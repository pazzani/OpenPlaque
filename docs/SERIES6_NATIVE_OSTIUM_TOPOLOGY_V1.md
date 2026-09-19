# Series 6 Native Ostium Topology v1

This experiment follows the negative/inconclusive Series 6 registered-path experiment.

The prior Series 6 run passed root registration but failed the RCA path-support control, so the left-origin question could not be interpreted. This experiment therefore avoids using registered Series-7 centerlines as the primary test.

## Primary question

In **native Series 6 geometry**, how many distinct CAS-Net coronary/aorta contact clusters exist?

## Inputs

- Series 6 BestSyst 32% native CCTA
- cached Series 6 CAS-Net prediction from the prior completed experiment
- a new **native Series 6 aorta segmentation** generated from Series 6 itself using TotalSegmentator `total --roi_subset aorta`
- frozen Series-7 anatomy only for local RCA labeling and descriptive association

## Primary gates

- root-region translation <= 6 mm, used only to label the expected RCA neighborhood
- native aorta volume 10,000–500,000 mm3
- contact defined as CAS-Net coronary voxels <=1.5 mm from the native aorta mask
- contact cluster >=10 voxels
- RCA control = qualifying contact centroid within 5 mm of the locally transformed frozen RCA root
- second candidate = another qualifying contact >=8 mm from the RCA contact centroid

## Statuses

- `SERIES6_NATIVE_SECOND_AORTIC_CONTACT_PRESENT`
- `SERIES6_NATIVE_SINGLE_CORONARY_AORTIC_CONTACT`
- `SERIES6_NATIVE_RCA_CONTROL_FAILED`
- `SERIES6_NATIVE_AORTA_SEGMENTATION_FAILED`
- `SERIES6_NATIVE_ROOT_REGISTRATION_FAILED`

A second contact is a **candidate ostium only**. It does not automatically establish clinical LM/LCX/OM identity.

The frozen master is never modified.
