The authors of the **Dataset of Laryngeal Endoscopic Images for Semantic Segmentation** assessed existing segmentation methods. In medical image analysis, the automated segmentation of anatomical structures is crucial for autonomous diagnosis and various computer-aided interventions, including those involving robots. The examination of laryngeal endoscopic images holds the promise of early pathology detection. Given that vocal folds, the primary functional organ within the larynx, are delicate structures critical for surgery, computer vision can play a pivotal role in assisting physicians in preserving or restoring voice functionality. This synergy involves combining augmented reality, robotics, laser surgery, and image processing methods. Notably, the segmentation of laryngeal images stands out as a key component for the effective implementation of such a comprehensive system.

## Dataset description

The dataset contain 536 manually segmented color images of the larynx during two different resection surgeries with a resolution of 512×512 pixels. The images have been captured with a stereo endoscope (VSii, Visionsense, Petach-Tikva, Israel). They are categorized in the 7 different classes: _void_, _vocal folds_, _other tissue_, _glottal space_, _pathology_, _surgical tool_. 

<img src="https://github.com/dataset-ninja/vocalfolds/assets/120389559/0a2946aa-2d85-4c66-8610-1cc3aeae374b" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">First row: Examples from vocal folds dataset. Second row: Manually segmented ground truth label maps with classes vocal folds (red), other tissue (blue), glottal space (green), pathology (purple), surgical tool (orange), intubation (yellow) and void (gray).</span>

The dataset consists of 5 different sequences from two patients. The sequences have following characteristics:

* ***SEQ1***: pre-operative with clearly visible tumor on vocal fold, changes in translation, rotation, scale, no instruments visible, without intubation
* ***SEQ2***: pre-operative with clearly visible tumor, visible instruments, changes in translation and scale, with intubation
* ***SEQ3–4***: post-operative with removed tumor, damaged tissue, changes in translation and scale, with intubation
* ***SEQ5–7***: pre-operative with instruments manipulating and grasping the vocal folds, changes in translation and scale, with intubation
* ***SEQ8***: post-operative with blood on vocal folds, instruments and surgical dressing, with intubation

Subsequent images have a temporal contiguity as they are sampled uniformly from videos. To reduce inter-frame correlation, images were extracted from the original videos only once per second. In the comparative study SEQ4–SEQ6 were not used due to high similarity to SEQ3 and SEQ7 respectively, as they do not offer any additional variance to the dataset. Segmentations have been manually created on a pen display (DTK-2241, K. K. Wacom).

<img src="https://github.com/dataset-ninja/vocalfolds/assets/120389559/237e8a74-61f7-453c-a7b6-a64bb9c9610c" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Number of annotated pixels per class in the dataset.</span>