# Class-agnostic loss function
Repository for Gabriella Norman's master thesis project: 

**A segmentation network with a class-agnostic loss function to train on incomplete data**

In repository:
- Predictions: Gif:s of ground truths and predictions from both baseline models and the class-agnostic model
- Preprocessing
  - preprocessing_class-agnostic_data: Reads NIFTI files (images and masks), performs preprocessing and saves slices and corresponding mask-slices as numpy files
  - preprocessing_one_class_data: Reads NIFTI files (images and masks), performs preprocessing and saves slices and corresponding mask-slice as numpy files. This is an example for the heart mask.
- Sorting
  - data_loader
  - feature_setup
  - my_vote
  - main_sort

# Predictions 

## Breasts

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/breast.gif)

### Mislabeled pixels

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/breast_error.gif)

## Breasts including lymph nodes

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/breast_ax.gif)

### Mislabeled pixels

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/breast_ax_error.gif)

## Lungs

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/lungs.gif)

### Mislabeled pixels

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/lungs_error.gif)

## Heart

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/heart.gif)

### Mislabeled pixels

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/heart_error.gif)

## Spinal cord

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/spinalcord.gif)

### Mislabeled pixels

![](https://github.com/jebriella/Class-agnostic-loss-function/blob/master/Predictions/spinalcord_error.gif)
