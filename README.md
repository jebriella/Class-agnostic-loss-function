# Class-agnostic loss function
Repository for Gabriella Norman's master thesis project: 

**A segmentation network with a class-agnostic loss function to train on incomplete data**

In repository:
- Sorting
  - **data_loader**: A function that loads CT-image and breast mask of all patients. Returns one list with patient ID:s, one list with CT images and one list breast masks
  - **feature_setup**: A function that extracts the 15 features used for sorting from the given image and mask. 
  - **my_vote**: 
  - **main_sort**: 
- Preprocessing
  - **preprocessing_class-agnostic_data**: Reads NIFTI files (images and masks), performs preprocessing and saves slices and corresponding mask-slices as numpy files
  - **preprocessing_one_class_data**: Reads NIFTI files (images and masks), performs preprocessing and saves slices and corresponding mask-slice as numpy files. This is an example for the heart mask.
- Deep learning implementations
  - **U_Net_2D**: An implementation of the U-Net for 2D data, with options for batch normalization and spatial dropout
  - **metric_loss**: Metrics for dice coefficient. Dice loss, cross-entropy and a combined loss function of dice loss and cross-entropy
  - **Class_agnostic_loss**: The class-agnostic loss function for an eight-mask model
- Predictions: Gif:s of ground truths and predictions from both baseline models and the class-agnostic model

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
