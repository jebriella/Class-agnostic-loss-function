import os
import numpy as np
import nibabel as nib

def data_loader(path):
    patients = os.listdir(path)

    nr_patients = len(patients)

    images = []
    breast = []

    for i in range(nr_patients):
        patient_path = os.path.join(path, patients[i])
        # Load image
        image_path = os.path.join(patient_path, "image.nii")
        images.append(read_file(image_path))
        # Load breast mask
        breast_path = os.path.join(patient_path, "breast.nii")
        breast.append(read_file(breast_path))

    return patients, images, breast

def read_file(path):
    seg = nib.load(path)
    array = np.array(seg.dataobj)
    return array
