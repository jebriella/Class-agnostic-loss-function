import os
import numpy as np
import nibabel as nib
from skimage.transform import resize


path = "D:/folder"

patients = os.listdir(path)
dir_nr_patients = len(patients)

p = 0

for i in range(dir_nr_patients):

    patient_path = os.path.join(path, patients[i])
    directory = os.listdir(patient_path)

    if directory.count('image.nii') > 0 and directory.count('heart.nii') > 0:
        p = p + 1

        img = nib.load(os.path.join(patient_path, "image.nii"))
        img = np.array(img.dataobj)
        img = np.swapaxes(img,0,2)
        img[img < -1000] = -1000
        img = (img + 1000)/(3000)

        mask = nib.load(os.path.join(patient_path, "heart.nii"))
        mask = np.array(mask.dataobj)
        mask = np.swapaxes(mask,0,2)

        for j in range(img.shape[0]):
            img_slice = img[j,:,:]
            img_r = resize(img_slice, (256,256), preserve_range=True)
            x = np.zeros((256, 256, 1), dtype=np.float16)
            x[:,:,0] = img_r

            y = mask[j,:,:].astype(bool)
            y = resize(y, (256,256), preserve_range=True)
            y = y.astype(np.float16)

            Yy = np.zeros((256, 256, 1), dtype=np.float16)
            Yy[:,:,0] = y

            np.save("D:/full_set_patient/1p_left_lung/images/" + patients[i] + "_" + str(j), x)
            np.save("D:/full_set_patient/1p_left_lung/masks/" + patients[i] + "_" + str(j), Yy)

print("Patients:")
print(p)
