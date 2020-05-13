import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

path = "D:/folder"

patients = os.listdir(path)
dir_nr_patients = len(patients)
print(dir_nr_patients)

count_lb = 0
count_rb = 0
count_la = 0
count_ra = 0
count_ll = 0
count_rl = 0
count_h = 0
count_s = 0

for i in range(dir_nr_patients):
    print(i)
    print(patients[i])

    patient_path = os.path.join(path, patients[i])
    directory = os.listdir(patient_path)

    img = nib.load(os.path.join(patient_path, "image.nii"))
    img = np.array(img.dataobj)
    img = np.swapaxes(img,0,2)
    img[img < -1000] = -1000
    img = (img + 1000)/(3000)

    _lb = False
    _rb = False
    _la = False
    _ra = False
    _ll = False
    _rl = False
    _h = False
    _s = False

    if directory.count('left_brest.nii') > 0:
        l_brest = nib.load(os.path.join(patient_path, "left_breast.nii"))
        l_brest = np.array(l_brest.dataobj)
        l_brest = np.swapaxes(l_brest,0,2)
        _lb = True
    if directory.count('left_brest_ax.nii') > 0:
        l_ax = nib.load(os.path.join(patient_path, "left_breast_ax.nii"))
        l_ax = np.array(l_ax.dataobj)
        l_ax = np.swapaxes(l_ax,0,2)
        _la = True
    if directory.count('right_brest.nii') > 0:
        r_brest = nib.load(os.path.join(patient_path, "right_breast.nii"))
        r_brest = np.array(r_brest.dataobj)
        r_brest = np.swapaxes(r_brest,0,2)
        _rb = True
    if directory.count('right_brest_ax.nii') > 0:
        r_ax = nib.load(os.path.join(patient_path, "right_breast_ax.nii"))
        r_ax = np.array(r_ax.dataobj)
        r_ax = np.swapaxes(r_ax,0,2)
        _ra = True

    if directory.count('left_lung.nii') > 0:
        left_lung = nib.load(os.path.join(patient_path, "left_lung.nii"))
        left_lung = np.array(left_lung.dataobj)
        left_lung = np.swapaxes(left_lung,0,2)
        _ll = True

    if directory.count('right_lung.nii') > 0:
        right_lung = nib.load(os.path.join(patient_path, "right_lung.nii"))
        right_lung = np.array(right_lung.dataobj)
        right_lung = np.swapaxes(right_lung,0,2)
        _rl = True

    if directory.count('heart.nii') > 0:
        heart = nib.load(os.path.join(patient_path, "heart.nii"))
        heart = np.array(heart.dataobj)
        heart = np.swapaxes(heart,0,2)
        _h = True

    if directory.count('spinalcord.nii') > 0:
        spinalcord = nib.load(os.path.join(patient_path, "spinalcord.nii"))
        spinalcord = np.array(spinalcord.dataobj)
        spinalcord = np.swapaxes(spinalcord,0,2)
        _s = True

    for j in range(img.shape[0]):
        img_slice = img[j,:,:]
        img_r = resize(img_slice, (256,256), preserve_range=True)
        x = np.zeros((256, 256, 1), dtype=np.float16)
        x[:,:,0] = img_r

        if _lb:
            l_b = l_brest[j,:,:].astype(bool)
            l_b = resize(l_b, (256,256), preserve_range=True)
            l_b = l_b.astype(np.float16)
            l_b[0,0] = 1
            count_lb = count_lb + 1
        else:
            l_b = np.zeros((256,256), dtype=np.float16)

        if _la:
            l_a = l_ax[j,:,:].astype(bool)
            l_a = resize(l_a, (256,256), preserve_range=True)
            l_a = l_a.astype(np.float16)
            l_a[0,0] = 1
            count_la = count_la + 1
        else:
            l_a = np.zeros((256,256), dtype=np.float16)

        if _rb:
            r_b = r_brest[j,:,:].astype(bool)
            r_b = resize(r_b, (256,256), preserve_range=True)
            r_b = r_b.astype(np.float16)
            r_b[0,0] = 1
            count_rb = count_rb + 1
        else:
            r_b = np.zeros((256,256), dtype=np.float16)

        if _ra:
            r_a = r_ax[j,:,:].astype(bool)
            r_a = resize(r_a, (256,256), preserve_range=True)
            r_a = r_a.astype(np.float16)
            r_a[0,0] = 1
            count_ra = count_ra + 1
        else:
            r_a = np.zeros((256,256), dtype=np.float16)

        if _ll:
            l_l = left_lung[j,:,:].astype(bool)
            l_l = resize(l_l, (256,256), preserve_range=True)
            l_l = l_l.astype(np.float16)
            l_l[0,0] = 1
            count_ll = count_ll + 1
        else:
            l_l = np.zeros((256,256), dtype=np.float16)

        if _rl:
            r_l = right_lung[j,:,:].astype(bool)
            r_l = resize(r_l, (256,256), preserve_range=True)
            r_l = r_l.astype(np.float16)
            r_l[0,0] = 1
            count_rl = count_rl + 1
        else:
            r_l = np.zeros((256,256), dtype=np.float16)

        if _h:
            h = heart[j,:,:].astype(bool)
            h = resize(h, (256,256), preserve_range=True)
            h = h.astype(np.float16)
            h[0,0] = 1
            count_h = count_h + 1
        else:
            h = np.zeros((256,256), dtype=np.float16)

        if _s:
            s = spinalcord[j,:,:].astype(bool)
            s = resize(s, (256,256), preserve_range=True)
            s = s.astype(np.float16)
            s[0,0] = 1
            count_s = count_s + 1
        else:
            s = np.zeros((256,256), dtype=np.float16)

        y = np.zeros((256, 256, 8), dtype=np.float16)

        y[:,:,0] = l_b
        y[:,:,1] = r_b
        y[:,:,2] = l_a
        y[:,:,3] = r_a
        y[:,:,4] = l_l
        y[:,:,5] = r_l
        y[:,:,6] = h
        y[:,:,7] = s

        #np.save("D:/Multi_testdata/images/" + patients[i] + "_" + str(j), x)
        #np.save("D:/Multi_testdata/masks/" + patients[i] + "_" + str(j), y)

        np.save("D:/full_set_patient/multi/images/" + "de16c56f986_" + str(j), x)
        np.save("D:/full_set_patient/multi/masks/" + "de16c56f986_" + str(j), y)

print("Left breast:")
print(count_lb)
print("Right breast:")
print(count_rb)
print("Left ax:")
print(count_la)
print("Right ax:")
print(count_ra)
print("Left lung:")
print(count_ll)
print("Right lung:")
print(count_rl)
print("Heart")
print(count_h)
print("Spinalcord")
print(count_s)
