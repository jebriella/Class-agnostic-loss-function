import numpy as np
import os
import shutil

def my_vote(patient, mask, data, models, path):
    votes = np.zeros(len(models))

    for i in range(len(models)):
        votes[i] = models[i].predict([data])
    label = 2
    if np.sum(votes) == len(models):
        label = 1
    elif np.sum(votes) == 0:
        label = 0

    a = np.sum(mask, axis = 2)
    a = np.sum(a, axis = 1)
    la = a.shape[0]
    a1 = np.sum(a[0:int(la/2)])
    a2 = np.sum(a[int(la/2):la])
    if a2 > a1:
        dir = 0
    else:
        dir = 1

    if label == 2:
        f = open("samles.txt","a")
        f.write(str(patient) + ":")
        f.write(" %d, " % votes[0])
        f.write(" %d, " % votes[1])
        f.write(" %d, " % votes[2])
        f.write(" %d, " % votes[3])
        f.write(" %d, " % votes[4])
        if dir == 0:
            f.write(" left, \r\n")
        else:
            f.write(" right, \r\n")
        f.close()
    elif dir == 0 and label == 0:
        old_path = os.path.join(path, str(patient))
        new_path = "C:/Sorted_data/LEFT"
        dest = shutil.move(old_path, new_path)
    elif dir == 0 and label == 1:
        old_path = os.path.join(path, str(patient))
        new_path = "C:/Sorted_data/LEFT+AX"
        dest = shutil.move(old_path, new_path)
    elif dir == 1 and label == 0:
        old_path = os.path.join(path, str(patient))
        new_path = "C:/Sorted_data/RIGHT"
        dest = shutil.move(old_path, new_path)
    elif dir == 1 and label == 1:
        old_path = os.path.join(path, str(patient))
        new_path = "C:/Sorted_data/RIGHT+AX"
        dest = shutil.move(old_path, new_path)
