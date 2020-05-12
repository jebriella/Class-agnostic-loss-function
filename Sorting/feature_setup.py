import numpy as np
from numpy import linalg as LA
from random import shuffle
from scipy.ndimage.filters import uniform_filter1d
from scipy.fftpack import fft
from scipy.signal import find_peaks

def feature_setup(image, mask, subt, div):

    data = np.zeros(15)
    z_ref = np.array([1, 0])

    at = np.sum(mask, axis = 2)
    at = np.sum(at, axis = 1)
    data[6] = np.mean(at) # X dence
    at = np.nonzero(at)
    data[0] = np.amax(at) - np.amin(at)

    # Y-length
    at = np.sum(mask, axis = 2)
    at = np.sum(at, axis = 0)
    data[7] = np.mean(at) # Y dence
    at = np.nonzero(at)
    data[1] = np.amax(at) - np.amin(at)

    # Z-length
    at = np.sum(mask, axis = 1)
    at = np.sum(at, axis = 0)
    data[8] = np.mean(at) # Z dence
    att = np.nonzero(at)
    data[2] = np.amax(att) - np.amin(att)
    # Derivata
    p = at[at != 0]
    grad = np.gradient(p)
    grad = uniform_filter1d(grad, size=6)
    neg = grad[grad < 0]
    neg_p = np.where(grad == neg[0])
    data[9] = 1
    for j in range(int(neg_p[0][0]), int(len(grad))):
        if grad[j] > 0:
            data[9] = 0
            break

    # Angle z
    ww = np.sum(mask, axis = 0)
    ww = np.rot90(ww)
    w = np.nonzero(ww)
    w_min = np.amin(w)
    w_max = np.amax(w)
    w_norm = (w - w_min)/(w_max - w_min)
    cov = np.cov(w_norm)
    u, s, vh = LA.svd(cov)
    data[3] = np.rad2deg(np.arccos((np.dot(z_ref, u[:,0])/(LA.norm(z_ref)*LA.norm(u[:,0])))))

    # Angle y
    ww = np.sum(mask, axis = 1)
    ww = np.rot90(ww)
    w = np.nonzero(ww)
    w_min = np.amin(w)
    w_max = np.amax(w)
    w_norm = (w - w_min)/(w_max - w_min)
    cov = np.cov(w_norm)
    u, s, vh = LA.svd(cov)
    data[4] = np.rad2deg(np.arccos((np.dot(z_ref, u[:,0])/(LA.norm(z_ref)*LA.norm(u[:,0])))))

    # Angle x
    ww = np.sum(mask, axis = 2)
    ww = np.rot90(ww)
    w = np.nonzero(ww)
    w_min = np.amin(w)
    w_max = np.amax(w)
    w_norm = (w - w_min)/(w_max - w_min)
    cov = np.cov(w_norm)
    u, s, vh = LA.svd(cov)
    data[5] = np.rad2deg(np.arccos((np.dot(z_ref, u[:,0])/(LA.norm(z_ref)*LA.norm(u[:,0])))))

    # Pattern max and mean of mask
    mask_pattern = np.multiply(image,mask)
    data[10] = np.amax(mask_pattern)
    n_t = np.nonzero(mask_pattern)
    box = mask_pattern[np.amin(n_t[0]):np.amax(n_t[0]),np.amin(n_t[1]):np.amax(n_t[1]),np.amin(n_t[2]):np.amax(n_t[2])]
    data[11] = np.mean(box)
    s = box.shape
    data[12] = np.mean(box[:,:,(s[2]-40):s[2]])

    # Fourier max value and number of peaks
    at = np.sum(mask_pattern, axis = 1)
    at = np.sum(at, axis = 0)
    p = at[at != 0]
    yf = fft(p)
    data[13] = np.real(np.amax(yf))
    yf = uniform_filter1d(abs(yf), size=6)
    peaks = find_peaks(abs(yf))
    data[14] = len(peaks[0])

    for i in range(15):
        data[i] = (data[i] - subt[i])/div[i]

    return data
