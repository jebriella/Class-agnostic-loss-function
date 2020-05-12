import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np

# Class-agnostic loss function
def class_agnostic_loss(y_true, y_pred):
    l1, p1 = m_loss_class(y_true[:,:,:,0], y_pred[:,:,:,0])
    l2, p2 = m_loss_class(y_true[:,:,:,1], y_pred[:,:,:,1])
    l3, p3 = m_loss_class(y_true[:,:,:,2], y_pred[:,:,:,2])
    l4, p4 = m_loss_class(y_true[:,:,:,3], y_pred[:,:,:,3])
    l5, p5 = m_loss_class(y_true[:,:,:,4], y_pred[:,:,:,4])
    l6, p6 = m_loss_class(y_true[:,:,:,5], y_pred[:,:,:,5])
    l7, p7 = m_loss_class(y_true[:,:,:,6], y_pred[:,:,:,6])
    l8, p8 = m_loss_class(y_true[:,:,:,7], y_pred[:,:,:,7])

    loss = (l1+l2+l3+l4+l5+l6+l7+*l8)/(tf.cast((p1+p2+p3+p4+p5+p6+p7+p8), dtype=tf.float32))
    
    return loss

def m_loss_class(y_true, y_pred):
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Chech if masks are present at all
    p = K.switch(K.equal(a,0), 0, 1)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    l = (0.5*(1.-dice_part(y_true_f, y_pred_f))) + (0.5*bc_part(y_true_f, y_pred_f))
    # Set loss to zero if mask does not excist
    l = w*l
    # Sum and div by number of present masks
    l = K.sum(l)/(a + K.epsilon())
    return l, p

def dice_part(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=1)
    values = (2. * intersection + K.epsilon()) / (K.sum(y_true, axis=1) + K.sum(y_pred,axis=1) + K.epsilon())
    return values

def bc_part(y_true, y_pred):
    cross = K.binary_crossentropy(y_true, y_pred)
    m = K.mean(cross, axis=1)
    return m

