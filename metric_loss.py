import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np

# Dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

# Dice loss
def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# Binary crossentropy
def b_cross(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return binary_crossentropy(y_true, y_pred)

def dice_class1(y_true, y_pred):
    true_mask1 = y_true[:,:,:,0]
    pred_mask1 = y_pred[:,:,:,0]
    return dice_coef(true_mask1, pred_mask1)

def dice_class2(y_true, y_pred):
    true_mask1 = y_true[:,:,:,1]
    pred_mask1 = y_pred[:,:,:,1]
    return dice_coef(true_mask1, pred_mask1)

def dice_class3(y_true, y_pred):
    true_mask1 = y_true[:,:,:,2]
    pred_mask1 = y_pred[:,:,:,2]
    return dice_coef(true_mask1, pred_mask1)


# Both 1 and 0 dice
#def dice_coef_w_empty(y_true, y_pred, smooth=eps):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)

# Combination DL & CE
def loss_dl_ce(y_true, y_pred):
    return (0.5*b_cross(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred))

def test_loss_dl_ce(y_true, y_pred):
    return (0.5*test_bc_part(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred))

def weighted_loss_dl_ce(y_true, y_pred):
    true_mask1 = y_true[:,:,:,0]
    pred_mask1 = y_pred[:,:,:,0]
    w1 = 1/21084
    loss1 = ((0.5*b_cross(true_mask1, pred_mask1)) + (0.5*dice_loss(true_mask1, pred_mask1)))*w1

    true_mask2 = y_true[:,:,:,1]
    pred_mask2 = y_pred[:,:,:,1]
    w2 = 1/23715
    loss2 = ((0.5*b_cross(true_mask2, pred_mask2)) + (0.5*dice_loss(true_mask2, pred_mask2)))*w2

    true_mask3 = y_true[:,:,:,2]
    pred_mask3 = y_pred[:,:,:,2]
    w3 = 1/7900
    loss3 = ((0.5*b_cross(true_mask3, pred_mask3)) + (0.5*dice_loss(true_mask3, pred_mask3)))*w3

    return (loss1 + loss2 + loss3)*(1/(w1+w2+w3))

def test_dice_part(y_true, y_pred):
    s = K.shape(y_true)[0]
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    values = (2. * intersection + K.epsilon()) / (K.sum(y_true_f, axis=1) + K.sum(y_pred_f,axis=1) + K.epsilon())
    return values

def test_bc_part(y_true, y_pred):
    s = K.shape(y_true)[0]
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    cross = K.binary_crossentropy(y_true_f, y_pred_f)
    m = K.mean(cross, axis=1)
    return m

def test_loss(y_true, y_pred):
    s = K.shape(y_true)[0]
    d1 = 1. - test_dice_part(y_true[:,:,:,0], y_pred[:,:,:,0])
    d2 = 1. - test_dice_part(y_true[:,:,:,1], y_pred[:,:,:,1])
    d3 = 1. - test_dice_part(y_true[:,:,:,2], y_pred[:,:,:,2])
    c1 = test_bc_part(y_true[:,:,:,0], y_pred[:,:,:,0])
    c2 = test_bc_part(y_true[:,:,:,1], y_pred[:,:,:,1])
    c3 = test_bc_part(y_true[:,:,:,2], y_pred[:,:,:,2])
    
    l1 = (0.5*d1) + (0.5*c1)
    l2 = (0.5*d2) + (0.5*c2)
    l3 = (0.5*d3) + (0.5*c3)
    
    #Sätt grejer till noll här 
    
    l1 = K.sum(l1)/tf.cast(s, tf.float32)
    l2 = K.sum(l2)/tf.cast(s, tf.float32)
    l3 = K.sum(l3)/tf.cast(s, tf.float32)
    
    loss = (l1+l2+l3)/3.
    
    return loss


# Mutli loss
def dice_part(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=1)
    values = (2. * intersection + K.epsilon()) / (K.sum(y_true, axis=1) + K.sum(y_pred,axis=1) + K.epsilon())
    return values

def bc_part(y_true, y_pred):
    cross = K.binary_crossentropy(y_true, y_pred)
    m = K.mean(cross, axis=1)
    return m

def m_loss_class(y_true, y_pred):
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    #print(K.shape(y_true))
    # How many times the mask accures
    a = K.sum(w)
    # Chech if masks are present at all
    p = K.switch(K.equal(a,0), 0, 1)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    l = (0.5*(1.-dice_part(y_true_f, y_pred_f))) + (0.5*bc_part(y_true_f, y_pred_f))
    #l = bc_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    l = w*l
    # Sum and div by number of present masks
    l = K.sum(l)/(a + K.epsilon())
    return l, p

def Multi_loss(y_true, y_pred):
    l1, p1 = m_loss_class(y_true[:,:,:,0], y_pred[:,:,:,0])
    l2, p2 = m_loss_class(y_true[:,:,:,1], y_pred[:,:,:,1])
    l3, p3 = m_loss_class(y_true[:,:,:,2], y_pred[:,:,:,2])
    l4, p4 = m_loss_class(y_true[:,:,:,3], y_pred[:,:,:,3])
    l5, p5 = m_loss_class(y_true[:,:,:,4], y_pred[:,:,:,4])
    l6, p6 = m_loss_class(y_true[:,:,:,5], y_pred[:,:,:,5])
    l7, p7 = m_loss_class(y_true[:,:,:,6], y_pred[:,:,:,6])
    l8, p8 = m_loss_class(y_true[:,:,:,7], y_pred[:,:,:,7])

    loss = (l1+l2+l3+l4+l5+l6+l7+(1*l8))/(tf.cast((p1+p2+p3+p4+p5+p6+p7), dtype=tf.float32)+(1*tf.cast((p8), dtype=tf.float32)))
    
    return loss

def m_dice_class(y_true, y_pred):
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
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d, p

def Multi_dice_coef(y_true, y_pred):
    l1, p1 = m_dice_class(y_true[:,:,:,0], y_pred[:,:,:,0])
    l2, p2 = m_dice_class(y_true[:,:,:,1], y_pred[:,:,:,1])
    l3, p3 = m_dice_class(y_true[:,:,:,2], y_pred[:,:,:,2])
    l4, p4 = m_dice_class(y_true[:,:,:,3], y_pred[:,:,:,3])
    l5, p5 = m_dice_class(y_true[:,:,:,4], y_pred[:,:,:,4])
    l6, p6 = m_dice_class(y_true[:,:,:,5], y_pred[:,:,:,5])
    l7, p7 = m_dice_class(y_true[:,:,:,6], y_pred[:,:,:,6])
    l8, p8 = m_dice_class(y_true[:,:,:,7], y_pred[:,:,:,7])

    dice = (l1+l2+l3+l4+l5+l6+l7+l8)/tf.cast((p1+p2+p3+p4+p5+p6+p7+p8), dtype=tf.float32)
    
    return dice


def class_dice1(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice2(y_true, y_pred):
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice3(y_true, y_pred):
    y_true = y_true[:,:,:,2]
    y_pred = y_pred[:,:,:,2]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice4(y_true, y_pred):
    y_true = y_true[:,:,:,3]
    y_pred = y_pred[:,:,:,3]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice5(y_true, y_pred):
    y_true = y_true[:,:,:,4]
    y_pred = y_pred[:,:,:,4]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice6(y_true, y_pred):
    y_true = y_true[:,:,:,5]
    y_pred = y_pred[:,:,:,5]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice7(y_true, y_pred):
    y_true = y_true[:,:,:,6]
    y_pred = y_pred[:,:,:,6]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d

def class_dice8(y_true, y_pred):
    y_true = y_true[:,:,:,7]
    y_pred = y_pred[:,:,:,7]
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1])
    y_part = y_true[:,:,1:256]
    y_true = K.concatenate([z, y_part])
    # How many times the mask accures
    a = K.sum(w)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, 65536))
    y_pred_f = K.reshape(y_pred, (s, 65536))
    d = dice_part(y_true_f, y_pred_f)
    # Set loss to zero if mask does not excist
    d = w*d
    # Sum and div by number of present masks
    d = K.sum(d)/(a + K.epsilon())
    return d