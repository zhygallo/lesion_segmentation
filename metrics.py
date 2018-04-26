import tensorlayer as tl
import tensorflow as tf
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_pred = y_pred[:,:,:,:,1:] # take only predictions for possitive class
    dice = tl.cost.dice_coe(y_pred, y_true, loss_type = 'sorensen', axis=(1,2,3))
    return dice

def iou_coe(y_true, y_pred):
    y_pred = y_pred[:, :, :, :, 1:]
    iou = tl.cost.iou_coe(y_pred, y_true)
    return iou

def precision(y_true, y_pred):
    y_pred = y_pred[:, :, :, :, 1:]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_pred = y_pred[:, :, :, :, 1:]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    # y_pred = y_pred[:, :, :, :, 1:]
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * (p * r) / (p + r + K.epsilon())
    return f1_score