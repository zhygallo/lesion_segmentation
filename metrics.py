from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
import tensorlayer

def tf_dice_coef(y_true, y_pred):
    y_pred = y_pred[:,:,:,:,1]
    y_true = y_true[:, :, :, 1]
    return tensorlayer.cost.dice_coe(y_pred, y_true, smooth = 1)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    y_pred_f = K.reshape(y_pred, (-1, 2))[:, 1]
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# def recall(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     c1 = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
#     # c2 = K.sum(K.round(K.clip(y_pred_f, 0, 1)))
#     c3 = K.sum(K.round(K.clip(y_true_f, 0, 1)))
#
#     return c1 / c3

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.reshape(y_pred, (-1, 2))
#     intersection = K.mean(y_true_f * y_pred_f[:, 0]) + K.mean((1.0 - y_true_f) * y_pred_f[:, 1])
#
#     return 2. * intersection;