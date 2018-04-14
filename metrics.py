from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.reshape(y_pred, (-1, 2))
#     intersection = K.mean(y_true_f * y_pred_f[:, 0]) + K.mean((1.0 - y_true_f) * y_pred_f[:, 1])
#
#     return 2. * intersection;