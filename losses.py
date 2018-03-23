from metrics import dice_coef

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)