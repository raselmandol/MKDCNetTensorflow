import tensorflow as tf

#Dice Loss
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        y_true = tf.reshape(y_true, shape=(-1,))
        y_pred = tf.reshape(y_pred, shape=(-1,))
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2 * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)
        return 1 - dice

#Dice + BCE Loss
class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1, **kwargs):
        super(DiceBCELoss, self).__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        y_true = tf.reshape(y_true, shape=(-1,))
        y_pred = tf.reshape(y_pred, shape=(-1,))
        intersection = tf.reduce_sum(y_true * y_pred)
        dice_loss = 1 - (2 * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)
        BCE = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

#Metrics
def precision(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (intersection + 1e-15) / (tf.reduce_sum(y_pred) + 1e-15)

def recall(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (intersection + 1e-15) / (tf.reduce_sum(y_true) + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * tf.reduce_sum(y_true * y_pred) + 1e-15) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-15)

def jac_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-15) / (union + 1e-15)
