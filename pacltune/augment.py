
import tensorflow as tf
import albumentations as A

# For compability reasons, augment_val can either be a bool or a string.
# With 'False' no augmentation is used and with 'True' 'tellez_pipe'.
# A string will use the augmentation with that name.
def get_fun(augment_val):
    if isinstance(augment_val, str):
        trans_pipe = globals()[augment_val]
    else:
        if augment_val:
            trans_pipe = tellez_pipe
        else:
            return None
    def augment_no_np(image):
        img = {"image":image}
        img_shape = image.shape
        aug_data = trans_pipe(**img)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = tf.image.resize(aug_img, [img_shape[0], img_shape[1]])
        aug_img = tf.clip_by_value(aug_img, 0, 1) # Values should be in [0,1]
        return aug_img
    def augment(image):
        aug_img = tf.numpy_function(func=augment_no_np,
                                    inp=[image],
                                    Tout=tf.float32)
        return aug_img
    return augment

# Tellez et al. ----------------------------------------------------------------
## Adopted from Tellez et al. (https://arxiv.org/pdf/1902.06543.pdf)
# We adopted their proposed basic, morphology, brightness & contrast (BC) and
# Hue-Saturation-Value (HSV) augmentation.
# We used fixed values for the elastic transform (α=80,σ=9) instead of variable
# ones (α∈[80,120],σ∈[9.0,11.0]), however, applied the transformation with 50%
# chance only. We increased the range of sigma of the Gaussian blur from
# [0,0.1] to [0,1]. For hue and saturation we used a middle ground between the
# two settings HSV-light ([-0.1,0.1]) and HSV-strong ([-1,1]) proposed by Tellez
# et al.: we used a ratio in [-0.3, 0.3].

tellez_pipe = A.Compose([
    A.RandomRotate90(p = 0.5),
    A.Flip(p = 0.5),
    A.ShiftScaleRotate(scale_limit=0.2,
                       shift_limit=0,
                       rotate_limit=0),
    A.ElasticTransform(alpha=80,
                       sigma=9),
    A.GaussNoise(var_limit=(0, 0.1)),
    A.GaussianBlur(sigma_limit=(0, 1)),
    A.RandomBrightnessContrast(brightness_limit=0.35,
                               contrast_limit=0.5),
    A.HueSaturationValue(hue_shift_limit=0.3*240,
                         sat_shift_limit=0.3,
                         val_shift_limit=0)
])

# ------------------------------------------------------------------------------
# Simple method ----------------------------------------------------------------
# Inspired by https://www.tensorflow.org/tutorials/images/data_augmentation#apply_augmentation_to_a_dataset
# and Tellez et al.

simple = A.Compose([
    A.RandomRotate90(p = 0.5),
    A.Flip(p = 0.5),
    A.ShiftScaleRotate(scale_limit=0.2,
                       shift_limit=0,
                       rotate_limit=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.35,
                               contrast_limit=0.5),
    A.HueSaturationValue(hue_shift_limit=0.3*240,
                         sat_shift_limit=0.3,
                         val_shift_limit=0)
])

# ------------------------------------------------------------------------------
