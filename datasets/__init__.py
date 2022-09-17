from .datasets import data_img_seg


def build_dataset(train=True):
    ds_train, ds_valid = data_img_seg()
    if train:
        return ds_train
    else:
        return ds_valid