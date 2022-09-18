from .mix_transformer import MixVisionTransformer
from .trans_reg import TransFoot



def get_model(n_classes):
    model = TransFoot(image_size=(224, 100), patch_size=4, dim=4, trans_depth=8, heads=8, mlp_dim=16, dim_head=16,
                      num_classes=n_classes,
                      channels=1).cuda()
    return model