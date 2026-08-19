from timm.models.vision_transformer import vit_small_patch16_224

from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM

from models.common import SSLModel


class DINO(SSLModel):
    def __init__(self):
        super().__init__()
        self.backbone = MaskedVisionTransformerTIMM(
            vit_small_patch16_224(dynamic_img_size=True)
        )
        self.projection_head = DINOProjectionHead(384, norm_last_layer=False)
        self.student_backbone = MaskedVisionTransformerTIMM(
            vit_small_patch16_224(dynamic_img_size=True, drop_path_rate=0.1)
        )
        self.student_projection_head = DINOProjectionHead(
            384, freeze_last_layer=1, norm_last_layer=False
        )

    def forward(self, images):
        return self.backbone(images)
