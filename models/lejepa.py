from timm.models.vision_transformer import vit_small_patch16_224

from lightly.models.modules import LeJEPAProjectionHead

from models.common import SSLModel


class LeJEPA(SSLModel):
    def __init__(self):
        super().__init__()
        self.backbone = vit_small_patch16_224(
            pretrained=False,
            pos_embed="learn",
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=0.1,
        )
        self.projection_head = LeJEPAProjectionHead(self.backbone.embed_dim)

    def forward(self, images):
        return self.backbone(images)
