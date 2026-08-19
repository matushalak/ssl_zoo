from timm.models.vision_transformer import vit_base_patch16_224

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

from models.common import SSLModel


class MAE(SSLModel):
    def __init__(self):
        super().__init__()
        vit = vit_base_patch16_224()
        self.patch_size = 16
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        self.backbone = MaskedVisionTransformerTIMM(vit)
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
        )

    def forward(self, images):
        return self.backbone(images)

    def reconstruct(self, images, mask_ratio=0.75):
        keep, mask = utils.random_token_mask(
            (len(images), self.sequence_length), mask_ratio, images.device
        )
        encoded = self.backbone.encode(images, keep)
        decoded = self.decoder.embed(encoded)
        tokens = self.decoder.mask_token.repeat(len(images), self.sequence_length, 1)
        tokens = utils.set_at_index(tokens, keep, decoded)
        pixels = self.decoder.predict(self.decoder.decode(tokens))
        return utils.get_at_index(pixels, mask), mask
