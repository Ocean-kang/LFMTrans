import logging
import warnings
from PIL import Image

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torchvision.transforms as T

from model.DINO_V2 import DINO_V2 as VisualEncoder
from upsample_anything.upsample_anything import UPA

warnings.filterwarnings('ignore', category=UserWarning)
ex = Experiment('LFMtrans')

def create_basic_stream_logger(format):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

ex.logger = create_basic_stream_logger('%(levelname)s - %(name)s - %(message)s')
ex.add_config('./configs/LFMTrans_cfg.yaml')



@ex.automain
def main(_run, _log):
    # cfg loading
    cfg = edict(_run.config)
    cfg.train.dataset=str(cfg.train.dataset)

    # device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 读图
    img_path = "./upsample_anything/sample.jpg"

    # DINOv2 ViT-L/14 的 patch size 是 14
    # 所以图片尺寸最好是 14 的整数倍
    # 224 -> 16x16 tokens
    # 518 -> 37x37 tokens
    H, W = 518, 518
    img1 = Image.open(img_path).convert("RGB")
    img = Image.open(img_path).convert("RGB").resize((W, H), Image.BICUBIC)

    # 2. 加载 DINOv2 ViT-L/14
    dinov2 = VisualEncoder(cfg=cfg.eval)
    dinov2.eval()
    dinov2.to(device)
    # dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    # dinov2 = dinov2.to(device).eval()

    # 3. 图像预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    breakpoint()
    # 4. 提取 DINOv2 patch tokens
    with torch.no_grad():
        _, patchtokens = dinov2(img_tensor, augment=False, ret_dense_feat=True)
        patch_tokens = patchtokens  # [1, N, C]

    B, N, C = patch_tokens.shape
    h_lr = H // 14
    w_lr = W // 14

    assert N == h_lr * w_lr, f"N={N}, but h_lr*w_lr={h_lr*w_lr}"

    feat_lr = patch_tokens.reshape(B, h_lr, w_lr, C)
    feat_lr = feat_lr.permute(0, 3, 1, 2).contiguous()  # [1, C, h_lr, w_lr]

    print("LR feature:", feat_lr.shape)

    # 5. Upsample Anything
    # 输入:
    # img: PIL RGB image
    # feat_lr: [1, C, h_lr, w_lr] CUDA tensor
    feat_hr = UPA(img, feat_lr)

    print("HR feature:", feat_hr.shape)

    # 保存一下，后面你可以加载做相似度/分割
    torch.save(
        {
            "feat_lr": feat_lr.detach().cpu(),
            "feat_hr": feat_hr.detach().cpu(),
        },
        "dinov2_vitl14_upa_feature.pt",
    )
    breakpoint()