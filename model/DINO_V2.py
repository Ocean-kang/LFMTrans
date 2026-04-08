
import random

import torch
import torch.nn as nn

from torchvision import transforms as T

from utils.kmeans import train_kmeans_faiss as train_kmeans

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class DINO_V2(nn.Module):
    """ Vision Transformer """
    def __init__(self, cfg):
        super().__init__()

        # load DINOv2
        # BACKBONE_SIZE = "small"
        # self.patch_size = 14
        self.patch_size = 14
        BACKBONE_SIZE = cfg.DINOv2_BACKBONE_SIZE # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": f"vits{self.patch_size}_reg",
            "base": f"vitb{self.patch_size}_reg",
            "large": f"vitl{self.patch_size}_reg",
            "giant": f"vitg{self.patch_size}_reg",
        }
        backbone_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backnbone_dim = backbone_dims[BACKBONE_SIZE]
        visual_backbone_arch = backbone_archs[BACKBONE_SIZE]
        visual_backbone_name = f"dinov2_{visual_backbone_arch}"

        self.visual_backbone = torch.hub.load(repo_or_dir="DINOv2", source='local', model=visual_backbone_name)
        self.visual_backbone.eval()

        self.img_aug = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        )

        self.img_transfer = torch.nn.Sequential(
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.llama_projector = nn.Linear(4096, self.backnbone_dim, bias=False)


    def data_transfer(self, img, augment=False):

        N, _, H, W = img.shape
        img = img[:, [2, 1, 0], :, :]  # rgb -> bgr
        img = img / 255.
        img = self.img_transfer(img)
        if augment:
            img = self.img_aug(img)

        return img


    def forward(self, img=None, ret_dense_feat=False, augment=False, feat_t=None, feat_v=None,
                cls_attn_vec=False, patch_attn_vec=False, cluster_on_patchtokens=False, k=0):
        '''
        img:            N x 3 x H x W
        text_embedding: N x L x D
        '''

        if img is not None:
            img_ = self.data_transfer(img, augment)
            N, _, H, W = img_.shape
            assert H % self.patch_size == 0 and W % self.patch_size == 0
            h, w = H // self.patch_size, W // self.patch_size

            with torch.no_grad():
                if not ret_dense_feat:
                    res = self.visual_backbone.forward_features(img_)
                    x_norm_clstoken = res['x_norm_clstoken']
                    return x_norm_clstoken
                elif cluster_on_patchtokens:
                    res = self.visual_backbone.forward_features(img_)
                    x_norm_clstoken = res['x_norm_clstoken']
                    x_norm_patchtokens = res['x_norm_patchtokens']
                    assert k > 0
                    x_norm_patchtokens = x_norm_patchtokens / (x_norm_patchtokens.norm(dim=-1, keepdim=True) + 1e-7)
                    patchtokens_centroids = torch.zeros_like(x_norm_clstoken)
                    patchtokens_centroids = patchtokens_centroids[:, None].repeat(1, k, 1)
                    for n in range(N):
                        patchtokens_centroids[n] = train_kmeans(x_norm_patchtokens[n], k=k, device='cuda',
                                                                gpu_index=[x_norm_patchtokens.device.index],
                                                                metric='ip', return_idx=False)

                    return patchtokens_centroids
                else:
                    if not cls_attn_vec:
                        res = self.visual_backbone.forward_features(img_)
                        x_norm_clstoken = res['x_norm_clstoken']
                        x_norm_patchtokens = res['x_norm_patchtokens']
                        return x_norm_clstoken, x_norm_patchtokens
                    else:
                        res = self.visual_backbone.forward_features_attn(img_, n=1)
                        # x_norm_clstoken = res['x_norm_clstoken']
                        x_norm_regtokens = res['x_norm_regtokens']
                        x_norm_patchtokens = res['x_norm_patchtokens']
                        attn_map = res['attn_map'][0]

                        n_regtokens = x_norm_regtokens.shape[1]
                        n_patchtokens = x_norm_patchtokens.shape[1]
                        n_heads = attn_map.shape[1]
                        dim = x_norm_patchtokens.shape[-1]

                        cls_attn = attn_map[:, :, :1,1+n_regtokens:]
                        patch_attn = attn_map[:, :, 1+n_regtokens:, 1+n_regtokens:]

                        x_norm_patchtokens_ = x_norm_patchtokens[:, None].repeat(1, n_heads, 1, 1)
                        # x_norm_patchtokens_ = x_norm_patchtokens_ / (x_norm_patchtokens_.norm(dim=-1, keepdim=True) + 1e-7)
                        # cls_attn_mean = cls_attn.mean(-1)[0].unsqueeze(-1)
                        # cls_attn[cls_attn < cls_attn_mean] = 0

                        cls_attn_max, cls_attn_min = cls_attn.max(-1)[0].unsqueeze(-1), cls_attn.min(-1)[0].unsqueeze(-1)
                        cls_attn = (cls_attn - cls_attn_min) / (cls_attn_max - cls_attn_min)

                        # cls attn vector
                        cls_attn = cls_attn[:, :, 0, :, None]
                        cls_attn_vec = (x_norm_patchtokens_ * cls_attn).sum(2)
                        cls_attn_vec = cls_attn_vec / (cls_attn_vec.norm(dim=-1, keepdim=True) + 1e-7)

                        test = cls_attn_vec @ cls_attn_vec.transpose(2, 1)

                        if not patch_attn_vec:
                            return cls_attn_vec
                        else:
                            # patch attn vector
                            patch_attn_max, patch_attn_min = patch_attn.mean(-1)[0].unsqueeze(-1), patch_attn.min(-1)[0].unsqueeze(-1)
                            patch_attn_norm = (patch_attn - patch_attn_min) / (patch_attn_max - patch_attn_min)

                            patch_attn_vec = torch.Tensor(N, n_heads, n_patchtokens, dim).to(x_norm_patchtokens.device)
                            patch_attn_norm = patch_attn_norm.unsqueeze(-1).softmax(3)
                            for i in range(n_patchtokens):
                                patch_attn_vec[:, :, i] = (x_norm_patchtokens_ * patch_attn_norm[:, :, i]).sum(2)

                            return cls_attn_vec, patch_attn_vec

        elif feat_t is not None and feat_v is not None:
            assert feat_t.shape[0] == 1
            return self.train_vt_translater(feat_t[0], feat_v)

    def transfor_llama(self, feat_llama, feat_dinov2):
        feat_llama_trans = self.llama_projector(feat_llama.float())

        # loss_dist = nn.functional.mse_loss(feat_llama_trans, feat_dinov2)

        loss_dist = nn.functional.cosine_embedding_loss(
            feat_llama_trans,
            feat_dinov2,
            torch.full((feat_llama.shape[0],), 1, dtype=torch.long, device=feat_llama.device)
        )

        return loss_dist

    def transfor_llama_unsupervised(self, feat_llama, feat_dinov2_patch):
        feat_llama_trans = self.llama_projector(feat_llama.float())

        feat_llama_trans_norm = feat_llama_trans / feat_llama_trans.norm(dim=-1, keepdim=True)
        feat_dinov2_patch_norm = feat_dinov2_patch / feat_dinov2_patch.norm(dim=-1, keepdim=True)
        sim = feat_llama_trans_norm @ feat_dinov2_patch_norm.transpose(1, 0)
        # loss_structure = 1 - sim.topk(int(len(feat_dinov2_patch) / 3), 1)[0].mean()
        loss_structure = 1 - sim.mean()


        # loss_sim = 1 - feat_llama_trans.mean(0) @ feat_dinov2_patch.mean(0)
        loss_sim = torch.nn.functional.mse_loss(feat_llama_trans.mean(0), feat_dinov2_patch.mean(0))
        w1, w2 = (0, 1) if loss_sim > 1 else (1, 0)
        loss = w1 * loss_structure + w2 * loss_sim

        return loss


    def train_vt_translater(self, feat_llama, feat_dinov2):
        '''

        Args:
            feat_llama:     N1 x d1
            feat_dinov2:    N2 x d2

        Returns:

        '''

        N_t, D_t = feat_llama.shape
        N_v, D_v = feat_dinov2.shape
        device = feat_llama.device

        # feat_llama_norm = feat_llama / feat_llama.norm(dim=-1, keepdim=True)
        # feat_dinov2_norm = feat_dinov2 / feat_dinov2.norm(dim=-1, keepdim=True)

        # ====== Re-construction ======
        feat_dinov2_rec = self.translater.reconstruct_v(feat_dinov2)
        feat_llama_rec = self.translater.reconstruct_t(feat_llama)

        # loss
        # VisualRecLoss = nn.CosineEmbeddingLoss()
        # TextRecLoss = nn.CosineEmbeddingLoss()
        # pad_id_rec = 1
        # label_visual_rec = torch.full((N_v,), pad_id_rec, dtype=torch.long, device=device)
        # label_text_rec = torch.full((N_t,), pad_id_rec, dtype=torch.long, device=device)
        # visual_rec_loss = VisualRecLoss(feat_dinov2_rec, feat_dinov2, label_visual_rec)
        # text_rec_loss = TextRecLoss(feat_llama_rec, feat_llama, label_text_rec)

        VisualRecLoss = nn.MSELoss()
        TextRecLoss = nn.MSELoss()
        visual_rec_loss = VisualRecLoss(feat_dinov2_rec, feat_dinov2)
        text_rec_loss = TextRecLoss(feat_llama_rec, feat_llama)


        # ====== Semantic Consistent ======:

        # visual token 1 -> semantic embedding -> text token -> semantic embedding -> visual token 2 <==> visual token 1
        feat_dinov2_consistent = self.translater.trans_t2v(self.translater.trans_v2t(feat_dinov2))

        # text token 1 -> semantic embedding -> visual token -> semantic embedding -> text token 2 <==> text token 1
        feat_llama_consistent = self.translater.trans_v2t(self.translater.trans_t2v(feat_llama))

        # loss
        # VisualConsistentLoss = nn.CosineEmbeddingLoss()
        # TextConsistentLoss = nn.CosineEmbeddingLoss()
        # pad_id_c = 1
        # label_visual_consistent = torch.full((N_v,), pad_id_c, dtype=torch.long, device=device)
        # label_text_consistent = torch.full((N_t,), pad_id_c, dtype=torch.long, device=device)
        # visual_consistent_loss = VisualConsistentLoss(
        #     feat_dinov2_consistent,
        #     feat_dinov2,
        #     label_visual_consistent
        # )
        # text_consistent_loss = TextConsistentLoss(
        #     feat_llama_consistent,
        #     feat_llama,
        #     label_text_consistent
        # )


        VisualConsistentLoss = nn.MSELoss()
        TextConsistentLoss = nn.MSELoss()
        visual_consistent_loss = VisualConsistentLoss(
            feat_dinov2_consistent,
            feat_dinov2
        )
        text_consistent_loss = TextConsistentLoss(
            feat_llama_consistent,
            feat_llama
        )

        results = dict()
        results['VisualRecLoss'] = visual_rec_loss
        results['TextRecLoss'] = text_rec_loss
        results['VisualConsistentLoss'] = visual_consistent_loss
        results['TextConsistentLoss'] = text_consistent_loss

        return results

    def grad_cam(self, img, cls_token, augment=True):
        '''
        img:            N x 3 x H x W
        text_embedding: N x L x D
        '''

        img_ = self.data_transfer(img) if augment else (img)

        N, _, H, W = img_.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h, w = H // self.patch_size, W // self.patch_size
        self.visual_backbone.eval()
        # with torch.no_grad():
        #     features = self.visual_backbone.forward_features(img_)
        features = self.visual_backbone.forward_features_attn(img_, n=12)
        x_norm_clstoken = features['x_norm_clstoken']
        x_norm_patchtokens = features['x_norm_patchtokens']
        n_patchtokens = x_norm_patchtokens.shape[1]
        x_prenorm = features['x_prenorm']
        attn_map = features['attn_map']
        for attn in attn_map:
            attn.retain_grad()
        # attn_map[-1].retain_grad()

        N_cls = cls_token.shape[0]

        num_register_tokens = self.visual_backbone.num_register_tokens
        attn_vis = attn_map[-1].mean(1)[:, 0, 1 + num_register_tokens:].view(N, h, w)
        attn_vis = attn_vis[:, None] / attn_vis.view(N, -1).max(-1)[0].view(N, 1, 1, 1)

        import matplotlib.pyplot as plt
        plt.imshow(img[0].permute(1, 2, 0).int().cpu())
        plt.show()
        for n in range(N_cls):

            n_cls_feat = cls_token[n][None]

            self.visual_backbone.zero_grad()
            for attn in attn_map:
                attn.grad = None
            # attn_map[-1].grad = None

            x_norm_clstoken_ = x_norm_clstoken / x_norm_clstoken.norm(dim=1, keepdim=True)
            n_cls_feat_ = n_cls_feat / n_cls_feat.norm(dim=1, keepdim=True)
            loss = (1 - x_norm_clstoken_ @ n_cls_feat_.transpose(1, 0)).mean()
            print(f"n:{n}, sim:{float(x_norm_clstoken_ @ n_cls_feat_.transpose(1, 0))}")
            # loss = torch.nn.functional.mse_loss(x_norm_clstoken, n_cls_feat)
            loss = torch.nn.functional.cross_entropy(x_norm_clstoken, n_cls_feat)
            loss.backward(retain_graph=True)
            #
            #
            attn_grad_vis = attn_map[-1].grad.mean(1)[:, 0, -n_patchtokens:].view(N, 1, h, w)
            attn_grad_fg = -attn_grad_vis
            attn_grad_fg[attn_grad_fg < 0] = 0

            plt.imshow(attn_grad_fg[0, 0].cpu())
            plt.show()


            for attn in attn_map:
                attn_grad_vis = attn.grad.mean(1)[:, 0, -n_patchtokens:].view(N, 1, h, w)
                attn_grad_fg = -attn_grad_vis
                attn_grad_fg[attn_grad_fg < 0] = 0

                plt.imshow(attn_grad_fg[0, 0].cpu())
                plt.show()


            # print()
            # x_norm_patchtokens_ = x_norm_patchtokens[0] / x_norm_patchtokens[0].norm(dim=1, keepdim=True)
            # n_cls_feat_ = cls_token[5:6] / cls_token[5:6].norm(dim=1, keepdim=True)
            # sim = x_norm_patchtokens_ @ n_cls_feat_.transpose(1, 0)
            # plt.imshow(sim.view(h, w).detach().cpu())
            # plt.show()
            #
            # patch_sim = x_norm_patchtokens_ @ x_norm_patchtokens_.transpose(1, 0)
            # plt.imshow(patch_sim[0].view(h, w).detach().cpu())
            # plt.show()
        #
        #
        #
        # if not ret_dense_feat:
        #     return x_norm_clstoken
        # else:
        #     x_norm_patchtokens = self.visual_backbone.forward_features(img_)['x_norm_patchtokens']
        #     return x_norm_clstoken, x_norm_patchtokens
