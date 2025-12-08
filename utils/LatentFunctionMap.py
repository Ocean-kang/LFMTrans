
import torch
import numpy as np
import tqdm

from utils.knngraph import Latent_knn_sysmmetric_graph_construct_numpy
from utils.laplacian_utils import laplacian_main_sparse
from utils.shuffle_utils import shuffle_features_and_labels
from utils.anchor_embeddings import anchor_embeddings_compute_unsupervised, anchor_embeddings_compute_supervised
from utils.fmap_util import pointmap2fmap, fmap2pointmap

from model.fmap_network import RegularizedFMNet


class LatentFunctionMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fm_net = RegularizedFMNet(bidirectional=True)
    
    def LFM(self, feat_v, feat_t, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''
        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.unsqueeze(0).to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_t = feat_t.unsqueeze(0).float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)
        
        # reduce batchsize
        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)        

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals

    def LFM_anchor(self, feat_v, feat_t, feat_labels_v, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''
        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_t = feat_t.float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # anchor descriptor
        feat_v_anchor, feat_t_anchor = anchor_embeddings_compute_unsupervised(self.cfg, feat_v, feat_t)

        # shuffle features
        feat_v_anchor, feat_labels_v = shuffle_features_and_labels(feat_v_anchor, feat_labels_v, self.cfg.seed)

        # add btach size
        feat_v_anchor = feat_v_anchor.unsqueeze(0)
        feat_t_anchor = feat_t_anchor.unsqueeze(0)

        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_v_anchor, feat_t_anchor, v_vals, t_vals, v_vecs, t_vecs)
        
        # reduce batchsize
        Cxy = Cxy.squeeze(0)
        Cyx = Cyx.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals, feat_labels_v
    
    def LFM_zoomout(self, feat_v, feat_t, step, device):
        '''
        feat_v: feature of vision,
        feat_t: feature of language,
        '''

        # compute once eigenvecs and eigenvals in each multimodels
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor and adding batchsize
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.unsqueeze(0).to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)

        # language
        feat_t = feat_t.unsqueeze(0).float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        Cxy_k, _ = self.fm_net(feat_v, feat_t, v_vals[:, :step], t_vals[:, :step], v_vecs[:, :step, :], t_vecs[:, :step, :])
        # Cxy_k = self.compute_fmap_basic(v_vecs[:, :step, :], t_vecs[:, :step, :], feat_v, feat_t)
        Cxy_k = Cxy_k.squeeze(0)
        v_vecs = v_vecs.squeeze(0)
        t_vecs = t_vecs.squeeze(0)

        for k in tqdm.tqdm(range(step, self.cfg.laplacian_mat.k-step, step), colour='red'):
            next_k = min(k+step, self.cfg.laplacian_mat.k)
            Pxy_k = fmap2pointmap(Cxy_k, v_vecs[:k].t(), t_vecs[:k].t())
            Cxy_k = pointmap2fmap(Pxy_k, v_vecs[:next_k].t(), t_vecs[:next_k].t())
        return Pxy_k
    
    def compute_fmap_basic(self, evecs_x, evecs_y, feat_x, feat_y, lambda_reg=1e-6):
        """
        Compute functional map C between X and Y using closed-form L2 least-squares.

        Inputs:
        evecs_x: [B, N, K] eigenvectors of X
        evecs_y: [B, N, K] eigenvectors of Y
        feat_x:  [B, C, N] descriptors on X
        feat_y:  [B, C, N] descriptors on Y
        lambda_reg: ridge regularization scalar

        Output:
        C: [B, K, K] functional map
        """
        evecs_x = evecs_x.transpose(1, 2)
        evecs_y = evecs_y.transpose(1, 2)
        feat_x = feat_x.transpose(1, 2)
        feat_y = feat_y.transpose(1, 2)

        B, C, N = feat_x.shape
        K = evecs_x.shape[-1]

        # Make descriptors [B, N, C]
        Fx = feat_x.transpose(1, 2)  # [B, N, C]
        Fy = feat_y.transpose(1, 2)  # [B, N, C]

        # 1. Project descriptors into spectral domain
        # A = Phi_X^T F_X  -> [B, K, C]
        A = torch.bmm(evecs_x.transpose(1, 2), Fx)

        # B = Phi_Y^T F_Y  -> [B, K, C]
        Bspec = torch.bmm(evecs_y.transpose(1, 2), Fy)

        # 2. Build Gram matrices
        AAT = torch.bmm(A, A.transpose(1, 2))          # [B, K, K]
        BAT = torch.bmm(Bspec, A.transpose(1, 2))      # [B, K, K]

        # 3. Ridge regularization
        I = torch.eye(K, device=evecs_x.device)[None].repeat(B, 1, 1)
        AAT_reg = AAT + lambda_reg * I

        # 4. Solve C = B A^T (A A^T + λ I)^(-1)
        C = torch.bmm(BAT, torch.linalg.inv(AAT_reg))
        return C

    def compute_fmap_extended(
        self,
        evecs_x, evecs_y, 
        feat_x, feat_y, 
        evals_x=None, evals_y=None,
        SX_list=None, SY_list=None,
        alpha_lap=0.0,
        beta_desc=0.0,
        lambda_reg=1e-6,
    ):
        """
        Closed-form functional map:
            || C A - B ||^2
        + alpha || Λ_Y C - C Λ_X ||^2
        + beta Σ_i || S^Y_i C - C S^X_i ||^2

        Inputs:
        evecs_x: [B, N, K]
        evecs_y: [B, N, K]
        feat_x:  [B, C, N]
        feat_y:  [B, C, N]

        evals_x: [B, K] (optional, for Laplacian commutativity)
        evals_y: [B, K]

        SX_list: list of descriptor operators on X, each [B, K, K]
        SY_list: list of descriptor operators on Y, each [B, K, K]

        alpha_lap: weight for Laplacian commutativity
        beta_desc: weight for descriptor commutativity
        lambda_reg: ridge regularization

        Returns:
        C: [B, K, K]
        """

        evecs_x = evecs_x.transpose(1, 2)
        evecs_y = evecs_y.transpose(1, 2)
        feat_x = feat_x.transpose(1, 2)
        feat_y = feat_y.transpose(1, 2)

        device = evecs_x.device
        B, N, K = evecs_x.shape

        # ----- 1: Project descriptors to spectral domain -----
        Fx = feat_x.transpose(1, 2)        # [B, N, C]
        Fy = feat_y.transpose(1, 2)        # [B, N, C]
        A = torch.bmm(evecs_x.transpose(1, 2), Fx)  # [B, K, C]
        Bspec = torch.bmm(evecs_y.transpose(1, 2), Fy)  # [B, K, C]

        # Build AAT / BAT
        AAT = torch.bmm(A, A.transpose(1, 2))      # [B, K, K]
        BAT = torch.bmm(Bspec, A.transpose(1, 2))  # [B, K, K]

        # RHS = BAT
        RHS = BAT.clone()

        # ----- 2: Laplacian Commutativity -----
        if alpha_lap > 0 and evals_x is not None and evals_y is not None:
            # Λ_X, Λ_Y: [B, K, K]
            LambdaX = torch.diag_embed(evals_x)  # [B, K, K]
            LambdaY = torch.diag_embed(evals_y)

            # Term: || Λ_Y C - C Λ_X ||^2
            #
            # Expands to additional A_lap and B_lap:
            # A_lap = alpha (Λ_Y^2 + Λ_X^2)
            # B_lap = alpha (Λ_Y Λ_X)
            #
            A_lap = alpha_lap * (torch.bmm(LambdaY, LambdaY) + torch.bmm(LambdaX, LambdaX))
            B_lap = alpha_lap * torch.bmm(LambdaY, LambdaX)

            AAT = AAT + A_lap
            RHS = RHS + B_lap

        # ----- 3: Descriptor Commutativity -----
        if beta_desc > 0 and SX_list is not None and SY_list is not None:
            for Sx, Sy in zip(SX_list, SY_list):
                # Each Sx, Sy: [B, K, K]
                #
                # || Sy C - C Sx ||^2  =>
                # A_desc += beta (Sy^T Sy + Sx Sx^T)
                # RHS    += beta (Sy Sx^T)
                A_desc = beta_desc * (torch.bmm(Sy.transpose(1, 2), Sy) +
                                    torch.bmm(Sx, Sx.transpose(1, 2)))
                B_desc = beta_desc * torch.bmm(Sy, Sx.transpose(1, 2))

                AAT = AAT + A_desc
                RHS = RHS + B_desc

        # ----- 4: Ridge regularization -----
        I = torch.eye(K, device=device).expand(B, K, K)
        AAT = AAT + lambda_reg * I

        # ----- 5: Solve C = RHS * (AAT)^(-1) -----
        C = torch.bmm(RHS, torch.linalg.inv(AAT))

        return C
    

class DeepFunctionMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fm_net = RegularizedFMNet(bidirectional=True)

    def Spectral_Basis(self, v, t, device):
        
        # vision
        W_v = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, v, device, symmetrize=False)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)
        # language
        W_t = Latent_knn_sysmmetric_graph_construct_numpy(self.cfg, t, device, symmetrize=False)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        return (v_vecs, t_vecs, v_vals, t_vals)
    
    def FunctionMap(self, feat_v, feat_t, v, t, device):

        v_vecs, t_vecs, v_vals, t_vals = self.Spectral_Basis(v, t, device)
        
        # adding batch dimension
        # vision
        feat_v = feat_v.unsqueeze(0).to(device)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)
        # language
        feat_t = feat_t.unsqueeze(0).float().to(device)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)
        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)

        return Cxy, Cyx, v_vecs, v_vals, t_vecs, t_vals
    
    def compute_fmap_basic(self, evecs_x, evecs_y, feat_x, feat_y, lambda_reg=1e-6):
        """
        Compute functional map C between X and Y using closed-form L2 least-squares.

        Inputs:
        evecs_x: [B, N, K] eigenvectors of X
        evecs_y: [B, N, K] eigenvectors of Y
        feat_x:  [B, C, N] descriptors on X
        feat_y:  [B, C, N] descriptors on Y
        lambda_reg: ridge regularization scalar

        Output:
        C: [B, K, K] functional map
        """

        B, C, N = feat_x.shape
        K = evecs_x.shape[-1]

        # Make descriptors [B, N, C]
        Fx = feat_x.transpose(1, 2)  # [B, N, C]
        Fy = feat_y.transpose(1, 2)  # [B, N, C]

        # 1. Project descriptors into spectral domain
        # A = Phi_X^T F_X  -> [B, K, C]
        A = torch.bmm(evecs_x.transpose(1, 2), Fx)

        # B = Phi_Y^T F_Y  -> [B, K, C]
        Bspec = torch.bmm(evecs_y.transpose(1, 2), Fy)

        # 2. Build Gram matrices
        AAT = torch.bmm(A, A.transpose(1, 2))          # [B, K, K]
        BAT = torch.bmm(Bspec, A.transpose(1, 2))      # [B, K, K]

        # 3. Ridge regularization
        I = torch.eye(K, device=evecs_x.device)[None].repeat(B, 1, 1)
        AAT_reg = AAT + lambda_reg * I

        # 4. Solve C = B A^T (A A^T + λ I)^(-1)
        C = torch.bmm(BAT, torch.linalg.inv(AAT_reg))
        return C

    def compute_fmap_extended(
        self,
        evecs_x, evecs_y, 
        feat_x, feat_y, 
        evals_x=None, evals_y=None,
        SX_list=None, SY_list=None,
        alpha_lap=0.0,
        beta_desc=0.0,
        lambda_reg=1e-6,
    ):
        """
        Closed-form functional map:
            || C A - B ||^2
        + alpha || Λ_Y C - C Λ_X ||^2
        + beta Σ_i || S^Y_i C - C S^X_i ||^2

        Inputs:
        evecs_x: [B, N, K]
        evecs_y: [B, N, K]
        feat_x:  [B, C, N]
        feat_y:  [B, C, N]

        evals_x: [B, K] (optional, for Laplacian commutativity)
        evals_y: [B, K]

        SX_list: list of descriptor operators on X, each [B, K, K]
        SY_list: list of descriptor operators on Y, each [B, K, K]

        alpha_lap: weight for Laplacian commutativity
        beta_desc: weight for descriptor commutativity
        lambda_reg: ridge regularization

        Returns:
        C: [B, K, K]
        """

        device = evecs_x.device
        B, N, K = evecs_x.shape

        # ----- 1: Project descriptors to spectral domain -----
        Fx = feat_x.transpose(1, 2)        # [B, N, C]
        Fy = feat_y.transpose(1, 2)        # [B, N, C]
        A = torch.bmm(evecs_x.transpose(1, 2), Fx)  # [B, K, C]
        Bspec = torch.bmm(evecs_y.transpose(1, 2), Fy)  # [B, K, C]

        # Build AAT / BAT
        AAT = torch.bmm(A, A.transpose(1, 2))      # [B, K, K]
        BAT = torch.bmm(Bspec, A.transpose(1, 2))  # [B, K, K]

        # RHS = BAT
        RHS = BAT.clone()

        # ----- 2: Laplacian Commutativity -----
        if alpha_lap > 0 and evals_x is not None and evals_y is not None:
            # Λ_X, Λ_Y: [B, K, K]
            LambdaX = torch.diag_embed(evals_x)  # [B, K, K]
            LambdaY = torch.diag_embed(evals_y)

            # Term: || Λ_Y C - C Λ_X ||^2
            #
            # Expands to additional A_lap and B_lap:
            # A_lap = alpha (Λ_Y^2 + Λ_X^2)
            # B_lap = alpha (Λ_Y Λ_X)
            #
            A_lap = alpha_lap * (torch.bmm(LambdaY, LambdaY) + torch.bmm(LambdaX, LambdaX))
            B_lap = alpha_lap * torch.bmm(LambdaY, LambdaX)

            AAT = AAT + A_lap
            RHS = RHS + B_lap

        # ----- 3: Descriptor Commutativity -----
        if beta_desc > 0 and SX_list is not None and SY_list is not None:
            for Sx, Sy in zip(SX_list, SY_list):
                # Each Sx, Sy: [B, K, K]
                #
                # || Sy C - C Sx ||^2  =>
                # A_desc += beta (Sy^T Sy + Sx Sx^T)
                # RHS    += beta (Sy Sx^T)
                A_desc = beta_desc * (torch.bmm(Sy.transpose(1, 2), Sy) +
                                    torch.bmm(Sx, Sx.transpose(1, 2)))
                B_desc = beta_desc * torch.bmm(Sy, Sx.transpose(1, 2))

                AAT = AAT + A_desc
                RHS = RHS + B_desc

        # ----- 4: Ridge regularization -----
        I = torch.eye(K, device=device).expand(B, K, K)
        AAT = AAT + lambda_reg * I

        # ----- 5: Solve C = RHS * (AAT)^(-1) -----
        C = torch.bmm(RHS, torch.linalg.inv(AAT))

        return C