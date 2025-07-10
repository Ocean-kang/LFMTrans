import torch
import torch.nn as nn

from utils.knngraph import Latent_knn_graph_construct, Latent_knn_graph_construct_numpy
from utils.LatentFuncitonMap import build_normalized_laplacian_matrix, laplacian_eigendecomposition, laplacian_main_sparse
from model.fmap_network import RegularizedFMNet
from loss.fmap_loss import SURFMNetLoss

class proj_loss(nn.Module):

    def __init__(self, cfg):
        super(proj_loss, self).__init__()
        self.cfg = cfg
        pass
    
    def forward(self, feat_v, feat_t):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # vision
        W_v = Latent_knn_graph_construct(self.cfg, feat_v, device, symmetrize=True)
        L_v = build_normalized_laplacian_matrix(W_v, device)
        v_vecs, v_vals = laplacian_eigendecomposition(L_v, self.cfg.laplacian_mat.k, device)
        #language
        W_t = Latent_knn_graph_construct(self.cfg, feat_t, device, symmetrize=True)
        L_t = build_normalized_laplacian_matrix(W_t, device)
        t_vecs, t_vals = laplacian_eigendecomposition(L_t, self.cfg.laplacian_mat.k, device)

        # adding batch dimension
        # vision
        feat_v = feat_v.to(device).unsqueeze(0)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)
        # language
        feat_t = feat_t.float().to(device).unsqueeze(0)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # detach grad of eigenvec and eigenval
        v_vecs = v_vecs.detach()
        v_vals = v_vals.detach()
        t_vecs = t_vecs.detach()
        t_vals = t_vals.detach()
        
        # build regularized_funciton_map model
        fm_net = RegularizedFMNet(bidirectional=True)
        Cxy, Cyx = fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)

        # fm_loss
        fucmap_loss = SURFMNetLoss()
        fm_loss_dict = fucmap_loss(Cxy, Cyx, v_vals, t_vals)

        return fm_loss_dict
    

class proj_loss_sparse(nn.Module):

    def __init__(self, cfg):
        super(proj_loss_sparse, self).__init__()
        self.cfg = cfg
        pass
    
    def forward(self, feat_v, feat_t):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # vision
        W_v = Latent_knn_graph_construct_numpy(self.cfg, feat_v, device, symmetrize=True)
        v_vecs, v_vals = laplacian_main_sparse(W_v, self.cfg.laplacian_mat.k)

        #language
        W_t = Latent_knn_graph_construct_numpy(self.cfg, feat_t, device, symmetrize=True)
        t_vecs, t_vals = laplacian_main_sparse(W_t, self.cfg.laplacian_mat.k)

        # cpu -> gpu and np.arrary -> torch.tensor
        # vision
        v_vecs = torch.from_numpy(v_vecs).to(device).float()
        v_vals = torch.from_numpy(v_vals).to(device).float()
        # language
        t_vecs = torch.from_numpy(t_vecs).to(device).float()
        t_vals = torch.from_numpy(t_vals).to(device).float()

        # adding batch dimension
        # vision
        feat_v = feat_v.to(device).unsqueeze(0)
        v_vecs = v_vecs.unsqueeze(0)
        v_vals = v_vals.unsqueeze(0)
        # language
        feat_t = feat_t.float().to(device).unsqueeze(0)
        t_vecs = t_vecs.unsqueeze(0)
        t_vals = t_vals.unsqueeze(0)

        # detach grad of eigenvec and eigenval
        v_vecs = v_vecs.detach()
        v_vals = v_vals.detach()
        t_vecs = t_vecs.detach()
        t_vals = t_vals.detach()
        
        # build regularized_funciton_map model
        fm_net = RegularizedFMNet(bidirectional=True)
        Cxy, Cyx = fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)

        # fm_loss
        fucmap_loss = SURFMNetLoss()
        fm_loss_dict = fucmap_loss(Cxy, Cyx, v_vals, t_vals)

        return fm_loss_dict