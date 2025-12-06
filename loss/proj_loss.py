import torch
import torch.nn as nn

from utils.knngraph import Latent_knn_graph_construct, Latent_knn_graph_construct_numpy
from utils.laplacian_utils import build_normalized_laplacian_matrix, laplacian_eigendecomposition, laplacian_main_sparse
from model.fmap_network import RegularizedFMNet
from loss.fmap_loss import SURFMNetLoss, SquaredFrobeniusLoss
from loss.ot_loss import SW


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
        self.fm_net = RegularizedFMNet(bidirectional=True)
        self.fucmap_loss = SURFMNetLoss()
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
        Cxy, Cyx = self.fm_net(feat_v, feat_t, v_vals, t_vals, v_vecs, t_vecs)

        # fm_loss
        fm_loss_dict = self.fucmap_loss(Cxy, Cyx, v_vals, t_vals)

        return fm_loss_dict
    

class proj_loss_sparse_oncefmap(nn.Module):

    def __init__(self, cfg):
        super(proj_loss_sparse_oncefmap, self).__init__()
        self.cfg = cfg
        self.fm_net = RegularizedFMNet(bidirectional=True)
        self.fucmap_loss = SURFMNetLoss()
        self.permute_loss = SquaredFrobeniusLoss()
        self.ot_loss = SW(loss_weight=cfg.loss.w_ot, L=cfg.loss.L_ot)
        self.loss_dict = dict()
        pass

    def _compute_alignment_loss(self, Cxy: torch.Tensor, Cyx: torch.Tensor, 
                            Pxy: torch.Tensor, Pyx: torch.Tensor,
                            evecs_x: torch.Tensor, evecs_y: torch.Tensor,
                            evecs_trans_x: torch.Tensor, evecs_trans_y: torch.Tensor) -> None:
        """Compute alignment loss between functional maps and permutation matrices.
        
        Args:
            Cxy, Cyx: Functional maps
            Pxy, Pyx: Permutation matrices
            evecs_x, evecs_y: Eigenvectors
            evecs_trans_x, evecs_trans_y: Transposed eigenvectors
        """
        # Compute estimated functional maps from permutation matrices
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        
        # Compute alignment loss
        self.loss_dict['l_align'] = self.permute_loss(Cxy, Cxy_est)
        
        # Add bidirectional loss if not partial
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
        self.loss_dict['l_align'] += self.permute_loss(Cyx, Cyx_est)


    def _compute_additional_losses(self, feat_x: torch.Tensor, feat_y: torch.Tensor, Pxy: torch.Tensor, Pyx: torch.Tensor) -> None:
            """Compute additional losses if available.
            
            Args:
                feat_x, feat_y: Feature tensors
                Pxy, Pyx: Permutation matrices
            """
            self.loss_dict['l_ot'] = self.ot_loss(feat_x, feat_y, Pxy, Pyx)
            

    
    def forward(self, feat_x, feat_y, x_vals, y_vals, x_vecs, y_vecs, Pxy, Pyx):

        feat_y = feat_y.float()
        feat_x = feat_x.float()

        # build regularized_funciton_map model
        Cxy, Cyx = self.fm_net(feat_x, feat_y, x_vals, y_vals, x_vecs, y_vecs)

        # fm_loss
        fm_loss_dict = self.fucmap_loss(Cxy, Cyx, x_vals, y_vals)
        self.loss_dict['l_lap'] = fm_loss_dict['l_lap']
        self.loss_dict['l_orth'] = fm_loss_dict['l_orth']
        self.loss_dict['l_bij'] = fm_loss_dict['l_bij']

        # permutation loss
        x_vecs_trans = x_vecs.transpose(2, 1)
        y_vecs_trans = y_vecs.transpose(2, 1)
        self._compute_alignment_loss(Cxy, Cyx, Pxy, Pyx, x_vecs_trans, y_vecs_trans, x_vecs, y_vecs)

        # optimal tansport loss
        self._compute_additional_losses(feat_x, feat_y, Pxy, Pyx)

        return self.loss_dict
    

class projector_loss(nn.Module):

    def __init__(self, cfg):
        super(projector_loss, self).__init__()
        self.cfg = cfg
        self.fucmap_loss = SURFMNetLoss()
        self.permute_loss = SquaredFrobeniusLoss()
        self.ot_loss = SW(loss_weight=cfg.loss.w_ot, L=cfg.loss.L_ot)
        self.loss_dict = dict()
        pass

    def _compute_alignment_loss(self, Cxy: torch.Tensor, Cyx: torch.Tensor, 
                            Pxy: torch.Tensor, Pyx: torch.Tensor,
                            evecs_x: torch.Tensor, evecs_y: torch.Tensor,
                            evecs_trans_x: torch.Tensor, evecs_trans_y: torch.Tensor) -> None:
        """Compute alignment loss between functional maps and permutation matrices.
        
        Args:
            Cxy, Cyx: Functional maps
            Pxy, Pyx: Permutation matrices
            evecs_x, evecs_y: Eigenvectors
            evecs_trans_x, evecs_trans_y: Transposed eigenvectors
        """
        # Compute estimated functional maps from permutation matrices
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        
        # Compute alignment loss
        self.loss_dict['l_align'] = self.permute_loss(Cxy, Cxy_est)
        
        # Add bidirectional loss if not partial
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
        self.loss_dict['l_align'] += self.permute_loss(Cyx, Cyx_est)


    def _compute_additional_losses(self, feat_x: torch.Tensor, feat_y: torch.Tensor, Pxy: torch.Tensor, Pyx: torch.Tensor) -> None:
            """Compute additional losses if available.
            
            Args:
                feat_x, feat_y: Feature tensors
                Pxy, Pyx: Permutation matrices
            """
            self.loss_dict['l_ot'] = self.ot_loss(feat_x, feat_y, Pxy, Pyx)
            

    def forward(self, feat_x, feat_y, Cxy, Cyx, x_vals, y_vals, x_vecs, y_vecs, Pxy, Pyx):

        feat_y = feat_y.float()
        feat_x = feat_x.float()

        # fm_loss
        fm_loss_dict = self.fucmap_loss(Cxy, Cyx, x_vals, y_vals)
        self.loss_dict['l_lap'] = fm_loss_dict['l_lap']
        self.loss_dict['l_orth'] = fm_loss_dict['l_orth']
        self.loss_dict['l_bij'] = fm_loss_dict['l_bij']

        # permutation loss
        x_vecs_trans = x_vecs.transpose(2, 1)
        y_vecs_trans = y_vecs.transpose(2, 1)
        self._compute_alignment_loss(Cxy, Cyx, Pxy, Pyx, x_vecs_trans, y_vecs_trans, x_vecs, y_vecs)

        # optimal tansport loss
        self._compute_additional_losses(feat_x, feat_y, Pxy, Pyx)

        return self.loss_dict