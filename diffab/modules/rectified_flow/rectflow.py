import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
import numpy as np
import logging

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix, construct_3d_basis_from_e
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3, compute_global_R_from_vec_to_standard,compute_global_R_from_standard_to_vec, rotation_to_quaternion, quaternion_to_rotation_matrix, quaternion_diff, grassmann_product
from diffab.modules.encoders.ga import GAEncoder
from .utils import matrix_to_euler_angles, euler_angles_to_matrix

from diffab.modules.common.layers import clampped_one_hot 

from .trans import AminoacidCategoricalTransition, PositionTransition, QuaternionTransition

class PredictNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Linear(20, res_feat_dim, bias=True)  # 22 is padding
        self.t_res_embedding = nn.Linear(1, res_feat_dim, bias=True)
        self.t_pair_embedding = nn.Linear(1, pair_feat_dim, bias=True)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.seq_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 20), 
        )

        self.crd_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 3)
        )
        self.quaternion_net = nn.Sequential(
            nn.Linear(res_feat_dim, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, res_feat_dim*2), nn.ReLU(),
            nn.Linear(res_feat_dim*2, 4)
        )

    def forward(self, R, p, s, t, res_feat, pair_feat, mask_generate, mask_res,
                generate_mask=None, antigen_mask=None, ra_emb=None, relation_gate=None,
                log_gate=False):
        N, L = mask_res.size()
        t_res_embed = self.t_res_embedding(t.unsqueeze(1))
        t_pair_embed = self.t_pair_embedding(t.unsqueeze(1))

        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s)], dim=-1))
        res_feat = res_feat + t_res_embed[:, None, :]
        pair_feat = pair_feat + t_pair_embed[:, None, None,  :]

        in_feat, gate_stats = self.encoder(R, p, res_feat, pair_feat, mask_res,
                                           generate_mask, antigen_mask, ra_emb, relation_gate,
                                           log_gate)

        vel_s = self.seq_net(in_feat)
        vel_s = torch.where(mask_generate[..., None].expand_as(vel_s), vel_s, torch.zeros_like(vel_s))

        vel_crd = self.crd_net(in_feat)
        vel_pos = apply_rotation_to_vector(R, vel_crd)
        vel_pos = torch.where(mask_generate[:, :, None].expand_as(vel_pos), vel_pos, torch.zeros_like(vel_pos))

        vel_qua = self.quaternion_net(in_feat)
        vel_qua = torch.where(mask_generate[:, :, None].expand_as(vel_qua), vel_qua, torch.zeros_like(vel_qua))

        return vel_s, vel_pos, vel_qua, gate_stats


class RectFlowGenerator(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.pred_net = PredictNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = 100
        self.trans_seq = AminoacidCategoricalTransition(num_steps)
        self.trans_pos = PositionTransition(num_steps)
        self.trans_qua = QuaternionTransition(num_steps)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1,  -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1,  -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, R_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, mask_anchor, R_template, p_template, s_template, mask_template_generate, \
        denoise_structure, denoise_sequence, template_enable=False, t=None, pdbid=None,
        generate_mask=None, antigen_mask=None, ra_emb=None, relation_gate=None,
        return_velocities=False):
        
        N, L = res_feat.shape[:2]
        t = torch.rand((N,), device=self._dummy.device) 
        p_0 = self._normalize_position(p_0)
        q_0 = rotation_to_quaternion(R_0)

        # Ensure all template tensors match the exact sequence length L
        # Slice if longer, pad if shorter
        
        # Handle R_template (N, *, 3, 3)
        if R_template.shape[1] > L:
            R_template = R_template[:, :L]
        elif R_template.shape[1] < L:
            pad_size = (0, 0, 0, 0, 0, L - R_template.shape[1])
            R_template = F.pad(R_template, pad_size, 'constant', 0)
        
        # Handle p_template (N, *, 3)
        if p_template.shape[1] > L:
            p_template = p_template[:, :L]
        elif p_template.shape[1] < L:
            pad_size = (0, 0, 0, L - p_template.shape[1])
            p_template = F.pad(p_template, pad_size, 'constant', 0)
        
        # Handle s_template (N, *)
        if s_template.shape[1] > L:
            s_template = s_template[:, :L]
        elif s_template.shape[1] < L:
            s_template = F.pad(s_template, (0, L - s_template.shape[1]), 'constant', 0)
        
        # Handle mask_template_generate (N, *)
        if mask_template_generate.shape[1] > L:
            mask_template_generate = mask_template_generate[:, :L]
        elif mask_template_generate.shape[1] < L:
            mask_template_generate = F.pad(mask_template_generate, (0, L - mask_template_generate.shape[1]), 'constant', False)

        # 确保template_enable是正确形状的张量，匹配mask_generate的长度
        if not isinstance(template_enable, torch.Tensor):
            template_enable = torch.full((N, L), template_enable, device=self._dummy.device)
        elif template_enable.dim() == 1:
            template_enable = template_enable.unsqueeze(1).expand(N, L)
        elif template_enable.dim() == 0:
            template_enable = template_enable.expand(N, L)
        elif template_enable.shape != (N, L):
            # 调整template_enable到正确形状
            if template_enable.shape[1] > L:
                template_enable = template_enable[:, :L].to(self._dummy.device)
            elif template_enable.shape[1] < L:
                template_enable = F.pad(template_enable, (0, L - template_enable.shape[1]), 'constant', False).to(self._dummy.device)
        
        p_template_norm = self._normalize_position(p_template)
        p_template = torch.where(template_enable[..., None], p_template_norm, torch.zeros_like(p_template_norm))
        q_template = rotation_to_quaternion(R_template)
        
        q_interp, q_init = self.trans_qua.interpolate(q_0, mask_generate, t, mask_template_generate=mask_template_generate, template_enable=template_enable, q_template=q_template, pdbid=pdbid)
    
        R_interp = quaternion_to_rotation_matrix(q_interp)
        R_interp = torch.where(mask_generate[..., None, None].expand_as(R_0), R_interp, R_0)

        if torch.isnan(torch.masked_select(R_interp, mask_generate[..., None, None])).any():
            logging.warning(f'none detected.')
            R_interp = torch.where(torch.isnan(R_interp), torch.zeros_like(R_interp), R_interp)
        
        p_interp, p_init = self.trans_pos.interpolate(p_0, mask_generate, t, mask_template_generate=mask_template_generate, template_enable=template_enable, p_template=p_template)
        
        s_0 = clampped_one_hot(s_0, num_classes=20).float()

        s_template = clampped_one_hot(s_template, num_classes=20).float()
        
        s_interp, s_init = self.trans_seq.interpolate(s_0, mask_generate, t, mask_template_generate=mask_template_generate, x_template=s_template, template_enable=template_enable)

        vel_s, vel_pos, vel_qua, gate_stats = self.pred_net(
            R_interp, p_interp, s_interp, t, res_feat, pair_feat, mask_generate, mask_res,
            generate_mask, antigen_mask, ra_emb, relation_gate,
            log_gate=(relation_gate is not None)
            )  
        
        loss_dict = {}
      
        loss_seq = F.mse_loss(s_0 - s_init, vel_s, reduction='none').mean(dim=-1)
        loss_seq = (loss_seq * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        loss_pos = F.mse_loss(p_0-p_init, vel_pos, reduction='none').mean(dim=-1)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        q_init_normalized = q_init / (torch.norm(q_init, dim=-1, keepdim=True)+1e-8)
        q_true_normalized = q_0 / (torch.norm(q_0, dim=-1, keepdim=True)+1e-8)

        delta_q = quaternion_diff(q_true_normalized, q_init_normalized)
        delta_q = torch.where(mask_generate[..., None].expand_as(delta_q), delta_q, torch.zeros_like(delta_q))
    
        loss_qua = F.mse_loss(delta_q, vel_qua, reduction='none').mean(dim=-1)
        loss_qua = (loss_qua * mask_generate).sum() / (mask_generate.sum().float()+1e-8)
        loss_dict['qua'] = loss_qua

        if gate_stats:
            for k, v in gate_stats.items():
                loss_dict[f'gate_{k}'] = v

        if return_velocities:
            predictions = {
                'vel_s': vel_s,
                'vel_pos': vel_pos,
                'vel_qua': vel_qua,
                'mask_generate': mask_generate,
            }
            return loss_dict, predictions

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        R, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, mask_anchor, 
        sample_structure=True, sample_sequence=True,
        pbar=False, template_enable=True, R_template=None,
        p_template=None, s_template=None, mask_template_generate=None,
        generate_mask=None, antigen_mask=None, ra_emb=None, relation_gate=None,
    ):
        N, L = p.shape[:2]

        p = self._normalize_position(p)
    
        if template_enable:
            p_template = self._normalize_position(p_template)
        
        if template_enable:
            R_template = torch.masked_select(R_template, mask_template_generate[..., None, None])
            R_init = R.clone().masked_scatter(mask_generate[..., None, None].expand_as(R), R_template)
        else:
            v_init = random_uniform_so3([N, L], device=R.device)
            R_init = so3vec_to_rotation(v_init)
            R_init = torch.where(mask_generate[..., None, None].expand_as(R), R_init, R)
        
        if template_enable:
            p_template = torch.masked_select(p_template, mask_template_generate[..., None])
            p_init = p.clone().masked_scatter(mask_generate[..., None].expand_as(p), p_template)

            e_rand = torch.randn_like(p_init)
            p_init = e_rand + p_init

            p_init = torch.where(mask_generate[..., None].expand_as(p), p_init, p)
        else:
            aa_mask = p.norm(dim=-1)
            aa_mask = ~(aa_mask==0)
            context_mask = torch.logical_and(aa_mask, ~mask_generate)
            p_avg = (p*context_mask[:, :, None]).sum(dim=1) / context_mask.sum(dim=1)[:, None]
            e_rand = torch.randn_like(p)
            p_init = e_rand + p_avg.detach().clone()[:, None, :]
            p_init = torch.where(mask_generate[..., None].expand_as(p), p_init, p)

        s = clampped_one_hot(s, num_classes=20).float()
        if template_enable:
            s_template = clampped_one_hot(s_template, num_classes=20).float()
        if template_enable:
            s_template = torch.masked_select(s_template, mask_template_generate[..., None])
            s_init = s.clone().masked_scatter(mask_generate[..., None], s_template)
            s_init = s_init + torch.randn_like(s_init, device=s_init.device)
            s_init = torch.where(mask_generate[..., None], s_init, s)
        else:
            s_init = torch.randn_like(s, device=s.device)
            s_init = torch.where(mask_generate[..., None], s_init, s)

        return_c_init = torch.argmax(s_init, dim=-1)
        return_v_init = rotation_to_so3vec(R_init.detach().clone())
        return_p_init = self._unnormalize_position(p_init)

        traj = {0: (return_v_init, return_p_init, return_c_init, s_init, p_init, R_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:  
            pbar = lambda x: x 

        dt = 1./self.num_steps
        
        for t in pbar(range(0, self.num_steps)):
            
            _, _, _, s_t, p_t, R_t = traj[t]
    
            t_tensor = torch.ones((N,), device=self._dummy.device) * t / self.num_steps + 1e-3
            vel_s, vel_pos, vel_qua, _ = self.pred_net(
                R_t, p_t, s_t, t_tensor, res_feat, pair_feat, mask_generate, mask_res,
                generate_mask, antigen_mask, ra_emb, relation_gate
            )

            s_next = s_t + dt * vel_s
            p_next = p_t + dt * vel_pos

            q_t = rotation_to_quaternion(R_t)
            q_t_normalized = q_t / (torch.norm(q_t, dim=-1, keepdim=True)+1e-8)
            
            vel_qua_normalized = vel_qua / (torch.norm(vel_qua, dim=-1, keepdim=True)+1e-8)
            theta = 2*torch.arccos(vel_qua_normalized[:, :, 0])
            vel_qua_normalized[:, :, 0] /= torch.cos(theta/2) + torch.ones_like(theta)*1e-8
            vel_qua_normalized[:, :, 1:] /= torch.sin(theta.unsqueeze(-1) / 2) + torch.ones_like(theta.unsqueeze(-1))*1e-8
            theta = theta * dt
            vel_qua_normalized[:, :, 0] *= torch.cos(theta/2)
            vel_qua_normalized[:, :, 1:] *= torch.sin(theta.unsqueeze(-1) / 2)

            q_next = grassmann_product(vel_qua_normalized, q_t_normalized)
            q_next = q_next / (torch.norm(q_next, dim=-1, keepdim=True)+1e-8)
            R_next = quaternion_to_rotation_matrix(q_next)
            R_next = torch.where(mask_generate[..., None, None].expand_as(R_next), R_next, R)
            v_next = rotation_to_so3vec(R_next)
            return_c_next = torch.argmax(s_next, dim=-1)
            return_v_next = v_next.detach().clone()
            return_p_next = self._unnormalize_position(p_next).detach().clone()

            traj[t+1] = (return_v_next, return_p_next, return_c_next, s_next.detach().clone(), p_next.detach().clone(), R_next.detach().clone())
            traj[t] = tuple(x.cpu() for x in traj[t])
        
        reverse_traj = {}
        for t in range(0, self.num_steps+1):
            reverse_traj[self.num_steps - t] = traj[t]

        return reverse_traj

