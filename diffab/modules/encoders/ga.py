import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffab.modules.common.geometry import global_to_local, local_to_global, normalize_vector, construct_3d_basis, angstrom_to_nm
from diffab.modules.common.layers import mask_zero, LayerNorm
from diffab.utils.protein.constants import BBHeavyAtom


def _alpha_from_logits(logits, mask, inf=1e5):
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)

    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * pair_feat_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_feat_dim,
        )

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
        self.layer_norm_2 = LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) *
                       (1 / np.sqrt(self.query_key_dim))).sum(-1)
        return logits_node

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.num_heads, -1)

        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.num_heads, -1)

        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        return feat_node.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points,
                                                                  3))
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * \
                      value_points.unsqueeze(1)
        aggr_points = aggr_points.sum(dim=2)

        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1),
            feat_distance.reshape(N, L, -1),
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, mask,
                generate_mask=None, antigen_mask=None, ra_emb=None, relation_gate=None,
                log_gate=False):
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)

        logits_sum = logits_node + logits_pair + logits_spatial
        gate_stats = {}

        if relation_gate is not None and generate_mask is not None and antigen_mask is not None \
                and ra_emb is not None:
            if generate_mask.any() and antigen_mask.any():
                logits_sum, gate_stats = self._inject_relation_bias(
                    logits_sum, x, z, R, t, relation_gate, ra_emb,
                    generate_mask, antigen_mask, log_gate
                )

        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)

        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated, gate_stats

    def _inject_relation_bias(self, logits_sum, x, z, R, t, relation_gate, ra_emb,
                               generate_mask, antigen_mask, log_gate=False):
        N, L, F = x.shape
        num_heads = self.num_heads
        all_stats = {}

        for n in range(N):
            gm = generate_mask[n]
            am = antigen_mask[n]
            if not gm.any() or not am.any():
                continue

            cdr_idx = gm.nonzero(as_tuple=False).squeeze(-1)
            ag_idx = am.nonzero(as_tuple=False).squeeze(-1)

            if len(cdr_idx) == 0 or len(ag_idx) == 0:
                continue

            cdr_feat = x[n, cdr_idx].unsqueeze(0)
            ag_feat = x[n, ag_idx].unsqueeze(0)
            ra_emb_n = ra_emb[n, ag_idx].unsqueeze(0)

            pair_feat_sub = None
            if relation_gate.use_pair_feature:
                pair_feat_sub = z[n, cdr_idx][:, ag_idx].unsqueeze(0)

            distance = None
            if relation_gate.use_distance:
                cdr_pos = t[n, cdr_idx]
                ag_pos = t[n, ag_idx]
                distance = torch.cdist(cdr_pos, ag_pos).unsqueeze(0)

            gate_vals, _, stats = relation_gate(
                cdr_feat, ag_feat, ra_emb_n,
                pair_feat=pair_feat_sub, distance=distance,
                cdr_mask=torch.ones(len(cdr_idx), dtype=torch.bool, device=x.device),
                ag_mask=torch.ones(len(ag_idx), dtype=torch.bool, device=x.device),
                log_gate=log_gate,
            )

            if log_gate:
                all_stats = stats

            gate_bias = torch.log(gate_vals.squeeze(0).squeeze(-1) + 1e-6)

            for h in range(num_heads):
                logits_sum[n, cdr_idx[:, None], ag_idx[None, :], h] += gate_bias

        return logits_sum, all_stats


class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GABlock(node_feat_dim, pair_feat_dim, **ga_block_opt)
            for _ in range(num_layers)
        ])

    def forward(self, R, t, res_feat, pair_feat, mask,
                generate_mask=None, antigen_mask=None, ra_emb=None, relation_gate=None,
                log_gate=False):
        gate_stats = {}
        for i, block in enumerate(self.blocks):
            if i == 0 and relation_gate is not None:
                res_feat, gate_stats = block(R, t, res_feat, pair_feat, mask,
                                             generate_mask, antigen_mask, ra_emb, relation_gate,
                                             log_gate)
            else:
                res_feat, _ = block(R, t, res_feat, pair_feat, mask)
        return res_feat, gate_stats