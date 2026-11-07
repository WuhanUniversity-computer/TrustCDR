import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class ReliabilityEmbedding(nn.Module):

    def __init__(self, ra_feat_dim=8, hidden_dim=128, out_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ra_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, ra_features):
        """
        Args:
            ra_features: [..., 8] raw U_A features
        Returns:
            [..., out_dim] reliability embedding
        """
        return self.mlp(ra_features)


class RelationGate(nn.Module):

    def __init__(self, node_feat_dim, ra_emb_dim=128, gate_hidden_dim=128, gate_dropout=0.1,
                 gate_mode='scalar', gate_init='near_identity', use_distance=True,
                 use_pair_feature=True, pair_feat_dim=64, num_heads=12):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.gate_mode = gate_mode
        self.gate_init = gate_init
        self.use_distance = use_distance
        self.use_pair_feature = use_pair_feature
        self.num_heads = num_heads

        input_dim = node_feat_dim * 2 + ra_emb_dim  # cdr_i + antigen_j + reliability_j
        if use_pair_feature:
            input_dim += pair_feat_dim
        if use_distance:
            input_dim += 1

        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
        )

        if gate_mode == 'scalar':
            self.gate_proj = nn.Linear(gate_hidden_dim, 1)
        elif gate_mode == 'multi_channel':
            self.gate_proj = nn.Linear(gate_hidden_dim, 3)
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")

        self._init_weights()

        self.gate_stats = {}

    def _init_weights(self):
        last_linear = None
        for module in reversed(list(self.gate_mlp.modules())):
            if isinstance(module, nn.Linear):
                last_linear = module
                break

        if self.gate_init == 'near_identity':
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.constant_(last_linear.bias, 0.5)
            nn.init.zeros_(self.gate_proj.weight)
            if self.gate_proj.bias is not None:
                nn.init.constant_(self.gate_proj.bias, 0.5)
        elif self.gate_init == 'zero':
            nn.init.zeros_(self.gate_proj.weight)
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)
        else:
            pass

    def compute_gate(self, cdr_feat, ag_feat, ra_emb, pair_feat=None, distance=None):
        """
        Args:
            cdr_feat: [B, L_cdr, node_feat_dim]
            ag_feat: [B, L_ag, node_feat_dim]
            ra_emb: [B, L_ag, ra_emb_dim]
            pair_feat: [B, L_cdr, L_ag, pair_feat_dim] or None
            distance: [B, L_cdr, L_ag, 1] or None
        Returns:
            gate: [B, L_cdr, L_ag, 1] scalar gate values
        """
        B, L_cdr, _ = cdr_feat.shape
        _, L_ag, _ = ag_feat.shape

        cdr_expand = cdr_feat.unsqueeze(2).expand(-1, -1, L_ag, -1)
        ag_expand = ag_feat.unsqueeze(1).expand(-1, L_cdr, -1, -1)
        ra_expand = ra_emb.unsqueeze(1).expand(-1, L_cdr, -1, -1)

        features = [cdr_expand, ag_expand, ra_expand]

        if self.use_pair_feature and pair_feat is not None:
            features.append(pair_feat)

        if self.use_distance and distance is not None:
            if distance.dim() == 2:
                distance = distance.unsqueeze(0).unsqueeze(-1).expand(B, L_cdr, L_ag, 1)
            elif distance.dim() == 3:
                distance = distance.unsqueeze(-1)
            features.append(distance)

        feat_cat = torch.cat(features, dim=-1)

        hidden = self.gate_mlp(feat_cat)
        gate_raw = self.gate_proj(hidden)
        gate = torch.sigmoid(gate_raw)

        gate = torch.clamp(gate, 0.01, 0.99)

        return gate

    def compute_attention_bias(self, gate_values, cdr_mask=None, ag_mask=None):
        """
        Convert gate values to attention logit bias.

        Args:
            gate_values: [B, L_cdr, L_ag, 1]
            cdr_mask: [B, L_cdr] or None
            ag_mask: [B, L_ag] or None
        Returns:
            bias: [B, L_total, L_total, num_heads] attention logit bias
            (non-zero only for CDR->antigen positions)
        """
        B, L_cdr, L_ag, _ = gate_values.shape

        bias = torch.log(gate_values + 1e-6)

        if cdr_mask is not None and ag_mask is not None:
            valid_pair = cdr_mask.unsqueeze(-1) & ag_mask.unsqueeze(-2)
            bias = bias * valid_pair.unsqueeze(-1).float()

        return bias

    def forward(self, cdr_feat, ag_feat, ra_emb, pair_feat=None, distance=None,
                cdr_mask=None, ag_mask=None, log_gate=False):
        """
        Full forward pass returning gate values and attention bias.

        Returns:
            gate_values: [B, L_cdr, L_ag, 1]
            attention_bias: [B, L_cdr, L_ag, 1] or None
            stats: dict
        """
        gate_values = self.compute_gate(cdr_feat, ag_feat, ra_emb, pair_feat, distance)

        attention_bias = None
        if log_gate:
            stats = {
                'mean_gate': gate_values.mean().detach(),
                'gate_std': gate_values.std().detach(),
                'gate_min': gate_values.min().detach(),
                'gate_max': gate_values.max().detach(),
            }
            self.gate_stats = stats
        else:
            stats = {}

        return gate_values, attention_bias, stats


def _mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def inject_relation_gate_to_ga_block(ga_block, relation_gate, res_feat, pair_feat,
                                      R, t, mask, generate_mask, antigen_mask, ra_emb,
                                      log_gate=False):
    """
    Modify attention logits in a GABlock to add reliability-conditioned bias.

    This function computes the modified attention logits for CDR→antigen direction.

    Args:
        ga_block: GABlock instance
        relation_gate: RelationGate instance
        res_feat: [N, L, F] residue features
        pair_feat: [N, L, L, C] pair features
        R: [N, L, 3, 3] frame basis
        t: [N, L, 3] frame coordinates
        mask: [N, L] mask
        generate_mask: [N, L] mask of generated residues (CDR)
        antigen_mask: [N, L] mask of antigen residues
        ra_emb: [N, L, ra_emb_dim] reliability embeddings
        log_gate: whether to compute gate statistics

    Returns:
        logits_sum: modified attention logits, stats dict
    """
    N, L, F = res_feat.shape
    num_heads = ga_block.num_heads

    logits_node = ga_block._node_logits(res_feat)
    logits_pair = ga_block._pair_logits(pair_feat)
    logits_spatial = ga_block._spatial_logits(R, t, res_feat)

    logits_sum = logits_node + logits_pair + logits_spatial

    if relation_gate is not None and generate_mask.any() and antigen_mask.any():
        cdr_indices = generate_mask.nonzero(as_tuple=True)
        ag_indices = antigen_mask.nonzero(as_tuple=True)

        if len(cdr_indices[0]) > 0 and len(ag_indices[0]) > 0:
            for n in range(N):
                cdr_idx = cdr_indices[1][cdr_indices[0] == n]
                ag_idx = ag_indices[1][ag_indices[0] == n]

                if len(cdr_idx) == 0 or len(ag_idx) == 0:
                    continue

                cdr_feat = res_feat[n, cdr_idx]
                ag_feat = res_feat[n, ag_idx]
                ra_emb_n = ra_emb[n, ag_idx]

                pair_feat_sub = None
                if relation_gate.use_pair_feature:
                    pair_feat_sub = pair_feat[n, cdr_idx][:, ag_idx]

                distance = None
                if relation_gate.use_distance:
                    cdr_pos = t[n, cdr_idx]
                    ag_pos = t[n, ag_idx]
                    distance = torch.cdist(cdr_pos, ag_pos)

                gate_vals, _, stats = relation_gate(
                    cdr_feat, ag_feat, ra_emb_n,
                    pair_feat=pair_feat_sub, distance=distance,
                    cdr_mask=torch.ones(len(cdr_idx), dtype=torch.bool, device=mask.device),
                    ag_mask=torch.ones(len(ag_idx), dtype=torch.bool, device=mask.device),
                    log_gate=log_gate,
                )

                gate_bias = torch.log(gate_vals.squeeze(-1) + 1e-6)

                for h in range(num_heads):
                    logits_sum[n, cdr_idx[:, None], ag_idx[None, :], h] += gate_bias

    return logits_sum