import torch
import torch.nn as nn
import torch.nn.functional as F


class PairModulation(nn.Module):

    def __init__(self, pair_feat_dim, ra_emb_dim, hidden_dim=128, dropout=0.0,
                 alpha_init=0.0, gate_bias_init=2.0, use_pair_feature_in_gate=False,
                 use_distance_feature=True, use_orientation_feature=True):
        super().__init__()
        self.pair_feat_dim = pair_feat_dim
        self.use_pair_feature_in_gate = use_pair_feature_in_gate
        self.use_distance_feature = use_distance_feature
        self.use_orientation_feature = use_orientation_feature

        gate_input_dim = ra_emb_dim
        if use_distance_feature:
            gate_input_dim += 1
        if use_orientation_feature:
            gate_input_dim += 3
        if use_pair_feature_in_gate:
            gate_input_dim += pair_feat_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

        delta_input_dim = pair_feat_dim + ra_emb_dim
        if use_distance_feature:
            delta_input_dim += 1
        if use_orientation_feature:
            delta_input_dim += 3

        self.delta_mlp = nn.Sequential(
            nn.Linear(delta_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_feat_dim),
        )

        nn.init.normal_(self.delta_mlp[-1].weight, std=1e-4)
        nn.init.zeros_(self.delta_mlp[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self.stats = {}

    def forward(self, pair_feat, generate_mask, antigen_mask, ra_emb, pos_ca):
        """
        Args:
            pair_feat: [N, L, L, C]
            generate_mask: [N, L] CDR mask
            antigen_mask: [N, L] antigen mask
            ra_emb: [N, L, D]
            pos_ca: [N, L, 3]
        Returns:
            pair_feat: [N, L, L, C]
            stats: dict
        """
        N, L, _, C = pair_feat.shape
        device = pair_feat.device
        modified = pair_feat.clone()

        all_gate_mean = []
        all_gate_std = []
        all_gate_min = []
        all_gate_max = []
        all_delta_norm = []
        all_update_norm = []

        for n in range(N):
            gm = generate_mask[n]
            am = antigen_mask[n]
            if not gm.any() or not am.any():
                continue

            cdr_idx = gm.nonzero(as_tuple=False).squeeze(-1)
            ag_idx = am.nonzero(as_tuple=False).squeeze(-1)

            if len(cdr_idx) == 0 or len(ag_idx) == 0:
                continue

            L_cdr = len(cdr_idx)
            L_ag = len(ag_idx)

            # --- build gate inputs ---
            gate_inputs = []

            ra_emb_ag = ra_emb[n, ag_idx]
            ra_emb_expanded = ra_emb_ag.unsqueeze(0).expand(L_cdr, -1, -1)
            gate_inputs.append(ra_emb_expanded)

            cdr_pos = pos_ca[n, cdr_idx]
            ag_pos = pos_ca[n, ag_idx]

            dist = torch.cdist(cdr_pos, ag_pos).unsqueeze(-1)

            if self.use_distance_feature:
                gate_inputs.append(dist)

            direction_norm = None
            if self.use_orientation_feature:
                direction = ag_pos.unsqueeze(0) - cdr_pos.unsqueeze(1)
                direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                gate_inputs.append(direction_norm)

            if self.use_pair_feature_in_gate:
                pair_sub = pair_feat[n, cdr_idx][:, ag_idx]
                gate_inputs.append(pair_sub)

            gate_input_cat = torch.cat(gate_inputs, dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input_cat))

            # --- build delta inputs ---
            delta_inputs = []

            pair_sub = pair_feat[n, cdr_idx][:, ag_idx]
            delta_inputs.append(pair_sub)
            delta_inputs.append(ra_emb_expanded)

            if self.use_distance_feature:
                delta_inputs.append(dist)
            if self.use_orientation_feature:
                delta_inputs.append(direction_norm)

            delta_input_cat = torch.cat(delta_inputs, dim=-1)
            delta_z = self.delta_mlp(delta_input_cat)

            # --- residual safe injection ---
            update = self.alpha * gate * delta_z

            modified[n, cdr_idx[:, None], ag_idx[None, :]] += update

            # --- stats ---
            all_gate_mean.append(gate.mean())
            all_gate_std.append(gate.std())
            all_gate_min.append(gate.min())
            all_gate_max.append(gate.max())
            all_delta_norm.append(delta_z.norm(dim=-1).mean())
            all_update_norm.append(update.norm(dim=-1).mean())

        stats = {}
        if all_gate_mean:
            stats['pair_gate_mean'] = torch.stack(all_gate_mean).mean().detach()
            stats['pair_gate_std'] = torch.stack(all_gate_std).mean().detach()
            stats['pair_gate_min'] = torch.stack(all_gate_min).min().detach()
            stats['pair_gate_max'] = torch.stack(all_gate_max).max().detach()
            stats['pair_alpha'] = self.alpha.detach()
            stats['pair_delta_norm'] = torch.stack(all_delta_norm).mean().detach()
            stats['pair_update_norm'] = torch.stack(all_update_norm).mean().detach()

        self.stats = stats
        return modified, stats