"""Soft-C pair modulation with mean-preserving reliability scaling.

Based on PairModulationFix (C version) but applies a soft reliability-dependent
scale to the update. The scale is mean-preserving (avg ≈ 1.0) so that the overall
pair update magnitude is not systematically reduced.

Key differences from C:
- Computes scalar reliability rho from U_A 8-channel features
- Constructs mean-preserving relative reliability scale
- Applies scale to the final update (not to gate, alpha, or delta)
- No alpha cap, no delta tanh bound, no timestep schedule, no update-ratio loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.trust_cdr.contact_reliability_gate import compute_contact_reliability_gate
from diffab.trust_cdr.antigen_sequence_context import AntigenSequenceContext


class PairModulationSoftReliability(nn.Module):
    """Soft-C: C's pair modulation with mean-preserving reliability scaling.

    When contact_gate_enabled=True, replaces the Soft-C reliability scale with
    a contact-aware reliability gate (see contact_reliability_gate.py).
    """

    def __init__(self, pair_feat_dim, ra_emb_dim, hidden_dim=128, dropout=0.0,
                 alpha_init=0.0, gate_bias_init=0.0,
                 use_pair_feature_in_gate=False,
                 use_distance_feature=True, use_orientation_feature=True,
                 use_reliability_in_gate=True, use_geometry_in_gate=True,
                 softc_reliability_beta=0.10,
                 contact_gate_enabled=False,
                 contact_gate_variant='contact_all',
                 contact_gate_lambda=0.10,
                 contact_gate_clip=0.20,
                 contact_gate_temperature=1.0,
                 contact_gate_cdr_lambdas=None,
                 contact_gate_normalization='contact_aware_mean_preserving',
                 use_antigen_seq_context=False,
                 ag_seq_aa_embedding_dim=16,
                 ag_seq_physchem_dim=6,
                 ag_seq_output_dim=32,
                 ag_seq_hidden_dim=128,
                 ag_seq_zero_init_last_layer=True,
                 ag_seq_disable_branch=False):
        super().__init__()
        self.pair_feat_dim = pair_feat_dim
        self.use_pair_feature_in_gate = use_pair_feature_in_gate
        self.use_distance_feature = use_distance_feature
        self.use_orientation_feature = use_orientation_feature
        self.use_reliability_in_gate = use_reliability_in_gate
        self.use_geometry_in_gate = use_geometry_in_gate
        self.softc_reliability_beta = softc_reliability_beta

        # Contact gate config
        self.contact_gate_enabled = contact_gate_enabled
        self.contact_gate_variant = contact_gate_variant
        self.contact_gate_lambda = contact_gate_lambda
        self.contact_gate_clip = contact_gate_clip
        self.contact_gate_temperature = contact_gate_temperature
        self.contact_gate_cdr_lambdas = contact_gate_cdr_lambdas
        self.contact_gate_normalization = contact_gate_normalization

        # Antigen sequence context branch config (non-invasive; default off).
        # When disabled, this module is exactly equivalent to the original Soft-C.
        self.use_antigen_seq_context = use_antigen_seq_context
        self.ag_seq_disable_branch = ag_seq_disable_branch
        self.ag_seq_zero_init_last_layer = ag_seq_zero_init_last_layer

        if use_reliability_in_gate:
            self.reliability_mlp = nn.Sequential(
                nn.Linear(ra_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.reliability_mlp[-1].weight)
            nn.init.zeros_(self.reliability_mlp[-1].bias)
        else:
            self.reliability_mlp = None

        geo_input_dim = 0
        if use_distance_feature:
            geo_input_dim += 1
        if use_orientation_feature:
            geo_input_dim += 3
        if use_geometry_in_gate and geo_input_dim > 0:
            self.geometry_mlp = nn.Sequential(
                nn.Linear(geo_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.geometry_mlp[-1].weight)
            nn.init.zeros_(self.geometry_mlp[-1].bias)
        else:
            self.geometry_mlp = None

        if use_pair_feature_in_gate:
            self.pair_mlp = nn.Sequential(
                nn.Linear(pair_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.pair_mlp[-1].weight)
            nn.init.zeros_(self.pair_mlp[-1].bias)
        else:
            self.pair_mlp = None

        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))

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

        # --- Antigen sequence context branch (non-invasive) ---
        # Only instantiated when use_antigen_seq_context=True.
        # seq_delta_mlp last layer is zero-init so seq_delta_z = 0 at start,
        # making the model output-identical to the original contact_downonly model
        # when loaded from the old checkpoint.
        self.ag_seq_context = None
        self.seq_delta_mlp = None
        if self.use_antigen_seq_context:
            self.ag_seq_context = AntigenSequenceContext(
                aa_embedding_dim=ag_seq_aa_embedding_dim,
                physchem_dim=ag_seq_physchem_dim,
                output_dim=ag_seq_output_dim,
                hidden_dim=ag_seq_hidden_dim,
                dropout=dropout,
            )
            # seq_delta_mlp: concat(pair_sub, antigen_seq_context_j) -> pair_feat_dim
            seq_in_dim = pair_feat_dim + ag_seq_output_dim
            self.seq_delta_mlp = nn.Sequential(
                nn.Linear(seq_in_dim, ag_seq_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ag_seq_hidden_dim, ag_seq_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ag_seq_hidden_dim, pair_feat_dim),
            )
            if self.ag_seq_zero_init_last_layer:
                nn.init.zeros_(self.seq_delta_mlp[-1].weight)
                nn.init.zeros_(self.seq_delta_mlp[-1].bias)
            else:
                nn.init.normal_(self.seq_delta_mlp[-1].weight, std=1e-4)
                nn.init.zeros_(self.seq_delta_mlp[-1].bias)

    def _compute_scalar_reliability(self, ra_features):
        """Compute scalar reliability rho from U_A 8-channel features.

        Channel order (confirmed from reliability.py):
            0: coordinate_u, 1: coordinate_r
            2: topology_u,   3: topology_r
            4: orientation_u, 5: orientation_r
            6: surface_u,     7: surface_r

        rho = mean([1-u_coord, r_coord, 1-u_topo, r_topo, 1-u_orient, r_orient, 1-u_surf, r_surf])
        """
        ra = ra_features  # (N, L, 8)
        device = ra.device

        rho = torch.stack([
            1.0 - ra[..., 0],  # 1 - coordinate_u
            ra[..., 1],        # coordinate_r
            1.0 - ra[..., 2],  # 1 - topology_u
            ra[..., 3],        # topology_r
            1.0 - ra[..., 4],  # 1 - orientation_u
            ra[..., 5],        # orientation_r
            1.0 - ra[..., 6],  # 1 - surface_u
            ra[..., 7],        # surface_r
        ], dim=-1).mean(dim=-1)  # (N, L)

        return rho

    def _compute_reliability_scale(self, rho, antigen_mask):
        """Compute mean-preserving relative reliability scale.

        rho_centered = rho - mean_valid(rho)
        rho_relative = rho_centered / (max_abs_valid(rho_centered) + eps)
        scale = 1.0 + beta * rho_relative

        For each antigen sample, the scale has mean ≈ 1.0 over valid residues.
        """
        N = rho.size(0)
        device = rho.device
        reliability_scale = torch.ones_like(rho)  # (N, L)

        for n in range(N):
            am = antigen_mask[n]
            if not am.any():
                continue

            rho_valid = rho[n, am]
            if rho_valid.numel() == 0:
                continue

            rho_mean = rho_valid.mean()
            rho_centered = rho_valid - rho_mean
            max_abs = rho_centered.abs().max()
            if max_abs < 1e-8:
                # All reliabilities nearly identical
                continue

            rho_relative = rho_centered / (max_abs + 1e-8)
            scale_valid = 1.0 + self.softc_reliability_beta * rho_relative
            reliability_scale[n, am] = scale_valid

        return reliability_scale.clamp(0.5, 1.5)

    def compute_gate_logits(self, ra_emb_ag, dist, direction_norm, pair_sub=None):
        L_cdr, L_ag, _ = ra_emb_ag.shape
        device = ra_emb_ag.device

        gate_logit = torch.zeros(L_cdr, L_ag, 1, device=device)

        if self.use_reliability_in_gate and self.reliability_mlp is not None:
            reliability_logit = self.reliability_mlp(ra_emb_ag)
            gate_logit = gate_logit + reliability_logit

        if self.use_geometry_in_gate and self.geometry_mlp is not None:
            geo_inputs = []
            if self.use_distance_feature:
                geo_inputs.append(dist)
            if self.use_orientation_feature and direction_norm is not None:
                geo_inputs.append(direction_norm)
            if geo_inputs:
                geo_cat = torch.cat(geo_inputs, dim=-1)
                geo_bias = self.geometry_mlp(geo_cat)
                gate_logit = gate_logit + geo_bias

        if self.use_pair_feature_in_gate and self.pair_mlp is not None and pair_sub is not None:
            pair_bias = self.pair_mlp(pair_sub)
            gate_logit = gate_logit + pair_bias

        gate_logit = gate_logit + self.gate_bias

        return gate_logit

    def _compute_valid_pair_mask(self, generate_mask, antigen_mask, cdr_idx, ag_idx):
        valid_cdr = generate_mask[cdr_idx]
        valid_ag = antigen_mask[ag_idx]
        valid_pair_mask = valid_cdr.unsqueeze(1) & valid_ag.unsqueeze(0)
        return valid_pair_mask

    def _compute_rms(self, tensor, valid_mask):
        masked = tensor[valid_mask.unsqueeze(-1).expand_as(tensor)]
        if masked.numel() == 0:
            return torch.tensor(0.0, device=tensor.device)
        return torch.sqrt((masked ** 2).mean())

    def forward(self, pair_feat, generate_mask, antigen_mask, ra_emb, pos_ca, ra_features, cdr_flag=None, aa=None,
                shuffle_antigen_aa=False, disable_seq_branch_runtime=False):
        N, L, _, C = pair_feat.shape
        device = pair_feat.device
        modified = pair_feat.clone()

        # Compute scalar reliability from ra_features
        rho = self._compute_scalar_reliability(ra_features)  # (N, L)
        reliability_scale = self._compute_reliability_scale(rho, antigen_mask)  # (N, L)

        # --- Antigen sequence context (precomputed once for the whole batch) ---
        # Only used if the branch is enabled, AA is provided, and not disabled at runtime.
        seq_branch_active = (
            self.use_antigen_seq_context
            and self.ag_seq_context is not None
            and self.seq_delta_mlp is not None
            and aa is not None
            and not disable_seq_branch_runtime
            and not self.ag_seq_disable_branch
        )
        seq_context_full = None
        if seq_branch_active:
            seq_context_full = self.ag_seq_context(
                aa, antigen_mask, shuffle_antigen_aa=shuffle_antigen_aa
            )  # [N, L, output_dim]

        # Reset contact gate stats collector
        self._contact_gate_stats = {}

        # Seq-branch stats collectors
        all_seq_delta_rms = []
        all_base_delta_rms = []
        all_final_delta_rms = []
        all_seq_to_base_ratio = []
        all_seq_context_norm = []

        all_gate_mean = []
        all_gate_std = []
        all_gate_min = []
        all_gate_max = []
        all_delta_norm = []
        all_update_norm = []
        all_delta_rms = []
        all_update_rms = []
        all_z_pair_rms = []
        all_update_to_z_ratio = []
        all_reliability_gate_corr = []
        all_mean_gate_high_rel = []
        all_mean_gate_low_rel = []
        all_valid_count = []

        # Soft-C specific stats
        all_scale_mean = []
        all_scale_std = []
        all_scale_min = []
        all_scale_max = []
        all_scale_high = []
        all_scale_low = []
        all_corr_scale_rel = []
        all_original_update_rms = []
        all_mean_scale_deviation = []

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

            valid_pair_mask = self._compute_valid_pair_mask(generate_mask[n], antigen_mask[n], cdr_idx, ag_idx)
            valid_count = valid_pair_mask.sum().float()
            all_valid_count.append(valid_count)

            ra_emb_ag = ra_emb[n, ag_idx]
            ra_emb_expanded = ra_emb_ag.unsqueeze(0).expand(L_cdr, -1, -1)

            cdr_pos = pos_ca[n, cdr_idx]
            ag_pos = pos_ca[n, ag_idx]

            dist = torch.cdist(cdr_pos, ag_pos).unsqueeze(-1)

            direction_norm = None
            if self.use_orientation_feature:
                direction = ag_pos.unsqueeze(0) - cdr_pos.unsqueeze(1)
                direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

            pair_sub = pair_feat[n, cdr_idx][:, ag_idx]

            gate_logit = self.compute_gate_logits(
                ra_emb_ag=ra_emb_expanded,
                dist=dist,
                direction_norm=direction_norm,
                pair_sub=pair_sub if self.use_pair_feature_in_gate else None,
            )
            gate = torch.sigmoid(gate_logit)

            delta_inputs = [pair_sub, ra_emb_expanded]
            if self.use_distance_feature:
                delta_inputs.append(dist)
            if self.use_orientation_feature:
                delta_inputs.append(direction_norm)

            delta_input_cat = torch.cat(delta_inputs, dim=-1)
            delta_z_base = self.delta_mlp(delta_input_cat)

            # --- Sequence-aware auxiliary residual correction branch ---
            # seq_delta_z is added to delta_z_base. Zero-init makes it 0 at start.
            if seq_branch_active:
                seq_ctx_ag = seq_context_full[n, ag_idx]                   # [L_ag, D_seq]
                seq_ctx_expanded = seq_ctx_ag.unsqueeze(0).expand(L_cdr, -1, -1)  # [L_cdr, L_ag, D_seq]
                seq_input_ij = torch.cat([pair_sub, seq_ctx_expanded], dim=-1)    # [L_cdr, L_ag, pair+D_seq]
                seq_delta_z = self.seq_delta_mlp(seq_input_ij)                    # [L_cdr, L_ag, pair_feat_dim]
                delta_z = delta_z_base + seq_delta_z

                # Stats
                all_base_delta_rms.append(self._compute_rms(delta_z_base, valid_pair_mask))
                all_seq_delta_rms.append(self._compute_rms(seq_delta_z, valid_pair_mask))
                all_final_delta_rms.append(self._compute_rms(delta_z, valid_pair_mask))
                base_rms = self._compute_rms(delta_z_base, valid_pair_mask).clamp(min=1e-12)
                seq_rms = self._compute_rms(seq_delta_z, valid_pair_mask)
                all_seq_to_base_ratio.append((seq_rms / base_rms).detach())
                all_seq_context_norm.append(seq_ctx_expanded.norm(dim=-1).mean().detach())
            else:
                delta_z = delta_z_base

            # Original C update (before reliability scaling)
            update_c = self.alpha * gate * delta_z

            # Apply reliability gate or scale
            if self.contact_gate_enabled:
                # Contact-aware reliability gate
                rho_ag = rho[n, ag_idx]  # [L_ag]
                cdr_flag_cdr = None
                if cdr_flag is not None:
                    cdr_flag_cdr = cdr_flag[n, cdr_idx]  # [L_cdr]

                w_ij, gate_stats = compute_contact_reliability_gate(
                    reliability=rho_ag,
                    pair_mask=valid_pair_mask,
                    cdr_flag=cdr_flag_cdr,
                    variant=self.contact_gate_variant,
                    lambda_value=self.contact_gate_lambda,
                    clip_value=self.contact_gate_clip,
                    temperature=self.contact_gate_temperature,
                    cdr_lambdas=self.contact_gate_cdr_lambdas,
                    normalization=self.contact_gate_normalization,
                )
                # w_ij: [L_cdr, L_ag], expand to [L_cdr, L_ag, 1]
                w_ij_expanded = w_ij.unsqueeze(-1)
                update = w_ij_expanded * update_c

                # For parity with original Soft-C stats, compute scale_ag/scale_ij
                # scale_ag: per-antigen mean gate weight (column mean of w_ij)
                scale_ag = w_ij.mean(dim=0)  # [L_ag]
                scale_ij = w_ij_expanded

                # Collect contact gate stats
                for k, v in gate_stats.items():
                    if k not in self._contact_gate_stats:
                        self._contact_gate_stats[k] = []
                    self._contact_gate_stats[k].append(v)
            else:
                # Original Soft-C mean-preserving reliability scale
                scale_ag = reliability_scale[n, ag_idx]  # (L_ag,)
                scale_ij = scale_ag.unsqueeze(0).expand(L_cdr, -1).unsqueeze(-1)  # (L_cdr, L_ag, 1)
                update = scale_ij * update_c

            modified[n, cdr_idx[:, None], ag_idx[None, :]] += update

            gate_valid = gate[valid_pair_mask]
            delta_z_valid = delta_z[valid_pair_mask]
            update_valid = update[valid_pair_mask]
            update_c_valid = update_c[valid_pair_mask]
            pair_sub_valid = pair_sub[valid_pair_mask]
            if self.contact_gate_enabled:
                scale_valid = w_ij_expanded[valid_pair_mask]
            else:
                scale_valid = scale_ij[valid_pair_mask]

            all_gate_mean.append(gate_valid.mean())
            all_gate_std.append(gate_valid.std())
            all_gate_min.append(gate_valid.min())
            all_gate_max.append(gate_valid.max())

            all_delta_norm.append(delta_z_valid.norm(dim=-1).mean())
            all_update_norm.append(update_valid.norm(dim=-1).mean())

            all_delta_rms.append(self._compute_rms(delta_z, valid_pair_mask))
            all_original_update_rms.append(self._compute_rms(update_c, valid_pair_mask))
            all_update_rms.append(self._compute_rms(update, valid_pair_mask))
            all_z_pair_rms.append(self._compute_rms(pair_sub, valid_pair_mask))
            if pair_sub_valid.numel() > 0:
                z_pair_rms_val = torch.sqrt((pair_sub_valid ** 2).mean() + 1e-8)
                update_rms_val = torch.sqrt((update_valid ** 2).mean() + 1e-8)
                ratio = update_rms_val / (z_pair_rms_val + 1e-8)
                all_update_to_z_ratio.append(ratio)
            else:
                all_update_to_z_ratio.append(torch.tensor(0.0, device=device))

            # Soft-C specific stats
            rho_ag = rho[n, ag_idx]
            all_scale_mean.append(scale_valid.mean())
            all_scale_std.append(scale_valid.std())
            all_scale_min.append(scale_valid.min())
            all_scale_max.append(scale_valid.max())
            all_mean_scale_deviation.append((scale_valid - 1.0).abs().mean())

            if rho_ag.numel() >= 3:
                k = max(1, rho_ag.numel() // 3)
                _, top_idx = torch.topk(rho_ag, k)
                _, bot_idx = torch.topk(rho_ag, k, largest=False)
                all_scale_high.append(scale_ag[top_idx].mean())
                all_scale_low.append(scale_ag[bot_idx].mean())

            if rho_ag.numel() > 1 and scale_ag.numel() > 1:
                corr = torch.corrcoef(torch.stack([rho_ag.flatten(), scale_ag.flatten()]))[0, 1]
                if not torch.isnan(corr):
                    all_corr_scale_rel.append(corr)

            # Gate-reliability correlation (same as C)
            u_mean = ra_emb_ag.mean(dim=-1)
            u_tiled = u_mean.unsqueeze(0).expand(L_cdr, -1)
            gate_flat = gate.squeeze(-1)
            u_flat = u_tiled
            gate_valid_flat = gate_flat[valid_pair_mask]
            u_valid_flat = u_flat[valid_pair_mask]
            if u_valid_flat.numel() > 1:
                corr = torch.corrcoef(torch.stack([gate_valid_flat.flatten(), u_valid_flat.flatten()]))[0, 1]
                if not torch.isnan(corr):
                    all_reliability_gate_corr.append(corr)

            u_mean_ag = u_mean
            n_ag = u_mean_ag.numel()
            if n_ag >= 3:
                k = max(1, n_ag // 3)
                _, top_indices = torch.topk(u_mean_ag, k)
                _, bottom_indices = torch.topk(u_mean_ag, k, largest=False)
                gate_full = gate.squeeze(-1)
                gate_top = gate_full[:, top_indices].mean()
                gate_bottom = gate_full[:, bottom_indices].mean()
                all_mean_gate_high_rel.append(gate_top)
                all_mean_gate_low_rel.append(gate_bottom)

        loss_dict = {}
        stats_dict = {}

        if all_gate_mean:
            stats_dict['pair_gate_mean_valid'] = torch.stack(all_gate_mean).mean().detach()
            stats_dict['pair_gate_std_valid'] = torch.stack(all_gate_std).mean().detach()
            stats_dict['pair_gate_min_valid'] = torch.stack(all_gate_min).min().detach()
            stats_dict['pair_gate_max_valid'] = torch.stack(all_gate_max).max().detach()
            stats_dict['pair_gate_valid_count'] = torch.stack(all_valid_count).mean().detach()
            stats_dict['pair_alpha'] = self.alpha.detach()
            stats_dict['pair_delta_norm'] = torch.stack(all_delta_norm).mean().detach()
            stats_dict['pair_update_norm'] = torch.stack(all_update_norm).mean().detach()
            stats_dict['pair_delta_rms'] = torch.stack(all_delta_rms).mean().detach()
            stats_dict['pair_update_rms'] = torch.stack(all_update_rms).mean().detach()
            stats_dict['z_pair_rms'] = torch.stack(all_z_pair_rms).mean().detach()
            stats_dict['pair_update_to_z_ratio'] = torch.stack(all_update_to_z_ratio).mean().detach()

            if all_reliability_gate_corr:
                stats_dict['corr_pair_gate_reliability'] = torch.stack(all_reliability_gate_corr).mean().detach()
            if all_mean_gate_high_rel and all_mean_gate_low_rel:
                stats_dict['mean_gate_high_reliability'] = torch.stack(all_mean_gate_high_rel).mean().detach()
                stats_dict['mean_gate_low_reliability'] = torch.stack(all_mean_gate_low_rel).mean().detach()
                stats_dict['mean_gate_high_minus_low'] = (
                    stats_dict['mean_gate_high_reliability'] - stats_dict['mean_gate_low_reliability']
                )

            # Soft-C specific stats
            stats_dict['softc_reliability_scale_mean_valid'] = torch.stack(all_scale_mean).mean().detach()
            stats_dict['softc_reliability_scale_std_valid'] = torch.stack(all_scale_std).mean().detach()
            stats_dict['softc_reliability_scale_min_valid'] = torch.stack(all_scale_min).min().detach()
            stats_dict['softc_reliability_scale_max_valid'] = torch.stack(all_scale_max).max().detach()
            stats_dict['softc_original_update_rms_before_scale'] = torch.stack(all_original_update_rms).mean().detach()
            stats_dict['softc_mean_scale_deviation_from_one'] = torch.stack(all_mean_scale_deviation).mean().detach()

            if all_scale_high and all_scale_low:
                stats_dict['softc_scale_high_reliability'] = torch.stack(all_scale_high).mean().detach()
                stats_dict['softc_scale_low_reliability'] = torch.stack(all_scale_low).mean().detach()
                stats_dict['softc_scale_high_minus_low'] = (
                    stats_dict['softc_scale_high_reliability'] - stats_dict['softc_scale_low_reliability']
                )
            if all_corr_scale_rel:
                stats_dict['corr_softc_scale_reliability'] = torch.stack(all_corr_scale_rel).mean().detach()

        # Add contact gate stats
        if self.contact_gate_enabled and self._contact_gate_stats:
            for k, v_list in self._contact_gate_stats.items():
                if v_list:
                    stacked = torch.stack([v.to(device) if v.device != device else v for v in v_list])
                    stats_dict[f'contact_gate_{k}'] = stacked.mean().detach()

        # Add antigen sequence context branch stats
        if seq_branch_active:
            if all_base_delta_rms:
                stats_dict['ag_seq_base_delta_rms'] = torch.stack(all_base_delta_rms).mean().detach()
            if all_seq_delta_rms:
                stats_dict['ag_seq_seq_delta_rms'] = torch.stack(all_seq_delta_rms).mean().detach()
            if all_final_delta_rms:
                stats_dict['ag_seq_final_delta_rms'] = torch.stack(all_final_delta_rms).mean().detach()
            if all_seq_to_base_ratio:
                stats_dict['ag_seq_delta_to_base_ratio'] = torch.stack(all_seq_to_base_ratio).mean().detach()
            if all_seq_context_norm:
                stats_dict['ag_seq_context_norm'] = torch.stack(all_seq_context_norm).mean().detach()
            stats_dict['ag_seq_branch_active'] = torch.tensor(1.0, device=device)
        else:
            stats_dict['ag_seq_branch_active'] = torch.tensor(0.0, device=device)

        return modified, loss_dict, stats_dict