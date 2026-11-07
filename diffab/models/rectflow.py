import torch
import torch.nn as nn
import logging

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation, random_uniform_so3
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.rectified_flow.rectflow import RectFlowGenerator
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model

import pickle

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

@register_model('rectflow')
class RectFlowAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.rectflow_seq_only = RectFlowGenerator(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

        self.use_trust_cdr = cfg.get('use_trust_cdr', False)

        if self.use_trust_cdr:
            # --- TrustCDR core modules ---
            from diffab.trust_cdr.reliability import AntigenReliabilityEstimator
            from diffab.trust_cdr.relation_gate import RelationGate, ReliabilityEmbedding
            from diffab.trust_cdr.reliability_context_encoder import (
                ReliabilityContextEncoder, ConcatReliabilityAdapter
            )
            from diffab.trust_cdr.pair_modulation import PairModulation
            from diffab.trust_cdr.pair_modulation_soft_reliability import PairModulationSoftReliability

            self.ra_reliability_estimator = AntigenReliabilityEstimator(cfg)

            self.ra_reliability_embed = ReliabilityEmbedding(
                ra_feat_dim=8,
                hidden_dim=cfg.get('ra_gate_hidden_dim', 128),
                out_dim=cfg.get('ra_gate_hidden_dim', 128),
                dropout=cfg.get('ra_gate_dropout', 0.1),
            )

            if cfg.get('ra_use_relation_gate', True):
                self.ra_relation_gate = RelationGate(
                    node_feat_dim=cfg.res_feat_dim,
                    ra_emb_dim=cfg.get('ra_gate_hidden_dim', 128),
                    gate_hidden_dim=cfg.get('ra_gate_hidden_dim', 128),
                    gate_dropout=cfg.get('ra_gate_dropout', 0.1),
                    gate_mode=cfg.get('ra_gate_mode', 'scalar'),
                    gate_init=cfg.get('ra_gate_init', 'near_identity'),
                    use_distance=cfg.get('ra_gate_use_distance', True),
                    use_pair_feature=cfg.get('ra_gate_use_pair_feature', True),
                    pair_feat_dim=cfg.pair_feat_dim,
                    num_heads=12,
                )
            else:
                self.ra_relation_gate = None

            context_mode = cfg.get('ra_context_encoder_mode', 'none')
            if context_mode == 'mp_adapter':
                self.ra_context_encoder = ReliabilityContextEncoder(
                    node_feat_dim=cfg.res_feat_dim,
                    ra_feat_dim=8,
                    hidden_dim=cfg.get('ra_context_hidden_dim', 128),
                    num_layers=cfg.get('ra_context_layers', 2),
                    k=cfg.get('ra_context_k', 16),
                    dropout=cfg.get('ra_context_dropout', 0.1),
                )
            elif context_mode == 'concat':
                self.ra_context_encoder = ConcatReliabilityAdapter(
                    node_feat_dim=cfg.res_feat_dim,
                    ra_feat_dim=8,
                    hidden_dim=cfg.get('ra_context_hidden_dim', 128),
                )
            else:
                self.ra_context_encoder = None

            self.ra_context_encoder_mode = context_mode
            self.ra_compute_online = cfg.get('ra_compute_online_if_missing', False)

            if cfg.get('ra_use_denoising_head', False):
                from diffab.trust_cdr.reliability_context_encoder import MaskedReliabilityDenoisingHead
                self.ra_denoising_head = MaskedReliabilityDenoisingHead(
                    hidden_dim=cfg.get('ra_context_hidden_dim', 128),
                    ra_feat_dim=8,
                    mask_ratio=cfg.get('ra_denoising_mask_ratio', 0.15),
                    dropout=cfg.get('ra_context_dropout', 0.1),
                )
                self.ra_denoising_loss_weight = cfg.get('ra_denoising_loss_weight', 0.01)
            else:
                self.ra_denoising_head = None
                self.ra_denoising_loss_weight = 0.0

            # Pair modulation: only SoftReliability (contact_all_l0.10) and base PairModulation
            self.use_pair_soft_reliability = cfg.get('use_pair_soft_reliability', False)

            if cfg.get('ra_use_pair_modulation', False):
                pm_cfg = cfg.get('ra_pair_modulation', {})
                if self.use_pair_soft_reliability:
                    # Soft-C: C + mean-preserving reliability scale + contact gate
                    softc_beta = cfg.get('softc_reliability_beta', 0.10)
                    ag_seq_cfg = cfg.get('antigen_seq_context', {}) or {}
                    self.ra_pair_modulation = PairModulationSoftReliability(
                        pair_feat_dim=cfg.pair_feat_dim,
                        ra_emb_dim=pm_cfg.get('hidden_dim', 128),
                        hidden_dim=pm_cfg.get('hidden_dim', 128),
                        dropout=pm_cfg.get('dropout', 0.0),
                        alpha_init=pm_cfg.get('alpha_init', 0.0),
                        gate_bias_init=pm_cfg.get('gate_bias_init', 0.0),
                        use_pair_feature_in_gate=pm_cfg.get('use_pair_feature_in_gate', False),
                        use_distance_feature=pm_cfg.get('use_distance_feature', True),
                        use_orientation_feature=pm_cfg.get('use_orientation_feature', True),
                        use_reliability_in_gate=pm_cfg.get('pair_gate_use_reliability', True),
                        use_geometry_in_gate=pm_cfg.get('pair_gate_use_geometry', True),
                        softc_reliability_beta=softc_beta,
                        contact_gate_enabled=cfg.get('direct_contact_gate', {}).get('enabled', False),
                        contact_gate_variant=cfg.get('direct_contact_gate', {}).get('variant', 'contact_all'),
                        contact_gate_lambda=cfg.get('direct_contact_gate', {}).get('lambda', 0.10),
                        contact_gate_clip=cfg.get('direct_contact_gate', {}).get('clip_value', 0.20),
                        contact_gate_temperature=cfg.get('direct_contact_gate', {}).get('temperature', 1.0),
                        contact_gate_cdr_lambdas=cfg.get('cdr_lambdas', None),
                        contact_gate_normalization=cfg.get('direct_contact_gate', {}).get('normalization', 'contact_aware_mean_preserving'),
                        use_antigen_seq_context=cfg.get('use_antigen_seq_context', False) and ag_seq_cfg.get('enabled', True),
                        ag_seq_aa_embedding_dim=ag_seq_cfg.get('aa_embedding_dim', 16),
                        ag_seq_physchem_dim=ag_seq_cfg.get('physchem_dim', 6),
                        ag_seq_output_dim=ag_seq_cfg.get('output_dim', 32),
                        ag_seq_hidden_dim=ag_seq_cfg.get('seq_delta_hidden_dim', 128),
                        ag_seq_zero_init_last_layer=ag_seq_cfg.get('zero_init_last_layer', True),
                        ag_seq_disable_branch=ag_seq_cfg.get('disable_branch', False),
                    )
                    self.ag_seq_shuffle_antigen_aa = False
                    self.ag_seq_disable_branch_runtime = False
                else:
                    # Base PairModulation (no reliability correction)
                    self.ra_pair_modulation = PairModulation(
                        pair_feat_dim=cfg.pair_feat_dim,
                        ra_emb_dim=pm_cfg.get('hidden_dim', 128),
                        hidden_dim=pm_cfg.get('hidden_dim', 128),
                        dropout=pm_cfg.get('dropout', 0.0),
                        alpha_init=pm_cfg.get('alpha_init', 0.0),
                        gate_bias_init=pm_cfg.get('gate_bias_init', 2.0),
                        use_pair_feature_in_gate=pm_cfg.get('use_pair_feature_in_gate', False),
                        use_distance_feature=pm_cfg.get('use_distance_feature', True),
                        use_orientation_feature=pm_cfg.get('use_orientation_feature', True),
                    )
            else:
                self.ra_pair_modulation = None
                self.use_pair_soft_reliability = False
        else:
            self.ra_reliability_estimator = None
            self.ra_reliability_embed = None
            self.ra_relation_gate = None
            self.ra_context_encoder = None
            self.ra_context_encoder_mode = 'none'
            self.ra_compute_online = False
            self.ra_denoising_head = None
            self.ra_denoising_loss_weight = 0.0
            self.ra_pair_modulation = None
            self.use_pair_soft_reliability = False

    def _get_antigen_reliability(self, batch):
        if 'antigen_reliability' in batch and batch['antigen_reliability'] is not None:
            ra_features = batch['antigen_reliability']
            if ra_features.dim() == 3 and ra_features.size(-1) == 8:
                return ra_features
            elif ra_features.dim() == 2 and ra_features.size(-1) == 8:
                return ra_features.unsqueeze(0)
        return None

    def _compute_antigen_reliability_online(self, batch):
        if not self.use_trust_cdr or not self.ra_compute_online:
            return None
        pos_heavyatom = batch['pos_heavyatom']
        mask_heavyatom = batch['mask_heavyatom']
        fragment_type = batch['fragment_type']

        N, L = pos_heavyatom.shape[:2]
        device = pos_heavyatom.device

        antigen_mask = (fragment_type == 3)

        if not antigen_mask.any():
            return None

        ra_features = torch.zeros(N, L, 8, device=device)
        for n in range(N):
            am = antigen_mask[n]
            if am.any():
                pos_ca = pos_heavyatom[n, :, BBHeavyAtom.CA]
                feat, _ = self.ra_reliability_estimator(pos_ca, am)
                ra_features[n] = feat

        return ra_features

    def encode(self, batch, remove_structure, remove_sequence):
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_flag']
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            fragment_type=batch['fragment_type'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )

        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        if 'template' in batch and batch['template'] is not None:
            template = batch['template']
            templateR = construct_3d_basis(
                template['pos_heavyatom'][:, :, BBHeavyAtom.CA],
                template['pos_heavyatom'][:, :, BBHeavyAtom.C],
                template['pos_heavyatom'][:, :, BBHeavyAtom.N],
            )
            templatep = template['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        else:
            templateR = torch.zeros_like(R)
            templatep = torch.zeros_like(p)

        if self.use_trust_cdr:
            ra_features = self._get_antigen_reliability(batch)
            if ra_features is None and self.ra_compute_online:
                ra_features = self._compute_antigen_reliability_online(batch)

            if self.use_pair_soft_reliability and ra_features is None:
                has_antigen = (batch.get('fragment_type') == 3).any() if 'fragment_type' in batch else False
                if has_antigen:
                    batch_ids = batch.get('id', ['<unknown>'] * has_antigen.size(0) if has_antigen.dim() > 0 else ['<unknown>'])
                    ag_per_sample = (batch.get('fragment_type') == 3).any(dim=1) if 'fragment_type' in batch and batch.get('fragment_type').dim() > 1 else has_antigen
                    missing_ids = [batch_ids[i] for i in range(len(batch_ids)) if ag_per_sample[i].item()] if isinstance(batch_ids, (list, tuple)) else []
                    raise RuntimeError(
                        'use_pair_soft_reliability=true but antigen_reliability is missing for samples with antigens. '
                        f'Batch contains {len(batch_ids)} samples. '
                        f'Samples missing reliability: {missing_ids}. '
                        'Ensure antigen_reliability_cache_dir is configured correctly and cache files exist.'
                    )

            if ra_features is not None:
                generate_mask = batch['generate_flag']
                antigen_mask = (batch['fragment_type'] == 3)

                ra_emb = self.ra_reliability_embed(ra_features)

                if self.ra_context_encoder is not None and self.ra_context_encoder_mode == 'concat':
                    ag_feat_for_ctx = res_feat * antigen_mask.unsqueeze(-1).float()
                    ag_feat_ctx = self.ra_context_encoder(ag_feat_for_ctx, ra_features)
                    res_feat = torch.where(antigen_mask.unsqueeze(-1), ag_feat_ctx, res_feat)
                elif self.ra_context_encoder is not None and self.ra_context_encoder_mode == 'mp_adapter':
                    ag_coords = p * antigen_mask.unsqueeze(-1).float()
                    ag_feat_for_ctx = res_feat * antigen_mask.unsqueeze(-1).float()
                    ag_feat_ctx, u_ctx = self.ra_context_encoder(
                        ag_feat_for_ctx, ra_features, ag_coords, antigen_mask
                    )
                    res_feat = torch.where(antigen_mask.unsqueeze(-1), ag_feat_ctx, res_feat)
                    ra_emb = u_ctx
            else:
                ra_features = None
                ra_emb = None
        else:
                ra_features = None
                ra_emb = None

        pair_modulation_loss = {}
        pair_modulation_stats = {}
        if self.use_trust_cdr and self.ra_pair_modulation is not None and ra_features is not None:
            generate_mask = batch['generate_flag']
            antigen_mask = (batch['fragment_type'] == 3)
            if generate_mask.any() and antigen_mask.any():
                # Replace native CDR CA with noisy interpolated positions to prevent
                # native CDR-antigen geometry leakage into pair features.
                pos_ca_native = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
                pos_ca_current = pos_ca_native.clone()
                t_rand = torch.rand(1, device=pos_ca_current.device)
                p_noise = torch.randn_like(pos_ca_native)
                p_interp = (1 - t_rand) * p_noise + t_rand * pos_ca_native
                gm_expanded = generate_mask.unsqueeze(-1).expand_as(pos_ca_current)
                pos_ca_current = torch.where(gm_expanded, p_interp, pos_ca_current)

                if self.use_pair_soft_reliability:
                    # Soft-C: C + mean-preserving reliability scale + contact gate
                    cdr_flag = batch.get('cdr_flag', None)
                    ag_aa = batch.get('aa', None) if getattr(self, 'use_antigen_seq_context', False) or getattr(self, 'ag_seq_shuffle_antigen_aa', False) else None
                    result = self.ra_pair_modulation(
                        pair_feat,
                        generate_mask=generate_mask,
                        antigen_mask=antigen_mask,
                        ra_emb=ra_emb,
                        pos_ca=pos_ca_current,
                        ra_features=ra_features,
                        cdr_flag=cdr_flag,
                        aa=ag_aa,
                        shuffle_antigen_aa=getattr(self, 'ag_seq_shuffle_antigen_aa', False),
                        disable_seq_branch_runtime=getattr(self, 'ag_seq_disable_branch_runtime', False),
                    )
                    pair_feat, pair_modulation_loss, pair_modulation_stats = result
                else:
                    # Base PairModulation
                    result = self.ra_pair_modulation(
                        pair_feat,
                        generate_mask=generate_mask,
                        antigen_mask=antigen_mask,
                        ra_emb=ra_emb,
                        pos_ca=pos_ca_current,
                    )
                    pair_feat, pair_modulation_stats = result

        return res_feat, pair_feat, R, p, templateR, templatep, ra_features, ra_emb, pair_modulation_loss, pair_modulation_stats

    def forward(self, batch, return_velocities=False):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        mask_anchor = batch['anchor_flag']
        res_feat, pair_feat, R_0, p_0, R_template, p_template, ra_features, ra_emb, pair_modulation_loss, pair_modulation_stats = self.encode(
            batch,
            remove_structure=self.cfg.get('train_structure', True),
            remove_sequence=self.cfg.get('train_sequence', True)
        )

        s_0 = batch['aa']

        if 'template' in batch and batch['template'] is not None:
            template = batch['template']
            s_template = template['aa']
            mask_template_generate = template['generate_flag']
            template_enable = batch['template_enable']
            template_mask = torch.rand(template_enable.shape, device=template_enable.device) < 0.3
            template_enable = template_enable & template_mask
        else:
            s_template = s_0.clone()
            mask_template_generate = torch.zeros_like(batch['generate_flag'])
            template_enable = torch.zeros(mask_generate.shape, device=mask_generate.device, dtype=torch.bool)

        pdbid = batch['id']

        antigen_mask = (batch['fragment_type'] == 3)

        result = self.rectflow_seq_only(
            R_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            mask_anchor, R_template, p_template, s_template,
            mask_template_generate,
            denoise_structure=self.cfg.get('train_structure', True),
            denoise_sequence=self.cfg.get('train_sequence', True),
            template_enable=template_enable,
            pdbid=pdbid,
            generate_mask=mask_generate if self.use_trust_cdr else None,
            antigen_mask=antigen_mask if self.use_trust_cdr else None,
            ra_emb=ra_emb if self.use_trust_cdr else None,
            relation_gate=self.ra_relation_gate if self.use_trust_cdr else None,
            return_velocities=return_velocities,
        )

        if return_velocities:
            loss_dict, predictions = result
        else:
            loss_dict = result

        if self.use_trust_cdr and self.ra_denoising_head is not None and ra_features is not None:
            if self.ra_context_encoder_mode == 'mp_adapter':
                ag_coords = p_0 * antigen_mask.unsqueeze(-1).float()
                denoise_mask, ra_masked = self.ra_denoising_head.create_mask(ra_features, antigen_mask)
                u_ctx_denoise = self.ra_context_encoder.compute_u_ctx(ra_masked, ag_coords, antigen_mask)
                denoising_loss, _ = self.ra_denoising_head(
                    u_ctx_denoise, ra_features, denoise_mask, antigen_mask
                )
                loss_dict['ra_denoising_loss'] = denoising_loss * self.ra_denoising_loss_weight

        if pair_modulation_loss:
            for k, v in pair_modulation_loss.items():
                loss_dict[k] = v

        if pair_modulation_stats:
            for k, v in pair_modulation_stats.items():
                if isinstance(v, (int, float)):
                    loss_dict[k] = v
                elif hasattr(v, 'item'):
                    loss_dict[k] = v

        if return_velocities:
            return loss_dict, predictions

        return loss_dict

    @torch.no_grad()
    def sample(
        self,
        batch,
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        mask_anchor = batch['anchor_flag']
        res_feat, pair_feat, R_0, p_0, R_template, p_template, ra_features, ra_emb, _, _ = self.encode(
            batch,
            remove_structure=sample_opt.get('sample_structure', True),
            remove_sequence=sample_opt.get('sample_sequence', True)
        )
        s_0 = batch['aa']

        if 'template' in batch and batch['template'] is not None:
            template = batch['template']
            s_template = template['aa']
            mask_template_generate = template['generate_flag']
        else:
            s_template = s_0.clone()
            mask_template_generate = torch.zeros_like(mask_generate)

        template_enable = 'template' in batch and batch['template'] is not None

        antigen_mask = (batch['fragment_type'] == 3)

        traj = self.rectflow_seq_only.sample(R_0, p_0, s_0, res_feat,
                                              pair_feat, mask_generate,
                                              mask_res, mask_anchor,
                                              template_enable=template_enable,
                                              R_template=R_template,
                                              p_template=p_template,
                                              s_template=s_template,
                                              mask_template_generate=mask_template_generate,
                                              generate_mask=mask_generate if self.use_trust_cdr else None,
                                              antigen_mask=antigen_mask if self.use_trust_cdr else None,
                                              ra_emb=ra_emb if self.use_trust_cdr else None,
                                              relation_gate=self.ra_relation_gate if self.use_trust_cdr else None,
                                              **sample_opt)
        return traj
