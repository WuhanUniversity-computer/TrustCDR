import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from diffab.modules.common.geometry import construct_3d_basis
from diffab.utils.protein.constants import BBHeavyAtom


class AntigenReliabilityEstimator(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        self.num_ensemble = getattr(cfg, 'ra_num_ensemble', 8) if cfg else 8
        self.perturb_sigma = getattr(cfg, 'ra_perturb_sigma', 0.3) if cfg else 0.3
        self.smooth_window = getattr(cfg, 'ra_smooth_window', 3) if cfg else 3
        self.max_perturb = getattr(cfg, 'ra_max_perturb', 1.5) if cfg else 1.5
        self.topology_k = getattr(cfg, 'ra_topology_k', 16) if cfg else 16
        self.surface_cutoff = getattr(cfg, 'ra_surface_cutoff', 10.0) if cfg else 10.0
        self.reliability_alpha = getattr(cfg, 'ra_reliability_alpha', 1.0) if cfg else 1.0
        self.shuffle_reliability = getattr(cfg, 'ra_shuffle_reliability', False) if cfg else False

        channels = getattr(cfg, 'ra_reliability_channels', None) if cfg else None
        if channels is None:
            self.active_channels = {'coordinate', 'topology', 'orientation', 'surface'}
        else:
            self.active_channels = set(channels)

    def _local_smooth_perturbation(self, pos_ca):
        """
        pos_ca: [L, 3] antigen C-alpha coordinates in Angstrom
        Returns: [K, L, 3] ensemble of perturbed coordinates
        """
        L = pos_ca.shape[0]
        device = pos_ca.device
        K = self.num_ensemble

        # Use a local Generator seeded by a hash of the input coordinates.
        # This ensures the perturbation is deterministic regardless of global RNG state.
        coord_hash = (pos_ca * 1000).round().to(torch.int64)
        seed = (coord_hash[:, 0].sum() * 73856093 ^
                coord_hash[:, 1].sum() * 19349663 ^
                coord_hash[:, 2].sum() * 83492791) & 0x7fffffff
        g = torch.Generator(device=device)
        g.manual_seed(seed.item())

        raw_noise = torch.randn(K, L, 3, device=device, generator=g) * self.perturb_sigma

        if self.smooth_window > 1:
            window = self.smooth_window
            padding = window // 2
            smooth_kernel = torch.ones(1, 1, window, device=device) / window
            noise_smooth = raw_noise.clone()
            for d in range(3):
                noise_chan = raw_noise[:, :, d].unsqueeze(1)
                noise_chan_padded = F.pad(noise_chan, (padding, padding), mode='replicate')
                noise_smooth[:, :, d] = F.conv1d(noise_chan_padded, smooth_kernel, padding=0).squeeze(1)
            raw_noise = noise_smooth

        noise_magnitude = raw_noise.norm(dim=-1, keepdim=True)
        scale = torch.clamp(noise_magnitude / self.max_perturb, max=1.0)
        raw_noise = raw_noise * scale

        perturbed = pos_ca.unsqueeze(0) + raw_noise * 0.1

        return perturbed

    def _compute_rmsf(self, ensemble, pos_ca):
        """
        ensemble: [K, L, 3]
        pos_ca: [L, 3]
        Returns: [L] coordinate uncertainty (RMSF-like)
        """
        deviations = (ensemble - pos_ca.unsqueeze(0)).norm(dim=-1)
        rmsf = deviations.mean(dim=0)
        return rmsf

    def _compute_topology_uncertainty(self, ensemble, pos_ca):
        """
        ensemble: [K, L, 3]
        pos_ca: [L, 3]
        Returns: [L] topology uncertainty (1 - mean Jaccard similarity of kNN sets)
        """
        K, L, _ = ensemble.shape
        device = pos_ca.device
        k = min(self.topology_k, L - 1)

        dist_ref = torch.cdist(pos_ca.unsqueeze(0), pos_ca.unsqueeze(0)).squeeze(0)
        _, nn_ref_idx = dist_ref.topk(k + 1, dim=-1, largest=False)
        nn_ref = nn_ref_idx[:, 1:]

        consistencies = []
        for i in range(K):
            dist_pert = torch.cdist(ensemble[i], ensemble[i])
            _, nn_pert_idx = dist_pert.topk(k + 1, dim=-1, largest=False)
            nn_pert = nn_pert_idx[:, 1:]

            nn_ref_set = [set(nn_ref[j].tolist()) for j in range(L)]
            nn_pert_set = [set(nn_pert[j].tolist()) for j in range(L)]

            jaccards = []
            for j in range(L):
                ref_set = nn_ref_set[j]
                pert_set = nn_pert_set[j]
                intersection = len(ref_set & pert_set)
                union = len(ref_set | pert_set)
                if union > 0:
                    jaccards.append(intersection / union)
                else:
                    jaccards.append(1.0)
            consistencies.append(torch.tensor(jaccards, device=device))

        mean_consistency = torch.stack(consistencies).mean(dim=0)
        topology_uncertainty = 1.0 - mean_consistency
        return topology_uncertainty

    def _compute_orientation_uncertainty(self, ensemble, pos_ca):
        """
        ensemble: [K, L, 3]
        pos_ca: [L, 3]
        Returns: [L] orientation uncertainty
        """
        K, L, _ = ensemble.shape
        device = pos_ca.device

        def local_frame(coords, i):
            if i == 0 or i == L - 1:
                return torch.eye(3, device=device)
            n_vec = coords[i] - coords[i - 1]
            c_vec = coords[i + 1] - coords[i]
            n_vec_n = n_vec / (n_vec.norm() + 1e-8)
            c_vec_n = c_vec / (c_vec.norm() + 1e-8)
            z_axis = (n_vec_n + c_vec_n) / ((n_vec_n + c_vec_n).norm() + 1e-8)
            x_axis = torch.cross(n_vec_n, c_vec_n)
            x_axis = x_axis / (x_axis.norm() + 1e-8)
            y_axis = torch.cross(z_axis, x_axis)
            return torch.stack([x_axis, y_axis, z_axis], dim=0)

        R_ref = torch.stack([local_frame(pos_ca, i) for i in range(L)], dim=0)

        orientation_devs = torch.zeros(L, device=device)
        for k in range(K):
            R_pert = torch.stack([local_frame(ensemble[k], i) for i in range(L)], dim=0)
            R_diff = torch.bmm(R_pert, R_ref.transpose(1, 2))
            trace = R_diff.diagonal(dim1=1, dim2=2).sum(dim=-1)
            cos_angle = (trace - 1) / 2
            cos_angle = torch.clamp(cos_angle, -1 + 1e-8, 1 - 1e-8)
            angle = torch.acos(cos_angle)
            orientation_devs += angle

        orientation_devs /= K
        return orientation_devs

    def _compute_surface_uncertainty(self, ensemble, pos_ca):
        """
        ensemble: [K, L, 3]
        pos_ca: [L, 3]
        Returns: [L] surface uncertainty (exposure variation)
        """
        K, L, _ = ensemble.shape
        device = pos_ca.device
        cutoff = self.surface_cutoff

        def exposure(coords):
            dist = torch.cdist(coords, coords)
            neighbor_count = (dist < cutoff).sum(dim=-1).float() - 1
            max_neighbors = neighbor_count.max().clamp(min=1)
            exposure_score = 1.0 - neighbor_count / max_neighbors
            return exposure_score

        exposures = torch.stack([exposure(ensemble[k]) for k in range(K)], dim=0)
        exposure_std = exposures.std(dim=0)
        return exposure_std

    def _normalize_channels(self, uncertainties):
        """
        uncertainties: Dict[str, Tensor[L]]
        Returns: Dict with raw and normalized uncertainty + reliability
        """
        result = {}
        for name, raw in uncertainties.items():
            median = raw.median()
            iqr = torch.quantile(raw, 0.75) - torch.quantile(raw, 0.25)
            iqr = iqr.clamp(min=1e-6)
            normalized = (raw - median) / iqr
            normalized = torch.clamp(normalized, -3.0, 3.0)

            norm_min = normalized.min()
            norm_max = normalized.max()
            if norm_max - norm_min > 1e-8:
                normalized_01 = (normalized - norm_min) / (norm_max - norm_min)
            else:
                normalized_01 = torch.zeros_like(normalized)

            reliability = torch.exp(-self.reliability_alpha * normalized_01)
            reliability = torch.clamp(reliability, 0.01, 0.99)

            result[f'{name}_uncertainty_raw'] = raw
            result[f'{name}_uncertainty_norm'] = normalized_01
            result[f'{name}_reliability'] = reliability

        return result

    def forward(self, pos_ca, antigen_mask=None):
        """
        Args:
            pos_ca: [L, 3] antigen C-alpha coordinates
            antigen_mask: [L] boolean mask for antigen residues
        Returns:
            antigen_reliability: [L, 8] tensor with 4 uncertainty + 4 reliability channels
            metadata: dict with per-channel info
        """
        if antigen_mask is None:
            antigen_mask = torch.ones(pos_ca.shape[0], dtype=torch.bool, device=pos_ca.device)

        L_ag = antigen_mask.sum().item()
        if L_ag == 0:
            return torch.zeros(pos_ca.shape[0], 8, device=pos_ca.device), {}

        pos_ca_ag = pos_ca[antigen_mask]

        ensemble = self._local_smooth_perturbation(pos_ca_ag)

        uncertainties = {}

        if 'coordinate' in self.active_channels:
            uncertainties['coordinate'] = self._compute_rmsf(ensemble, pos_ca_ag)

        if 'topology' in self.active_channels:
            uncertainties['topology'] = self._compute_topology_uncertainty(ensemble, pos_ca_ag)

        if 'orientation' in self.active_channels:
            uncertainties['orientation'] = self._compute_orientation_uncertainty(ensemble, pos_ca_ag)

        if 'surface' in self.active_channels:
            uncertainties['surface'] = self._compute_surface_uncertainty(ensemble, pos_ca_ag)

        normalized = self._normalize_channels(uncertainties)

        channel_order = ['coordinate', 'topology', 'orientation', 'surface']
        features = []
        for ch in channel_order:
            if ch in self.active_channels:
                features.append(normalized[f'{ch}_uncertainty_norm'])
                features.append(normalized[f'{ch}_reliability'])
            else:
                features.append(torch.zeros(L_ag, device=pos_ca_ag.device))
                features.append(torch.ones(L_ag, device=pos_ca_ag.device))

        ra_features_ag = torch.stack(features, dim=-1)

        if self.shuffle_reliability:
            idx = torch.randperm(L_ag, device=ra_features_ag.device)
            ra_features_ag = ra_features_ag[idx]

        full_features = torch.zeros(pos_ca.shape[0], 8, device=pos_ca.device)
        full_features[antigen_mask] = ra_features_ag

        metadata = {
            'ensemble_size': self.num_ensemble,
            'antigen_length': L_ag,
            'active_channels': list(self.active_channels),
            'shuffle_reliability': self.shuffle_reliability,
        }

        return full_features, metadata


def compute_antigen_reliability(pos_heavyatom, mask_heavyatom, chain_nb, cfg=None):
    """
    Convenience function to compute antigen reliability from full batch tensors.

    Args:
        pos_heavyatom: [N, A, 3] or [N, L, A, 3]
        mask_heavyatom: [N, A] or [N, L, A]
        chain_nb: [N] or [N, L]
        cfg: RA config
    Returns:
        antigen_reliability: [L, 8] or [N, L, 8]
        metadata: dict
    """
    if cfg is None:
        estimator = AntigenReliabilityEstimator()
    else:
        estimator = AntigenReliabilityEstimator(cfg)

    if pos_heavyatom.dim() == 3:
        pos_ca = pos_heavyatom[:, BBHeavyAtom.CA]
        antigen_mask = mask_heavyatom[:, BBHeavyAtom.CA]
        return estimator(pos_ca, antigen_mask)
    else:
        N, L, A, _ = pos_heavyatom.shape
        all_features = []
        all_meta = []
        for n in range(N):
            pos_ca = pos_heavyatom[n, :, BBHeavyAtom.CA]
            antigen_mask = mask_heavyatom[n, :, BBHeavyAtom.CA]
            feat, meta = estimator(pos_ca, antigen_mask)
            all_features.append(feat)
            all_meta.append(meta)
        return torch.stack(all_features, dim=0), all_meta