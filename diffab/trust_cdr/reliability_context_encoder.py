import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class ReliabilityContextEncoder(nn.Module):

    def __init__(self, node_feat_dim=128, ra_feat_dim=8, hidden_dim=128,
                 num_layers=2, k=16, dropout=0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k = k

        self.ra_proj = nn.Linear(ra_feat_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MessagePassingAdapter(hidden_dim, k, dropout))

        self.res_proj = nn.Linear(node_feat_dim + hidden_dim, node_feat_dim)
        self.layer_norm = nn.LayerNorm(node_feat_dim)

    def _build_knn_graph(self, coords, mask):
        """
        Args:
            coords: [B, L, 3]
            mask: [B, L]
        Returns:
            edge_index: [B, L, k] indices of k nearest neighbors
            edge_dist: [B, L, k] distances
        """
        B, L, _ = coords.shape
        k = min(self.k, L - 1)
        device = coords.device

        dist = torch.cdist(coords, coords)
        dist = dist + (~mask.unsqueeze(-1) | ~mask.unsqueeze(-2)).float() * 1e10

        _, edge_index = dist.topk(k + 1, dim=-1, largest=False)
        edge_index = edge_index[:, :, 1:]

        edge_dist = torch.gather(dist, 2, edge_index)

        return edge_index, edge_dist

    def forward(self, h_ag, ra_features, ag_coords, ag_mask):
        """
        Args:
            h_ag: [B, L_ag, node_feat_dim] antigen residue features
            ra_features: [B, L_ag, 8] raw U_A features
            ag_coords: [B, L_ag, 3] antigen C-alpha coordinates
            ag_mask: [B, L_ag] antigen mask
        Returns:
            h_ag_ctx: [B, L_ag, node_feat_dim] contextualized antigen features
            u_ctx: [B, L_ag, hidden_dim] contextualized reliability embedding
        """
        u_ctx = self.compute_u_ctx(ra_features, ag_coords, ag_mask)

        h_ag_ctx = self.res_proj(torch.cat([h_ag, u_ctx], dim=-1))
        h_ag_ctx = self.layer_norm(h_ag + h_ag_ctx)
        h_ag_ctx = h_ag_ctx * ag_mask.unsqueeze(-1).float()

        return h_ag_ctx, u_ctx

    def compute_u_ctx(self, ra_features, ag_coords, ag_mask):
        """
        Compute contextualized reliability embeddings only (without updating h_ag).
        Used by the denoising head.

        Args:
            ra_features: [B, L, 8]
            ag_coords: [B, L, 3]
            ag_mask: [B, L]
        Returns:
            u_ctx: [B, L, hidden_dim]
        """
        ra_emb = self.ra_proj(ra_features)

        edge_index, edge_dist = self._build_knn_graph(ag_coords, ag_mask)

        u_ctx = ra_emb
        for layer in self.layers:
            u_ctx = layer(u_ctx, ra_emb, edge_index, edge_dist, ag_mask)

        return u_ctx


class MessagePassingAdapter(nn.Module):

    def __init__(self, hidden_dim, k=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k

        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.coord_rbf = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, u_ctx, ra_emb, edge_index, edge_dist, mask):
        """
        Args:
            u_ctx: [B, L, hidden_dim]
            ra_emb: [B, L, hidden_dim]
            edge_index: [B, L, k]
            edge_dist: [B, L, k]
            mask: [B, L]
        Returns:
            u_ctx_new: [B, L, hidden_dim]
        """
        B, L, H = u_ctx.shape
        k = edge_index.size(-1)

        target_feat = u_ctx.unsqueeze(2).expand(-1, -1, k, -1)

        u_ctx_flat = u_ctx.reshape(B * L, H)
        batch_offsets = torch.arange(B, device=u_ctx.device).view(B, 1, 1) * L
        flat_edge = (edge_index + batch_offsets).reshape(-1)
        src_feat = u_ctx_flat[flat_edge].view(B, L, k, H)

        dist_emb = self.coord_rbf(edge_dist.unsqueeze(-1))

        edge_feat = torch.cat([target_feat, src_feat, dist_emb], dim=-1)
        edge_msg = self.edge_update(edge_feat)
        edge_msg = edge_msg * mask.unsqueeze(-1).unsqueeze(-1).float()

        aggr = edge_msg.mean(dim=2)
        aggr = aggr * mask.unsqueeze(-1).float()

        node_update = self.node_update(torch.cat([u_ctx, aggr], dim=-1))
        node_update = node_update * mask.unsqueeze(-1).float()

        u_ctx_new = self.layer_norm(u_ctx + node_update)

        return u_ctx_new


class ConcatReliabilityAdapter(nn.Module):

    def __init__(self, node_feat_dim=128, ra_feat_dim=8, hidden_dim=128):
        super().__init__()
        self.ra_proj = nn.Linear(ra_feat_dim, hidden_dim)
        self.out_proj = nn.Linear(node_feat_dim + hidden_dim, node_feat_dim)

    def forward(self, h_ag, ra_features):
        """
        Simple concat adapter for ablation (ra_context_encoder_mode='concat').

        Args:
            h_ag: [B, L, node_feat_dim]
            ra_features: [B, L, 8]
        Returns:
            h_ag_ctx: [B, L, node_feat_dim]
        """
        ra_emb = self.ra_proj(ra_features)
        h_cat = torch.cat([h_ag, ra_emb], dim=-1)
        h_ag_ctx = self.out_proj(h_cat)
        return h_ag_ctx + h_ag


class MaskedReliabilityDenoisingHead(nn.Module):
    """
    Self-supervised denoising head for reliability field.
    Masks random antigen residues and predicts reliability channels from context.
    """

    def __init__(self, hidden_dim=128, ra_feat_dim=8, mask_ratio=0.15, dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ra_feat_dim = ra_feat_dim

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ra_feat_dim),
        )

    def create_mask(self, ra_features, ag_mask):
        """
        Creates a random mask for reliability field denoising.

        Args:
            ra_features: [B, L, 8]
            ag_mask: [B, L]
        Returns:
            mask: [B, L] bool, True = masked (to be predicted)
            ra_masked: [B, L, 8] reliability features with masked positions zeroed
        """
        B, L, _ = ra_features.shape
        device = ra_features.device

        rand = torch.rand(B, L, device=device)
        denoise_mask = (rand < self.mask_ratio) & ag_mask

        ra_masked = ra_features.clone()
        ra_masked[denoise_mask] = 0.0

        return denoise_mask, ra_masked

    def forward(self, u_ctx, ra_features_original, denoise_mask, ag_mask):
        """
        Compute denoising loss.

        Args:
            u_ctx: [B, L, hidden_dim] context encoder output
            ra_features_original: [B, L, 8] original reliability features
            denoise_mask: [B, L] positions that were masked
            ag_mask: [B, L] valid antigen positions
        Returns:
            denoising_loss: scalar MSE loss on masked positions
            ra_pred: [B, L, 8] predicted reliability features
        """
        ra_pred = self.predictor(u_ctx)

        valid = denoise_mask & ag_mask
        num_valid = valid.sum().float().clamp(min=1)

        diff = (ra_pred - ra_features_original) ** 2
        loss = (diff.mean(dim=-1) * valid.float()).sum() / num_valid

        return loss, ra_pred