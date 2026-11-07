"""Contact-Aware Reliability Gate for Soft-C pair correction.

Implements three variants:
- contact_all: all CDRs use reliability gate
- contact_h3l3 / contact_h3: only H3/L3 or H3 use gate
- contact_downonly: only suppress low-reliability regions

Gate computation:
1. Standardize reliability: r_hat = tanh((rho - mu_A) / (T * sigma_A + eps))
2. Raw gate: w_raw = 1 + lambda * r_hat  (or 1 - lambda * max(0, -r_hat) for downonly)
3. Clip: w_clip = clip(w_raw, 1-c, 1+c)  (or clip(w_raw, 1-c, 1) for downonly)
4. Contact-aware mean-preserving normalization:
   w_bar_i = sum_j(m_ij * w_clip_j) / (sum_j(m_ij) + eps)
   w_ij = w_clip_j / (w_bar_i + eps)
5. Final: Z'_ij = Z_ij + alpha * m_ij * w_ij * Delta_ij
"""

import torch
import torch.nn as nn


# CDR type constants (matching diffab.utils.protein.constants.CDR)
CDR_H1 = 1
CDR_H2 = 2
CDR_H3 = 3
CDR_L1 = 4
CDR_L2 = 5
CDR_L3 = 6

CDR_NAME_MAP = {
    CDR_H1: 'H1', CDR_H2: 'H2', CDR_H3: 'H3',
    CDR_L1: 'L1', CDR_L2: 'L2', CDR_L3: 'L3',
}


def compute_contact_reliability_gate(
    reliability,
    pair_mask,
    cdr_flag=None,
    variant="contact_all",
    lambda_value=0.10,
    clip_value=0.20,
    temperature=1.0,
    eps=1e-6,
    cdr_lambdas=None,
    normalization="contact_aware_mean_preserving",
):
    """Compute contact-aware reliability gate.

    Args:
        reliability: scalar reliability rho, shape [L_antigen] for single sample
        pair_mask: CDR-antigen contact mask, shape [L_cdr, L_antigen], bool
        cdr_flag: CDR type for each CDR residue, shape [L_cdr], int
            (1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3)
        variant: "contact_all", "contact_h3l3", "contact_h3", "contact_downonly"
        lambda_value: gate strength
        clip_value: clip range
        temperature: normalization temperature
        eps: numerical stability
        cdr_lambdas: dict mapping CDR name to lambda, e.g. {'H3': 0.10, 'L3': 0.10}
            If provided, overrides lambda_value for specific CDRs.
        normalization: one of
            "contact_aware_mean_preserving" (default, D): per-CDR-row mean-preserving.
            "global_mean_preserving" (deprecated C): single shared factor; under the
                production all-antigen mask this is mathematically equivalent to D.
            "none" / "no_mean_preserving" (C′ ablation): use the raw clipped down-only
                weight directly (w_ij = w_clip) without any normalization, so w_mean < 1
                and the overall residual update magnitude is NOT preserved.

    Returns:
        w_ij: gate weights, shape [L_cdr, L_antigen], broadcastable to pair update
        stats: dict of gate statistics
    """
    L_cdr, L_ag = pair_mask.shape
    device = reliability.device

    # Step 1: Standardize reliability over antigen residues
    rho_valid = reliability  # [L_ag]
    mu_A = rho_valid.mean()
    sigma_A = rho_valid.std()
    r_hat = torch.tanh((rho_valid - mu_A) / (temperature * sigma_A + eps))  # [L_ag]

    # Step 2: Compute per-antigen gate weights
    if variant == "contact_downonly":
        # Only suppress low-reliability: w_raw = 1 - lambda * max(0, -r_hat)
        w_raw = 1.0 - lambda_value * torch.clamp(-r_hat, min=0.0)  # [L_ag]
        w_clip = torch.clamp(w_raw, min=1.0 - clip_value, max=1.0)  # [L_ag]
    else:
        # Standard: w_raw = 1 + lambda * r_hat
        w_clip = 1.0 + lambda_value * r_hat  # [L_ag]
        w_clip = torch.clamp(w_clip, min=1.0 - clip_value, max=1.0 + clip_value)  # [L_ag]

    # Step 3: CDR-specific lambda adjustment
    # For contact_h3l3 / contact_h3, only apply gate to specific CDRs
    # We compute per-CDR-residue lambda
    if variant in ("contact_h3l3", "contact_h3") and cdr_flag is not None:
        # Default lambda is 0 for all CDRs
        per_cdr_lambda = torch.zeros(L_cdr, device=device)
        if cdr_lambdas is not None:
            for cdr_val, cdr_name in CDR_NAME_MAP.items():
                lam = cdr_lambdas.get(cdr_name, 0.0)
                per_cdr_lambda[cdr_flag == cdr_val] = lam
        else:
            # Default: H3=0.10, L3=0.10 for h3l3; H3=0.10 for h3
            if variant == "contact_h3l3":
                per_cdr_lambda[cdr_flag == CDR_H3] = lambda_value
                per_cdr_lambda[cdr_flag == CDR_L3] = lambda_value
            elif variant == "contact_h3":
                per_cdr_lambda[cdr_flag == CDR_H3] = lambda_value

        # Recompute w_clip per CDR-residue with per-CDR lambda
        # w_clip is [L_ag], we need [L_cdr, L_ag]
        # For CDRs with lambda=0, w=1; for others, use the gate
        lam_expanded = per_cdr_lambda.unsqueeze(1)  # [L_cdr, 1]
        r_hat_expanded = r_hat.unsqueeze(0)  # [1, L_ag]

        if variant == "contact_downonly":
            w_raw_2d = 1.0 - lam_expanded * torch.clamp(-r_hat_expanded, min=0.0)
            w_clip_2d = torch.clamp(w_raw_2d, min=1.0 - clip_value, max=1.0)
        else:
            w_raw_2d = 1.0 + lam_expanded * r_hat_expanded
            w_clip_2d = torch.clamp(w_raw_2d, min=1.0 - clip_value, max=1.0 + clip_value)

        w_clip = w_clip_2d  # [L_cdr, L_ag]
    else:
        # contact_all or no cdr_flag: same w_clip for all CDRs
        w_clip = w_clip.unsqueeze(0).expand(L_cdr, -1)  # [L_cdr, L_ag]

    # Step 4: Mean-preserving normalization
    # Three modes:
    #   contact_aware_mean_preserving (D, default):
    #       w_bar_i = sum_j(m_ij * w_clip_ij) / (sum_j(m_ij) + eps)   # per CDR row
    #       w_ij = w_clip_ij / (w_bar_i + eps)
    #   global_mean_preserving (deprecated C, ablation):
    #       w_bar = sum_ij(m_ij * w_clip_ij) / (sum_ij(m_ij) + eps)   # single scalar
    #       w_ij = w_clip_ij / (w_bar + eps)
    #       All CDR residues share the same normalization factor.
    #   none / no_mean_preserving (C′, ablation):
    #       w_ij = w_clip_ij   (raw clipped down-only weight, NOT normalized)
    #       w_mean < 1 because low-reliability residues are suppressed without
    #       rescaling the overall residual update magnitude back to 1.
    m = pair_mask.float()  # [L_cdr, L_ag]
    w_masked = w_clip * m  # [L_cdr, L_ag]
    if normalization in ("none", "no_mean_preserving"):
        # C′ ablation: use raw clipped down-only weight directly, no normalization.
        w_ij = w_clip  # [L_cdr, L_ag]
        # w_bar reported for stats only (NOT applied): actual per-row mean of w_clip
        w_bar = w_masked.sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + eps)  # [L_cdr, 1]
    elif normalization == "global_mean_preserving":
        w_bar = w_masked.sum() / (m.sum() + eps)  # scalar
        w_ij = w_clip / (w_bar + eps)  # [L_cdr, L_ag]
    else:
        # contact_aware_mean_preserving (default, backward compatible)
        w_bar = w_masked.sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + eps)  # [L_cdr, 1]
        w_ij = w_clip / (w_bar + eps)  # [L_cdr, L_ag]

    # Ensure non-gated CDRs have w_ij = 1
    if variant in ("contact_h3l3", "contact_h3") and cdr_flag is not None:
        # For CDRs with lambda=0, w_clip=1, w_bar=1, w_ij=1
        # This is already satisfied by the math, but let's enforce it
        zero_lambda_mask = (per_cdr_lambda == 0).float().unsqueeze(1)  # [L_cdr, 1]
        w_ij = w_ij * (1 - zero_lambda_mask) + 1.0 * zero_lambda_mask

    # Step 5: Apply pair mask (invalid pairs get w=0)
    w_ij = w_ij * m  # [L_cdr, L_ag]

    # Compute statistics
    valid_mask = pair_mask
    w_valid = w_ij[valid_mask]
    r_hat_valid = r_hat.unsqueeze(0).expand(L_cdr, -1)[valid_mask]

    stats = {
        'w_mean': w_valid.mean().detach() if w_valid.numel() > 0 else torch.tensor(0.0, device=device),
        'w_std': w_valid.std().detach() if w_valid.numel() > 1 else torch.tensor(0.0, device=device),
        'w_min': w_valid.min().detach() if w_valid.numel() > 0 else torch.tensor(0.0, device=device),
        'w_max': w_valid.max().detach() if w_valid.numel() > 0 else torch.tensor(0.0, device=device),
        'r_mean': mu_A.detach(),
        'r_std': sigma_A.detach(),
        'r_min': rho_valid.min().detach(),
        'r_max': rho_valid.max().detach(),
        'r_hat_mean': r_hat.mean().detach(),
        'r_hat_std': r_hat.std().detach() if r_hat.numel() > 1 else torch.tensor(0.0, device=device),
        'w_contact_norm_mean_error': (w_bar - 1.0).abs().mean().detach(),
        'w_contact_norm_max_error': (w_bar - 1.0).abs().max().detach(),
        'mean_abs_w_minus_1': (w_valid - 1.0).abs().mean().detach() if w_valid.numel() > 0 else torch.tensor(0.0, device=device),
    }

    # Per-CDR statistics
    if cdr_flag is not None:
        for cdr_val, cdr_name in CDR_NAME_MAP.items():
            cdr_mask = (cdr_flag == cdr_val)
            if cdr_mask.any():
                # Find which rows in pair_mask correspond to this CDR
                cdr_rows = cdr_mask.nonzero(as_tuple=False).squeeze(-1)
                w_cdr = w_ij[cdr_rows]  # [n_cdr, L_ag]
                m_cdr = pair_mask[cdr_rows]
                w_cdr_valid = w_cdr[m_cdr]
                if w_cdr_valid.numel() > 0:
                    stats[f'{cdr_name}_w_mean'] = w_cdr_valid.mean().detach()
                    stats[f'{cdr_name}_w_std'] = w_cdr_valid.std().detach() if w_cdr_valid.numel() > 1 else torch.tensor(0.0, device=device)
                else:
                    stats[f'{cdr_name}_w_mean'] = torch.tensor(1.0, device=device)
                    stats[f'{cdr_name}_w_std'] = torch.tensor(0.0, device=device)
            else:
                stats[f'{cdr_name}_w_mean'] = torch.tensor(1.0, device=device)
                stats[f'{cdr_name}_w_std'] = torch.tensor(0.0, device=device)

    return w_ij, stats
