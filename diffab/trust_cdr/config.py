from easydict import EasyDict


_RA_DEFAULT_CONFIG = {
    'use_trust_cdr': False,

    # Module 1: Reliability Field
    'ra_reliability_channels': ['coordinate', 'topology', 'orientation', 'surface'],
    'ra_num_ensemble': 8,
    'ra_perturb_sigma': 0.3,
    'ra_smooth_window': 3,
    'ra_max_perturb': 1.5,
    'ra_topology_k': 16,
    'ra_surface_cutoff': 10.0,
    'ra_reliability_alpha': 1.0,
    'use_antigen_reliability_cache': True,
    'antigen_reliability_cache_dir': None,
    'ra_compute_online_if_missing': False,
    'ra_shuffle_reliability': False,

    # Module 2: Context Encoder
    'ra_context_encoder_mode': 'none',
    'ra_context_layers': 2,
    'ra_context_k': 16,
    'ra_context_hidden_dim': 128,
    'ra_context_dropout': 0.1,

    # Module 3: Relation Gate
    'ra_use_relation_gate': True,
    'ra_gate_mode': 'scalar',
    'ra_gate_insert_attention': True,
    'ra_gate_insert_pair_update': True,
    'ra_gate_hidden_dim': 128,
    'ra_gate_dropout': 0.1,
    'ra_gate_init': 'near_identity',
    'ra_gate_use_distance': True,
    'ra_gate_use_pair_feature': True,
    'ra_gate_log_values': True,

    # Denoising head
    'ra_use_denoising_head': False,
    'ra_denoising_loss_weight': 0.01,
    'ra_denoising_mask_ratio': 0.15,
    'ra_denoising_target': 'reliability',

    # Regularization
    'ra_gate_balance_loss_weight': 0.0,
    'ra_gate_entropy_loss_weight': 0.0,
}


RAConfig = EasyDict(_RA_DEFAULT_CONFIG)


def get_ra_config(cfg):
    """
    Extract RA sub-config from the full config.

    Args:
        cfg: Full config dict/EasyDict (e.g. from yaml)
    Returns:
        EasyDict with RA-specific config, merged with defaults
    """
    ra_cfg = EasyDict(_RA_DEFAULT_CONFIG.copy())
    for key in _RA_DEFAULT_CONFIG:
        if key in cfg:
            ra_cfg[key] = cfg[key]
    return ra_cfg


def merge_ra_config_to_model_cfg(model_cfg, ra_cfg):
    """
    Merge RA config into the model config for passing to model constructor.

    Args:
        model_cfg: EasyDict model config
        ra_cfg: EasyDict RA config
    Returns:
        updated model_cfg
    """
    for key in _RA_DEFAULT_CONFIG:
        model_cfg[key] = ra_cfg[key]
    return model_cfg