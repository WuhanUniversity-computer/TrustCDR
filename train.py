import os
import csv
import shutil
import argparse
import torch
torch.cuda.set_device(0)
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *


def get_param_group_optimizer(config, model, logger=None):
    """Create optimizer with parameter-group learning rates."""
    train_cfg = config.train
    if not train_cfg.get('param_group_lr_enable', False):
        return get_optimizer(config.train.optimizer, model)

    lr_film = train_cfg.get('lr_film', 5e-4)
    lr_pair_correction = train_cfg.get('lr_pair_correction', 2e-5)
    lr_alpha = train_cfg.get('lr_alpha', 1e-6)
    lr_backbone = train_cfg.get('lr_backbone', 1e-5)
    lr_ag_seq = train_cfg.get('lr_ag_seq', 5e-4)

    groups = {
        'film': {'params': [], 'lr': lr_film, 'name': 'film'},
        'ag_seq': {'params': [], 'lr': lr_ag_seq, 'name': 'ag_seq'},
        'pair_correction': {'params': [], 'lr': lr_pair_correction, 'name': 'pair_correction'},
        'alpha': {'params': [], 'lr': lr_alpha, 'name': 'alpha'},
        'backbone': {'params': [], 'lr': lr_backbone, 'name': 'backbone'},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if 'film_mlp' in name_lower or 'film_layernorm' in name_lower or 'reliability_film' in name_lower:
            groups['film']['params'].append(param)
        elif name_lower.endswith('alpha') or name_lower == 'ra_pair_modulation.alpha':
            groups['alpha']['params'].append(param)
        elif 'ag_seq_context' in name_lower or 'seq_delta_mlp' in name_lower:
            # Antigen sequence context branch (must be checked before delta_mlp)
            groups['ag_seq']['params'].append(param)
        elif 'pair_modulation' in name_lower or 'delta_mlp' in name_lower or 'ra_pair_modulation' in name_lower:
            groups['pair_correction']['params'].append(param)
        else:
            groups['backbone']['params'].append(param)

    # Remove empty groups
    param_groups = [g for g in groups.values() if len(g['params']) > 0]

    if logger:
        for g in param_groups:
            logger.info(f'Param group [{g["name"]}]: lr={g["lr"]}, params={sum(p.numel() for p in g["params"])}')

    opt_cfg = config.train.optimizer
    return torch.optim.Adam(
        param_groups,
        lr=lr_backbone,
        weight_decay=opt_cfg.weight_decay,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
    )


def write_param_groups_report(config, model, report_path, logger=None):
    """Write parameter groups report to markdown."""
    train_cfg = config.train
    lr_film = train_cfg.get('lr_film', 5e-4)
    lr_pair_correction = train_cfg.get('lr_pair_correction', 2e-5)
    lr_alpha = train_cfg.get('lr_alpha', 1e-6)
    lr_backbone = train_cfg.get('lr_backbone', 1e-5)
    lr_ag_seq = train_cfg.get('lr_ag_seq', 5e-4)

    groups = {
        'film': {'params': [], 'lr': lr_film, 'examples': []},
        'ag_seq': {'params': [], 'lr': lr_ag_seq, 'examples': []},
        'pair_correction': {'params': [], 'lr': lr_pair_correction, 'examples': []},
        'alpha': {'params': [], 'lr': lr_alpha, 'examples': []},
        'backbone': {'params': [], 'lr': lr_backbone, 'examples': []},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if 'film_mlp' in name_lower or 'film_layernorm' in name_lower or 'reliability_film' in name_lower:
            groups['film']['params'].append(name)
            if len(groups['film']['examples']) < 3:
                groups['film']['examples'].append(name)
        elif name_lower.endswith('alpha') or name_lower == 'ra_pair_modulation.alpha':
            groups['alpha']['params'].append(name)
            if len(groups['alpha']['examples']) < 3:
                groups['alpha']['examples'].append(name)
        elif 'ag_seq_context' in name_lower or 'seq_delta_mlp' in name_lower:
            groups['ag_seq']['params'].append(name)
            if len(groups['ag_seq']['examples']) < 3:
                groups['ag_seq']['examples'].append(name)
        elif 'pair_modulation' in name_lower or 'delta_mlp' in name_lower or 'ra_pair_modulation' in name_lower:
            groups['pair_correction']['params'].append(name)
            if len(groups['pair_correction']['examples']) < 3:
                groups['pair_correction']['examples'].append(name)
        else:
            groups['backbone']['params'].append(name)
            if len(groups['backbone']['examples']) < 3:
                groups['backbone']['examples'].append(name)

    with open(report_path, 'w') as f:
        f.write('# Parameter Groups Report\n\n')
        f.write('| Group | Learning Rate | Num Params | Param Count | Examples |\n')
        f.write('|-------|--------------|------------|-------------|----------|\n')
        for name, g in groups.items():
            n_params = sum(p.numel() for p in model.parameters() if any(p is mp for mp in [model.get_parameter(n) for n in g['params'] if n in dict(model.named_parameters())]))
            f.write(f'| {name} | {g["lr"]} | {len(g["params"])} | {n_params} | {", ".join(g["examples"][:3])} |\n')


def write_checkpoint_loading_report(ckpt_path, model, missing_keys, unexpected_keys, report_path):
    """Write checkpoint loading report to markdown."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_dict = model.state_dict()
    loaded_count = sum(1 for k in model_dict if k in ckpt['model'])

    # Get alpha value
    alpha_val = None
    for k in model_dict:
        if k.lower().endswith('alpha') or k == 'ra_pair_modulation.alpha':
            alpha_val = model_dict[k].item()
            break

    # Identify new FiLM parameters
    film_params = [k for k in missing_keys if 'film' in k.lower()]

    with open(report_path, 'w') as f:
        f.write('# Checkpoint Loading Report\n\n')
        f.write(f'**Loaded checkpoint path:** `{ckpt_path}`\n\n')
        f.write(f'**Loaded parameter count:** {loaded_count}\n\n')
        f.write(f'**Missing keys (random init):** {len(missing_keys)}\n\n')
        for k in missing_keys:
            f.write(f'  - `{k}`\n')
        f.write(f'\n**Unexpected keys (ignored):** {len(unexpected_keys)}\n\n')
        for k in unexpected_keys[:20]:
            f.write(f'  - `{k}`\n')
        if len(unexpected_keys) > 20:
            f.write(f'  ... and {len(unexpected_keys) - 20} more\n')
        f.write(f'\n**Alpha value at load:** {alpha_val}\n\n')
        f.write(f'**New FiLM parameters:**\n\n')
        for k in film_params:
            f.write(f'  - `{k}` (shape={model_dict[k].shape})\n')


CSV_FIELDS = [
    'iteration', 'mode', 'loss_total', 'loss_fm', 'loss_clean', 'loss_perturbed',
    'loss_consistency', 'alpha', 'alpha_abs', 'delta_rms', 'pair_update_rms',
    'update_to_z_ratio', 'film_gamma_mean', 'film_gamma_std', 'film_gamma_abs_mean',
    'film_gamma_min', 'film_gamma_max', 'film_beta_mean', 'film_beta_std',
    'film_beta_abs_mean', 'film_beta_min', 'film_beta_max', 'film_mod_rms',
    'film_mod_to_z_ratio', 'film_mod_to_delta_ratio', 'gate_mean', 'gate_std',
    'scale_mean', 'scale_std', 'perturb_applied_ratio', 'num_clean_samples',
    'num_perturbed_samples', 'film_last_layer_weight_norm', 'film_last_layer_grad_norm',
    'film_total_grad_norm', 'delta_mlp_grad_norm', 'backbone_grad_norm',
    'velocity_clean_rms', 'velocity_perturbed_rms', 'velocity_gap_rms',
    'velocity_gap_to_clean_ratio',
    # Contact gate stats
    'cg_w_mean', 'cg_w_std', 'cg_w_min', 'cg_w_max',
    'cg_r_mean', 'cg_r_std', 'cg_r_min', 'cg_r_max',
    'cg_r_hat_mean', 'cg_r_hat_std',
    'cg_w_contact_norm_mean_error', 'cg_w_contact_norm_max_error',
    'cg_mean_abs_w_minus_1',
    'cg_H1_w_mean', 'cg_H1_w_std', 'cg_H2_w_mean', 'cg_H2_w_std',
    'cg_H3_w_mean', 'cg_H3_w_std', 'cg_L1_w_mean', 'cg_L1_w_std',
    'cg_L2_w_mean', 'cg_L2_w_std', 'cg_L3_w_mean', 'cg_L3_w_std',
    # Antigen sequence context branch stats
    'ag_seq_branch_active', 'ag_seq_base_delta_rms', 'ag_seq_seq_delta_rms',
    'ag_seq_final_delta_rms', 'ag_seq_delta_to_base_ratio',
    'ag_seq_context_norm', 'ag_seq_grad_norm',
]


def init_csv(csv_path, fields):
    """Initialize CSV with header."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def append_csv(csv_path, row, fields):
    """Append a row to CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        # Only write fields that exist in the row, fill missing with 0
        clean_row = {k: row.get(k, 0) for k in fields}
        writer.writerow(clean_row)


def get_tensor_val(d, key, default=0.0):
    """Extract scalar value from a dict that may contain tensors."""
    v = d.get(key, default)
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Perturbation training config
    perturb_cfg = config.train.get('perturb_train_enable', False)
    perturb_mode = config.train.get('perturb_train_mode', 'single_forward_replace')
    perturb_clean_prob = config.train.get('perturb_train_clean_prob', 0.70)
    perturb_level = config.train.get('perturb_train_level', 0.8)
    perturb_region = config.train.get('perturb_train_region', 'epitope')

    # Consistency training config
    consistency_enable = config.train.get('consistency_enable', False)
    consistency_step_prob = config.train.get('consistency_step_prob', 0.30)
    lambda_perturbed_fm = config.train.get('lambda_perturbed_fm', 0.50)
    lambda_consistency = config.train.get('lambda_consistency', 0.05)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Output directory for reports
    out_dir = config.train.get('output_dir', None)
    if out_dir is None:
        out_dir = os.path.dirname(log_dir) if not args.debug else '/tmp/softc_film_v2_debug'
    os.makedirs(out_dir, exist_ok=True)

    # CSV stats path
    csv_tag = 'consistency' if consistency_enable else 'perturb'
    csv_path = os.path.join(out_dir, f'film_v2_{csv_tag}_train_stats.csv')
    if not args.debug:
        init_csv(csv_path, CSV_FIELDS)
        logger.info(f'Train stats CSV: {csv_path}')

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler (with parameter group support)
    if config.train.get('param_group_lr_enable', False):
        optimizer = get_param_group_optimizer(config, model, logger=logger)
        # Write param groups report
        if not args.debug:
            pg_report = os.path.join(out_dir, f'param_groups_{csv_tag}.md')
            write_param_groups_report(config, model, pg_report, logger=logger)
            logger.info(f'Param groups report: {pg_report}')
    else:
        optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume / Finetune
    missing_keys = []
    unexpected_keys = []
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration'] + 1
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
        missing_keys = [k for k in model_dict if k not in pretrained_dict]
        unexpected_keys = [k for k in ckpt['model'] if k not in model_dict]
        if missing_keys:
            logger.info('Missing keys (random init): %s' % str(missing_keys[:20]))
        if unexpected_keys:
            logger.info('Unexpected keys (ignored): %s' % str(unexpected_keys[:20]))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # Write checkpoint loading report
        if not args.debug:
            ckpt_report = os.path.join(out_dir, f'checkpoint_loading_{csv_tag}.md')
            write_checkpoint_loading_report(ckpt_path, model, missing_keys, unexpected_keys, ckpt_report)
            logger.info(f'Checkpoint loading report: {ckpt_report}')

        logger.info('Resuming optimizer states...')
        if args.finetune is not None:
            # For finetune with new params, reinitialize optimizer
            logger.info('Finetune mode: optimizer reinitialized for new parameter groups')
        else:
            optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        # Fix: ensure scheduler min_lrs matches optimizer param_groups
        # (can mismatch when finetuning with different param groups)
        if hasattr(scheduler, 'min_lrs') and len(scheduler.min_lrs) != len(optimizer.param_groups):
            logger.warning(f'Scheduler min_lrs length ({len(scheduler.min_lrs)}) != optimizer param_groups ({len(optimizer.param_groups)}), fixing...')
            if len(scheduler.min_lrs) < len(optimizer.param_groups):
                scheduler.min_lrs.extend([scheduler.min_lrs[-1]] * (len(optimizer.param_groups) - len(scheduler.min_lrs)))
            else:
                scheduler.min_lrs = scheduler.min_lrs[:len(optimizer.param_groups)]

    # Checkpoint save freq
    ckpt_save_freq = config.checkpoint.get('save_freq', 10000)

    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()

        optimizer.zero_grad()
        batch = recursive_to(next(train_iterator), args.device)

        # Determine if this step uses perturbation
        use_perturb = False
        use_consistency = False
        if perturb_cfg and perturb_mode == 'single_forward_replace':
            use_perturb = torch.rand(1).item() > perturb_clean_prob
        elif consistency_enable:
            use_consistency = torch.rand(1).item() < consistency_step_prob
            if not use_consistency:
                # Non-consistency step: use single forward replace
                use_perturb = torch.rand(1).item() > perturb_clean_prob

        csv_row = {'iteration': it}

        if use_consistency:
            # === Consistency training: dual forward ===
            csv_row['mode'] = 'consistency'

            # Clean forward (with velocities)
            loss_dict_clean, preds_clean = model(batch, return_velocities=True)
            loss_clean = sum_weighted_losses(loss_dict_clean, config.train.loss_weights)

            # Perturbed forward
            from diffab.trust_cdr.train_perturbation import perturb_batch_antigen, compute_consistency_loss
            perturbed_batch = perturb_batch_antigen(batch, target_rmsd=perturb_level)
            perturbed_batch = recursive_to(perturbed_batch, args.device)

            loss_dict_pert, preds_pert = model(perturbed_batch, return_velocities=True)
            loss_perturbed = sum_weighted_losses(loss_dict_pert, config.train.loss_weights)

            # Consistency loss
            mask_generate = batch['generate_flag']
            loss_cons = compute_consistency_loss(preds_pert, preds_clean, mask_generate)

            # Total loss
            loss = loss_clean + lambda_perturbed_fm * loss_perturbed + lambda_consistency * loss_cons

            loss_dict = loss_dict_clean
            loss_dict['overall'] = loss
            loss_dict['loss_clean'] = loss_clean
            loss_dict['loss_perturbed'] = loss_perturbed
            loss_dict['loss_consistency'] = loss_cons

            # CSV extras
            csv_row['loss_clean'] = loss_clean.item()
            csv_row['loss_perturbed'] = loss_perturbed.item()
            csv_row['loss_consistency'] = loss_cons.item()
            csv_row['num_clean_samples'] = batch['pos_heavyatom'].shape[0]
            csv_row['num_perturbed_samples'] = batch['pos_heavyatom'].shape[0]
            csv_row['perturb_applied_ratio'] = 1.0

            # Velocity stats
            with torch.no_grad():
                v_clean_rms = torch.sqrt((preds_clean['vel_pos'] ** 2).mean()).item()
                v_pert_rms = torch.sqrt((preds_pert['vel_pos'] ** 2).mean()).item()
                v_gap = torch.sqrt(((preds_pert['vel_pos'] - preds_clean['vel_pos'].detach()) ** 2).mean()).item()
                csv_row['velocity_clean_rms'] = v_clean_rms
                csv_row['velocity_perturbed_rms'] = v_pert_rms
                csv_row['velocity_gap_rms'] = v_gap
                csv_row['velocity_gap_to_clean_ratio'] = v_gap / (v_clean_rms + 1e-8)

            time_forward_end = current_milli_time()

        elif use_perturb:
            # === Single forward with perturbed antigen ===
            csv_row['mode'] = 'perturbed'
            from diffab.trust_cdr.train_perturbation import perturb_batch_antigen
            batch = perturb_batch_antigen(batch, target_rmsd=perturb_level)
            batch = recursive_to(batch, args.device)

            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()

            csv_row['num_perturbed_samples'] = batch['pos_heavyatom'].shape[0]
            csv_row['perturb_applied_ratio'] = 1.0

        else:
            # === Normal clean forward ===
            csv_row['mode'] = 'clean'
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()
            csv_row['num_clean_samples'] = batch['pos_heavyatom'].shape[0]

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        log_losses(loss_dict, it, 'train', logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })

        # CSV stats (every 1000 steps)
        if it % 1000 == 0 or it == it_first:
            csv_row['loss_total'] = loss.item()
            csv_row['loss_fm'] = get_tensor_val(loss_dict, 'seq') + get_tensor_val(loss_dict, 'pos') + get_tensor_val(loss_dict, 'qua')
            csv_row['alpha'] = get_tensor_val(loss_dict, 'pair_alpha')
            csv_row['alpha_abs'] = abs(csv_row['alpha'])
            csv_row['delta_rms'] = get_tensor_val(loss_dict, 'pair_delta_rms')
            csv_row['pair_update_rms'] = get_tensor_val(loss_dict, 'pair_update_rms')
            csv_row['update_to_z_ratio'] = get_tensor_val(loss_dict, 'pair_update_to_z_ratio')
            csv_row['film_gamma_mean'] = get_tensor_val(loss_dict, 'film_gamma_mean')
            csv_row['film_gamma_std'] = get_tensor_val(loss_dict, 'film_gamma_std')
            csv_row['film_gamma_abs_mean'] = get_tensor_val(loss_dict, 'film_gamma_abs_mean')
            csv_row['film_gamma_min'] = get_tensor_val(loss_dict, 'film_gamma_min')
            csv_row['film_gamma_max'] = get_tensor_val(loss_dict, 'film_gamma_max')
            csv_row['film_beta_mean'] = get_tensor_val(loss_dict, 'film_beta_mean')
            csv_row['film_beta_std'] = get_tensor_val(loss_dict, 'film_beta_std')
            csv_row['film_beta_abs_mean'] = get_tensor_val(loss_dict, 'film_beta_abs_mean')
            csv_row['film_beta_min'] = get_tensor_val(loss_dict, 'film_beta_min')
            csv_row['film_beta_max'] = get_tensor_val(loss_dict, 'film_beta_max')
            csv_row['film_mod_rms'] = get_tensor_val(loss_dict, 'film_mod_rms')
            csv_row['film_mod_to_z_ratio'] = get_tensor_val(loss_dict, 'film_mod_to_z_ratio')
            csv_row['film_mod_to_delta_ratio'] = get_tensor_val(loss_dict, 'film_mod_to_delta_ratio')
            csv_row['gate_mean'] = get_tensor_val(loss_dict, 'pair_gate_mean_valid')
            csv_row['gate_std'] = get_tensor_val(loss_dict, 'pair_gate_std_valid')
            csv_row['scale_mean'] = get_tensor_val(loss_dict, 'softc_reliability_scale_mean_valid')
            csv_row['scale_std'] = get_tensor_val(loss_dict, 'softc_reliability_scale_std_valid')

            # Contact gate stats
            csv_row['cg_w_mean'] = get_tensor_val(loss_dict, 'contact_gate_w_mean')
            csv_row['cg_w_std'] = get_tensor_val(loss_dict, 'contact_gate_w_std')
            csv_row['cg_w_min'] = get_tensor_val(loss_dict, 'contact_gate_w_min')
            csv_row['cg_w_max'] = get_tensor_val(loss_dict, 'contact_gate_w_max')
            csv_row['cg_r_mean'] = get_tensor_val(loss_dict, 'contact_gate_r_mean')
            csv_row['cg_r_std'] = get_tensor_val(loss_dict, 'contact_gate_r_std')
            csv_row['cg_r_min'] = get_tensor_val(loss_dict, 'contact_gate_r_min')
            csv_row['cg_r_max'] = get_tensor_val(loss_dict, 'contact_gate_r_max')
            csv_row['cg_r_hat_mean'] = get_tensor_val(loss_dict, 'contact_gate_r_hat_mean')
            csv_row['cg_r_hat_std'] = get_tensor_val(loss_dict, 'contact_gate_r_hat_std')
            csv_row['cg_w_contact_norm_mean_error'] = get_tensor_val(loss_dict, 'contact_gate_w_contact_norm_mean_error')
            csv_row['cg_w_contact_norm_max_error'] = get_tensor_val(loss_dict, 'contact_gate_w_contact_norm_max_error')
            csv_row['cg_mean_abs_w_minus_1'] = get_tensor_val(loss_dict, 'contact_gate_mean_abs_w_minus_1')
            for cdr in ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']:
                csv_row[f'cg_{cdr}_w_mean'] = get_tensor_val(loss_dict, f'contact_gate_{cdr}_w_mean')
                csv_row[f'cg_{cdr}_w_std'] = get_tensor_val(loss_dict, f'contact_gate_{cdr}_w_std')

            # Antigen sequence context branch stats
            csv_row['ag_seq_branch_active'] = get_tensor_val(loss_dict, 'ag_seq_branch_active')
            csv_row['ag_seq_base_delta_rms'] = get_tensor_val(loss_dict, 'ag_seq_base_delta_rms')
            csv_row['ag_seq_seq_delta_rms'] = get_tensor_val(loss_dict, 'ag_seq_seq_delta_rms')
            csv_row['ag_seq_final_delta_rms'] = get_tensor_val(loss_dict, 'ag_seq_final_delta_rms')
            csv_row['ag_seq_delta_to_base_ratio'] = get_tensor_val(loss_dict, 'ag_seq_delta_to_base_ratio')
            csv_row['ag_seq_context_norm'] = get_tensor_val(loss_dict, 'ag_seq_context_norm')

            # Gradient norms by group
            film_grad_norm = 0.0
            delta_grad_norm = 0.0
            backbone_grad_norm = 0.0
            film_last_layer_grad = 0.0
            ag_seq_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                gn = param.grad.norm().item() ** 2
                name_lower = name.lower()
                if 'film_mlp' in name_lower or 'film_layernorm' in name_lower:
                    film_grad_norm += gn
                    if 'film_mlp.3' in name or 'film_mlp.1' in name:  # last linear
                        film_last_layer_grad += gn
                elif 'ag_seq_context' in name_lower or 'seq_delta_mlp' in name_lower:
                    ag_seq_grad_norm += gn
                elif 'delta_mlp' in name_lower:
                    delta_grad_norm += gn
                else:
                    backbone_grad_norm += gn
            csv_row['film_last_layer_grad_norm'] = film_last_layer_grad ** 0.5
            csv_row['film_total_grad_norm'] = film_grad_norm ** 0.5
            csv_row['delta_mlp_grad_norm'] = delta_grad_norm ** 0.5
            csv_row['backbone_grad_norm'] = backbone_grad_norm ** 0.5
            csv_row['ag_seq_grad_norm'] = ag_seq_grad_norm ** 0.5

            # Film last layer weight norm
            for name, param in model.named_parameters():
                if 'film_mlp' in name.lower() and 'weight' in name.lower():
                    csv_row['film_last_layer_weight_norm'] = param.norm().item()
                    break

            if not args.debug:
                append_csv(csv_path, csv_row, CSV_FIELDS)

        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss
                loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
            # Intermediate checkpoint save
            elif not args.debug and it % ckpt_save_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
                logger.info(f'Saved intermediate checkpoint: {ckpt_path}')

    except KeyboardInterrupt:
        logger.info('Terminating...')
