import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
import numpy as np
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from ._base import register_dataset


from diffab.utils.data import *
from diffab.utils.transforms import *

ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

FLOWTESTS = {
    '7che_H_L_R', '5w9h_H_I_G', '7d6i_B_C_A', '8ds5_C_B_A', '7chb_H_L_R',
    '5tlk_B_A_X', '7faf_g_f_B', '7ucf_H_L_G', '2r4s_H_L_A', '6al5_H_L_A',
    '4lmq_E_I_F', '4zs6_C_D_B', '4zso_D_C_F', '2hjf_A_B_C', '2yc1_A_B_C',
    '5ec2_A_B_C', '1h0d_B_A_C', '6by2_A_B_C', '7n4i_H_L_C', '7nd9_H_L_B'
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val

def generate_name(pdbcode, hchain, lchain, agchains):
    return pdbcode+'_'+nan_to_empty_string(hchain)+'_'+nan_to_empty_string(lchain)+'_'+''.join(agchains)


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    try:
        model = parser.get_structure(id, pdb_path)[0]
    except ValueError as e:
        logging.warning('[{}] Failed to parse PDB file: {}'.format(
            task['id'], str(e)
        ))
        return None
    except Exception as e:
        logging.warning('[{}] Unexpected error when parsing PDB file: {}'.format(
            task['id'], str(e)
        ))
        return None

    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if entry['H_chain'] is not None:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],
                max_resseq = 113    # Chothia, end of Heavy chain Fv
            ))
        
        if entry['L_chain'] is not None:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['L_chain']],
                max_resseq = 106    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')
    
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None

    return parsed


class SAbDabDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path = './data/sabdab_summary_all.tsv', 
        chothia_dir = './data/all_structures/chothia', 
        processed_dir = './data/processed',
        split = 'train',
        split_seed = 2022,
        transform = None,
        reset = False,
        antigen_reliability_cache_dir = None,
        shuffle_reliability = False,
        antigen_reliability_required = False,
    ):
        super().__init__()
        self.summary_path = summary_path
        self.chothia_dir = chothia_dir
        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.antigen_reliability_cache_dir = antigen_reliability_cache_dir
        self.shuffle_reliability = shuffle_reliability
        self.antigen_reliability_required = antigen_reliability_required
        
        try:
            with open ('./template/dict.pkl', 'rb') as fin:
                self.pkl_dict = pickle.load(fin)
        except:
            self.pkl_dict = None
            logging.warning('no template dict file found.')

        self.sabdab_entries = None
        self._load_sabdab_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

        self.clusters = None
        self.id_to_cluster = None
        self._load_clusters(reset)

        self.ids_in_split = None
        self._load_split(split, split_seed)

        self.transform = transform
        
    def _load_sabdab_entries(self):
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode = row['pdb'],
                H = nan_to_empty_string(row['Hchain']),
                L = nan_to_empty_string(row['Lchain']),
                Ag = ''.join(split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                ))
            )

            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'H_chain': nan_to_none(row['Hchain']),
                'L_chain': nan_to_none(row['Lchain']),
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }


            # Filtering
            if (
                (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
            ):
                entries_all.append(entry)
        self.sabdab_entries = entries_all

    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
        
    def _preprocess_structures(self):
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_sabdab_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)

    def _load_split(self, split, split_seed):
        assert split in ('train', 'val', 'test')
        
        # Load custom test set from diffab_test.idx
        custom_test_path = './data/diffab_test.idx'
        if os.path.exists(custom_test_path):
            with open(custom_test_path, 'r') as f:
                custom_test_ids = [line.strip() for line in f if line.strip()]
            # Convert custom test IDs to match the format in sabdab_entries
            custom_test_names = set(custom_test_ids)
            ids_test = [
                entry['id']
                for entry in self.sabdab_entries
                if generate_name(entry['pdbcode'], entry['H_chain'], entry['L_chain'], entry['ag_chains']) in custom_test_names
            ]
        else:
            # Fallback to default test set if custom test set not found
            ids_test = [
                entry['id']
                for entry in self.sabdab_entries
                if generate_name(entry['pdbcode'], entry['H_chain'], entry['L_chain'], entry['ag_chains']) in FLOWTESTS
            ]
        
        # Get all IDs except test set
        all_ids = [entry['id'] for entry in self.sabdab_entries]
        ids_train_val = [id for id in all_ids if id not in ids_test and id != '3o2d_H_L_A']
        random.Random(split_seed).shuffle(ids_train_val)
        
        if split == 'test':
            self.ids_in_split = ids_test
        elif split == 'val':
            self.ids_in_split = ids_train_val[:20]
        else:
            self.ids_in_split = ids_train_val[20:]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        # 循环查找有效的抗体数据
        while True:
            if index >= len(self.ids_in_split):
                # 如果超出范围，重新开始查找
                index = 0
            
            id = self.ids_in_split[index]
            data = self.get_structure(id)
            
            # 检查是否包含重链或轻链
            has_heavy = data['heavy'] is not None
            has_light = data['light'] is not None
            
            if not has_heavy and not has_light:
                # 输出缺少重链和轻链的抗体ID
                logging.warning(f"Skipping antibody {id}: missing both heavy and light chains")
                # 尝试下一个索引
                index += 1
                continue
            
            # Compute antigen offset before transforms
            heavy_len = data['heavy']['aa'].size(0) if data.get('heavy') is not None else 0
            light_len = data['light']['aa'].size(0) if data.get('light') is not None else 0
            ag_start_offset = heavy_len + light_len

            # Check if this sample has antigen data
            has_antigen = data.get('antigen') is not None

            # Load antigen reliability from cache before transforms
            ra_features_cache = None
            if self.antigen_reliability_cache_dir is not None:
                cache_path = os.path.join(self.antigen_reliability_cache_dir, 
                                          f'{id}_antigen_reliability.npz')
                if os.path.exists(cache_path):
                    cache = np.load(cache_path)
                    ra_features_cache = torch.from_numpy(cache['ra_features'].astype(np.float32))
                elif self.antigen_reliability_required and has_antigen:
                    raise FileNotFoundError(
                        f'antigen_reliability cache file not found for {id}. '
                        f'Expected at: {cache_path}. '
                        f'Run: python scripts/precompute_antigen_reliability.py to generate missing caches.'
                    )
            
            # 应用变换
            if self.transform is not None:
                try:
                    data = self.transform(data)
                except Exception as e:
                    logging.error(f"Error transforming data {id}: {str(e)}")
                    index += 1
                    continue

            # Align antigen reliability to cropped antigen residues
            if ra_features_cache is not None:
                patch_idx = data.get('patch_idx')
                if patch_idx is not None and data.get('fragment_type') is not None:
                    ag_mask = (data['fragment_type'] == constants.Fragment.Antigen)
                    if ag_mask.any():
                        ag_pos = ag_mask.nonzero(as_tuple=False).squeeze(-1)
                        orig_merged_idx = patch_idx[ag_pos]
                        ag_0based_idx = orig_merged_idx - ag_start_offset
                        ag_0based_idx = ag_0based_idx.clamp(0, ra_features_cache.size(0) - 1)
                        ra_features = ra_features_cache[ag_0based_idx]
                        if self.shuffle_reliability:
                            shuffle_idx = torch.randperm(ra_features.size(0))
                            ra_features = ra_features[shuffle_idx]
                        data['antigen_reliability'] = ra_features
                    else:
                        data['antigen_reliability'] = ra_features_cache
                else:
                    data['antigen_reliability'] = ra_features_cache

            if 'antigen_reliability' not in data:
                data['antigen_reliability'] = torch.zeros(0, 8)

            # 加载模板
            template = None
            template_enable = False
            
            if self.pkl_dict:
                # CHANGE BEGIN
                try:
                    template_path = self.pkl_dict.get(id)
                except:
                    print(index)
                    raise KeyError
                if id in ['7t3d_O_Q_G', '7d3k_H_L_2', '5th9_J_N_C', '7uov_H_L_A', '3csy_E_F_LK']:
                    template_path = None
                if template_path and os.path.exists(template_path):    
                    with open (template_path, 'rb') as fin:
                        template = pickle.load(fin)
                    if self.transform is not None:
                        try:
                            template = self.transform(template)
                        except Exception as e:
                            logging.error(f"Error transforming template {id}: {str(e)}")
                            template = None
                    template_enable = True
                    # logging.warning(f'{id} has a template.')
            
            if not template and self.pkl_dict:
                template_path = self.pkl_dict['6wwz_S_D_AB'] # just rand
                with open (template_path, 'rb') as fin:
                    template = pickle.load(fin)
                if self.transform is not None:
                    try:
                        template = self.transform(template)
                    except Exception as e:
                        logging.error(f"Error transforming default template {id}: {str(e)}")
                        template = None
            
            data['template'] = template
            data['template_enable'] = template_enable
            data['id'] = id
            
            return data


@register_dataset('sabdab')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDataset(
        summary_path = cfg.summary_path,
        chothia_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split = cfg.split,
        split_seed = cfg.get('split_seed', 2022),
        transform = transform,
        antigen_reliability_cache_dir = cfg.get('antigen_reliability_cache_dir', None),
        shuffle_reliability = cfg.get('ra_shuffle_reliability', False),
        antigen_reliability_required = cfg.get('antigen_reliability_required', False),
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split, 
        reset=args.reset
    )
    print(dataset[0])
