import pandas as pd
import os
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import umap
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from ecg.datasets.ECGDataset import ECGDataset
from echo.datasets.EchoDataset import EchoDataset
from echo.datasets.EchoStudyDataset import EchoStudyDataset

class Retriever(nn.Module):
    def __init__(
        self, 
        cfg: DictConfig,
        save_dir: str
    ):
        
        super(Retriever, self).__init__()

        self.cfg = cfg
        self.save_dir = save_dir

        self.phenotype = cfg.dataset.phenotype
        
        # Load model
        if cfg.model.experiment_path:
            model_cfg = OmegaConf.load(os.path.join(cfg.model.experiment_path, 'config.yaml'))

            # Override the model's config for view dropout
            if 'view_dropout' in model_cfg.model.echo:
                model_cfg.model.echo.view_dropout.use = self.cfg.model.echo.view_dropout.use
            else:
                OmegaConf.update(model_cfg, 'model.echo.view_dropout', {'use': self.cfg.model.echo.view_dropout.use}, merge=True)

            # If the model was trained on echo views but we want to do retrieval of echo studies, aggregate the echo view embeds using mean
            if self.cfg.model.echo.view_aggregation.use == True and 'use' in model_cfg.model.echo.view_aggregation:
                if model_cfg.model.echo.view_aggregation.use == False:
                    model_cfg.model.echo.view_aggregation.strategy = 'mean'
            
            if 'CLIP' in cfg.model.experiment_path:
                from multimodal.models.Multimodal_CLIP import MultimodalECGEchoCLIP
                self.model = MultimodalECGEchoCLIP(model_cfg, save_dir=None)
            
            if cfg.model.checkpoint_mode == 'last':
                checkpoint_name = 'checkpoints/last.ckpt'
                checkpoint_path = os.path.join(cfg.model.experiment_path, checkpoint_name)
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                checkpoint_dict = checkpoint['state_dict']
            elif cfg.model.checkpoint_mode == 'epoch':
                assert cfg.model.checkpoint_name is not None
                checkpoint_path = os.path.join(cfg.model.experiment_path, cfg.model.checkpoint_name)
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                checkpoint_dict = checkpoint['state_dict']
            elif cfg.model.checkpoint_mode == 'component':
                checkpoint_dict = {}
                for ckpt_file in ['ecg_multimodal.ckpt', 'echo_multimodal.ckpt', 'echo_projection_for_agg.ckpt', 'echo_view_aggregator.ckpt',
                                  'ecg_alignment_view.ckpt', 'echo_alignment_view.ckpt', 'ecg_alignment_study.ckpt', 'echo_alignment_study.ckpt',
                                  'token_aggregator.ckpt', 'ecg_alignment_global_token.ckpt', 'echo_alignment_global_token.ckpt']:
                    if os.path.exists(os.path.join(cfg.model.experiment_path, 'checkpoints', ckpt_file)):
                        if ckpt_file == 'ecg_multimodal.ckpt':
                            prefix = 'ecg_encoder'
                        elif ckpt_file == 'echo_multimodal.ckpt':
                            prefix = 'echo_encoder'
                        elif ckpt_file == 'echo_projection_for_agg.ckpt':
                            prefix = 'echo_projection_for_agg'
                        elif ckpt_file == 'echo_view_aggregator.ckpt':
                            prefix = 'echo_view_aggregator'
                        elif ckpt_file == 'ecg_alignment_view.ckpt':
                            prefix = 'ecg_alignment_view'
                        elif ckpt_file == 'echo_alignment_view.ckpt':
                            prefix = 'echo_alignment_view'
                        elif ckpt_file == 'ecg_alignment_study.ckpt':
                            prefix = 'ecg_alignment_study'
                        elif ckpt_file == 'echo_alignment_study.ckpt':
                            prefix = 'echo_alignment_study'
                        elif ckpt_file == 'token_aggregator.ckpt':
                            prefix = 'token_aggregator'
                        elif ckpt_file == 'ecg_alignment_global_token.ckpt':
                            prefix = 'ecg_alignment_global_token'
                        elif ckpt_file == 'echo_alignment_global_token.ckpt':
                            prefix = 'echo_alignment_global_token'
                        checkpoint = torch.load(os.path.join(cfg.model.experiment_path, 'checkpoints', ckpt_file))
                        new_checkpoint = {}
                        for k, v in checkpoint['model'].items():
                            new_checkpoint[f'{prefix}.{k}'] = v
                        checkpoint_dict.update(new_checkpoint)
            self.model.load_state_dict(checkpoint_dict, strict=False)
            print(f'Loaded checkpoint into model')

        self.model.eval()

        # Override some configs
        OmegaConf.set_struct(self.cfg, False)
        self.cfg.dataset.ecg.sig_len = model_cfg.model.ecg.time_steps
        self.cfg.dataset.echo.img_size = model_cfg.model.echo.img_size
        self.cfg.dataset.echo.num_frames = model_cfg.model.echo.num_frames
        self.cfg.dataset.echo.num_channels = model_cfg.model.echo.num_channels
        OmegaConf.set_struct(self.cfg, True)

        # Placeholder
        self.eval_mode = None
        self.device = None

    def setup(self, stage: str = None) -> None:
        if self.eval_mode is None:
            return

        assert self.eval_mode in ['train', 'val', 'test']
        print(f'Currently in {self.eval_mode} mode')
        print(f'Model device: {self.device}')

        # Move model to device
        self.model = self.model.to(self.device)

        # Precompute
        self._load_pairs()
        self._calculate_groundtruth()

        self._embed_ecgs()
        if self.cfg.model.echo.view_aggregation.use:
            self._embed_echo_studies()
        else:
            self._embed_echos()
        self._calculate_similarity_matrix()
        self._calculate_prediction()

    def forward(self) -> dict:
        out = {}

        metrics = {}
        metrics_ecg_to_echo = self.calculate_metrics(source_modality='ecg', target_modality='echo')
        metrics_echo_to_ecg = self.calculate_metrics(source_modality='echo', target_modality='ecg')
        metrics.update(metrics_ecg_to_echo)
        metrics.update(metrics_echo_to_ecg)
        out['metrics'] = metrics

        if self.cfg.plot_umap:
            image = self.plot_3d_umap()
            out['umap'] = image
        
        return out

    def _load_pairs(self) -> None:
        # load pairs
        if self.cfg.model.echo.view_aggregation.use:
            self.pairs = pd.read_csv(self.cfg.dataset.paths[f'data_{self.eval_mode}_study'])
        else:
            self.pairs = pd.read_csv(self.cfg.dataset.paths[f'data_{self.eval_mode}'])
        
        # retrieve phenotypes
        metadata = pd.read_csv(self.cfg.dataset.paths['metadata'])
        phenotypes = ['lvedv', 'lvesv', 'lvsv', 'lvef']
        for col in phenotypes:
            if col in self.pairs.columns:
                self.pairs.drop(columns=col, inplace=True)
        self.pairs = self.pairs.merge(metadata[['filename'] + phenotypes], how='left', left_on='echo_filename', right_on='filename')

        if self.phenotype is not None:
            self.pairs = self.pairs[~self.pairs[self.phenotype].isna()].reset_index(drop=True)
            std = self.pairs[['echo_study_id', self.phenotype]].drop_duplicates()[self.phenotype].std().item()
            self.threshold = self.cfg.train.tolerance * std

        if self.cfg.model.echo.view_aggregation.use:
            self.ecg_filenames = list(self.pairs['ecg_filepath'].unique())
            self.echo_studies = list(self.pairs['echo_study_id'].unique())
            self.num_ecg = len(self.ecg_filenames)
            self.num_echo_study = len(self.echo_studies)
            print(f'There are {self.num_ecg} ECGs and {self.num_echo_study} echo studies')
        else:
            self.ecg_filenames = list(self.pairs['ecg_filepath'].unique())
            self.echo_filenames = list(self.pairs['echo_filepath'].unique())
            self.num_ecg = len(self.ecg_filenames)
            self.num_echo = len(self.echo_filenames)
            print(f'There are {self.num_ecg} ECGs and {self.num_echo} echos')

    def _embed_dataset(self, dataloader: DataLoader, forward_fn: Callable, modality: str) -> torch.Tensor:
        assert modality in ['ecg', 'echo']
        embeddings = None
        for batch in tqdm(dataloader):
            # batch = {k: v.to(self.device) for k, v in batch.items()}
            batch[modality] = batch[modality].to(self.device)
            if 'attn_mask' in batch: # relevant for forward_echo_study
                mask = batch['attn_mask'].float().to(self.device)
                out = forward_fn(batch[modality], mask=mask)
            else:
                out = forward_fn(batch[modality])
            
            if f'{modality}_global_token_aligned' in out:
                global_token_aligned = out[f'{modality}_global_token_aligned']
            else: # Hierarchical CLIP
                if self.cfg.model.echo.view_aggregation.use:
                    global_token_aligned = out[f'{modality}_global_token_study']
                else:
                    global_token_aligned = out[f'{modality}_global_token_view']
            
            if embeddings is None:
                embeddings = global_token_aligned
            else:
                embeddings = torch.cat([embeddings, global_token_aligned], dim=0)

        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def _embed_ecgs(self):
        embedding_file_suffix = '' if self.phenotype is None else f'_{self.phenotype}'
        if os.path.exists(os.path.join(self.save_dir, f'{self.eval_mode}_ecg_embeddings{embedding_file_suffix}.pt')):
            print('ECG embeddings loaded')
            self.ecg_global_tokens_aligned = torch.load(os.path.join(self.save_dir, f'{self.eval_mode}_ecg_embeddings{embedding_file_suffix}.pt'), weights_only=False)
        else:
            print('Embedding ECGs...')
            dataset = ECGDataset(self.cfg, self.ecg_filenames)
            dataloader = DataLoader(dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False)
            self.ecg_global_tokens_aligned = self._embed_dataset(dataloader, self.model.forward_ecg, 'ecg') # (num_ecg, D)

            if self.cfg.save_embeddings:
                eval_mode = self.eval_mode
                torch.save(self.ecg_global_tokens_aligned, os.path.join(self.save_dir, f'{eval_mode}_ecg_embeddings{embedding_file_suffix}.pt'))
                torch.save(self.ecg_filenames, os.path.join(self.save_dir, f'{eval_mode}_ecg_filepaths{embedding_file_suffix}.pt'))
        
    @torch.no_grad()
    def _embed_echos(self):
        embedding_file_suffix = '' if self.phenotype is None else f'_{self.phenotype}'
        if os.path.exists(os.path.join(self.save_dir, f'{self.eval_mode}_echo_embeddings{embedding_file_suffix}.pt')):
            print('Echo embeddings loaded')
            self.echo_global_tokens_aligned = torch.load(os.path.join(self.save_dir, f'{self.eval_mode}_echo_embeddings{embedding_file_suffix}.pt'), weights_only=False)
        else:
            print('Embedding echos...')
            dataset = EchoDataset(self.cfg, self.echo_filenames)
            dataloader = DataLoader(dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False)
            self.echo_global_tokens_aligned = self._embed_dataset(dataloader, self.model.forward_echo, 'echo') # (num_echo, D)

            if self.cfg.save_embeddings:
                if self.eval_mode == 'pretrain/old':
                    eval_mode = 'pretrain'
                else:
                    eval_mode = self.eval_mode
                torch.save(self.echo_global_tokens_aligned, os.path.join(self.save_dir, f'{eval_mode}_echo_embeddings{embedding_file_suffix}.pt'))
                torch.save(self.echo_filenames, os.path.join(self.save_dir, f'{eval_mode}_echo_filepaths{embedding_file_suffix}.pt'))
    
    @torch.no_grad()
    def _embed_echo_studies(self):
        embedding_file_suffix = '' if self.phenotype is None else f'_{self.phenotype}'
        if os.path.exists(os.path.join(self.save_dir, f'{self.eval_mode}_echo_study_embeddings{embedding_file_suffix}.pt')):
            print('Echo study embeddings loaded')
            self.echo_global_tokens_aligned = torch.load(os.path.join(self.save_dir, f'{self.eval_mode}_echo_study_embeddings{embedding_file_suffix}.pt'), weights_only=False)
        else:
            print('Embedding echo studies...')
            self.echo_study_df = self.pairs[['echo_study_id', 'echo_filepath', 'echo_embed_idx']].drop_duplicates().reset_index(drop=True)
            dataset = EchoStudyDataset(self.cfg, self.echo_study_df)
            dataloader = DataLoader(dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False)
            # dataloader = DataLoader(dataset, batch_size=self.cfg.dataset.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
            # self.echo_global_tokens_aligned = self._embed_dataset(dataloader, self.model.forward_echo_study, 'echo') # (num_echo, D)
            
            self.echo_global_tokens_aligned = []
            for batch in tqdm(dataloader):
                echo_view_tokens = batch['echo_embed'].float().to(self.device) # (B, max_n_views, D)
                mask = batch['attn_mask'].float().to(self.device)

                if self.cfg.model.echo.view_aggregation.strategy == 'mean' and not hasattr(self.model, 'echo_projection_for_agg'): # !!! Only for retrieval purposes (ecg <-> echo view models)
                    echo_study_token = echo_view_tokens.mean(dim=1)
                else:
                    echo_view_tokens = self.model.echo_projection_for_agg(echo_view_tokens)
                    echo_study_token = self.model.echo_view_aggregator(echo_view_tokens, mask=mask)
                
                echo_global_token_aligned = self.model.echo_alignment_global_token(echo_study_token) # (B, D)
                self.echo_global_tokens_aligned.append(echo_global_token_aligned)
            
            self.echo_global_tokens_aligned = torch.cat(self.echo_global_tokens_aligned, dim=0) # (num_echo, D)

            self.echo_global_tokens_aligned = F.normalize(self.echo_global_tokens_aligned, dim=-1)

            if self.cfg.save_embeddings:
                eval_mode = self.eval_mode
                torch.save(self.echo_global_tokens_aligned, os.path.join(self.save_dir, f'{eval_mode}_echo_study_embeddings{embedding_file_suffix}.pt'))
                torch.save(self.echo_studies, os.path.join(self.save_dir, f'{eval_mode}_echo_filepaths{embedding_file_suffix}.pt'))

    def _calculate_similarity_matrix(self):
        # Cosine similarity
        self.sim_matrix = torch.matmul(self.ecg_global_tokens_aligned, self.echo_global_tokens_aligned.T) # (num_ecg, num_echo)
        print(f'Shape of similarity matrix: {self.sim_matrix.shape}')

    def _calculate_groundtruth(self):
        print('Creating groundtruth...')
        # Precompute: map filename → index
        ecg_to_idx = {f: i for i, f in enumerate(self.ecg_filenames)} # {ecg_filepath: ecg_idx}
        if self.cfg.model.echo.view_aggregation.use:
            echo_to_idx = {f: i for i, f in enumerate(self.echo_studies)} # {echo_study_id: echo_idx}
        else:
            echo_to_idx = {f: i for i, f in enumerate(self.echo_filenames)} # {echo_filepath: echo_idx}

        # Pre-group pairs by ECG and by Echo in one pass
        # This avoids repeated DataFrame filtering
        if self.phenotype is not None:
            ecg_groups = {} # {ecg_filepath: [echo_filepath/echo_study_id]}
            for ecg_filename in tqdm(self.ecg_filenames):
                ecg_phenotype_value = self.pairs[self.pairs['ecg_filepath'] == ecg_filename][['echo_study_id', self.phenotype]].drop_duplicates()[self.phenotype].mean().item()
                self.pairs['is_echo_gt'] = self.pairs[self.phenotype].apply(
                    lambda x: (ecg_phenotype_value - self.threshold) <= x <= (ecg_phenotype_value + self.threshold)
                )
                echo_gts = self.pairs[self.pairs['is_echo_gt']]['echo_study_id'].unique().tolist()
                ecg_groups[ecg_filename] = echo_gts
            
            echo_groups = {} # {echo_filepath/echo_study_id: [ecg_filepath]}
            if self.cfg.model.echo.view_aggregation.use:
                for echo_study in tqdm(self.echo_studies):
                    echo_study = int(echo_study)
                    echo_phenotype_value = self.pairs[self.pairs['echo_study_id'] == echo_study][['ecg_filepath', self.phenotype]].drop_duplicates()[self.phenotype].mean().item()
                    self.pairs['is_ecg_gt'] = self.pairs[self.phenotype].apply(
                        lambda x: (echo_phenotype_value - self.threshold) <= x <= (echo_phenotype_value + self.threshold)
                    )
                    ecg_gts = self.pairs[self.pairs['is_ecg_gt']]['ecg_filepath'].unique().tolist()
                    echo_groups[echo_study] = ecg_gts
            else:
                for echo_filename in tqdm(self.echo_filenames):
                    echo_phenotype_value = self.pairs[self.pairs['echo_filepath'] == echo_filename][['ecg_filepath', self.phenotype]].drop_duplicates()[self.phenotype].mean().item()
                    self.pairs['is_ecg_gt'] = self.pairs[self.phenotype].apply(
                        lambda x: (echo_phenotype_value - self.threshold) <= x <= (echo_phenotype_value + self.threshold)
                    )
                    ecg_gts = self.pairs[self.pairs['is_ecg_gt']]['ecg_filepath'].unique().tolist()
                    echo_groups[echo_filename] = ecg_gts
        else:
            ecg_groups = {} # {ecg_filepath: [echo_filepath/echo_study_id]}
            echo_groups = {} # {echo_filepath/echo_study_id: [ecg_filepath]}

            if self.cfg.model.echo.view_aggregation.use:
                ecg_echo = list(zip(self.pairs["ecg_filepath"], self.pairs["echo_study_id"]))
                ecg_echo = list(dict.fromkeys(ecg_echo)) # [(ecg_filepath, echo_study_id)]
            else:
                ecg_echo = zip(self.pairs["ecg_filepath"], self.pairs["echo_filepath"]) # [(ecg_filepath, echo_filepath)]
            for ecg_f, echo_f in ecg_echo:
                ecg_groups.setdefault(ecg_f, []).append(echo_f)
                echo_groups.setdefault(echo_f, []).append(ecg_f)

        # Build groundtruth using list comprehensions + dict lookups
        self.gt_ecg_to_echo = {
            ecg_to_idx[f]: [echo_to_idx[g] for g in ecg_groups.get(f, [])]
            for f in tqdm(self.ecg_filenames)
        } # {ecg_idx: [echo_idx]}
        if self.cfg.model.echo.view_aggregation.use:
            self.gt_echo_to_ecg = {
                echo_to_idx[f]: [ecg_to_idx[g] for g in echo_groups.get(f, [])]
                for f in tqdm(self.echo_studies)
            } # {echo_idx: [ecg_idx]}
        else:
            self.gt_echo_to_ecg = {
                echo_to_idx[f]: [ecg_to_idx[g] for g in echo_groups.get(f, [])]
                for f in tqdm(self.echo_filenames)
            } # {echo_idx: [ecg_idx]}

    def _calculate_prediction(self):
        print('Calculating prediction...')
        torch.cuda.memory.empty_cache()
        sim = self.sim_matrix.to(torch.device('cpu')) # to avoid memory issues
        _, self.pred_ecg_to_echo = torch.sort(sim, dim=1, descending=True) # (num_ecg, num_echo)
        _, self.pred_echo_to_ecg = torch.sort(sim, dim=0, descending=True) 
        self.pred_echo_to_ecg = self.pred_echo_to_ecg.permute(1, 0) # (num_echo, num_ecg)

    def find_matching_echo_for_ecg(self, ecg_idx, top_k=None):
        if ecg_idx >= self.num_ecg:
            raise Exception
        print(f'Finding a matching echo for ECG {self.ecg_filenames[ecg_idx]}')
        sim_scores = self.sim_matrix[ecg_idx, :]
        echo_indices_sim_scores_sorted = torch.sort(sim_scores, descending=True).indices
        if self.cfg.model.echo.view_aggregation.use:
            echo_studies_sorted = [self.echo_studies[i] for i in echo_indices_sim_scores_sorted]
            if top_k is not None:
                echo_studies_sorted = echo_studies_sorted[:top_k]
            gt_echo_studies = list(self.pairs[self.pairs['ecg_filepath'] == self.ecg_filenames[ecg_idx]]['echo_study_id'])
            return {
                'pred': echo_studies_sorted,
                'gt': gt_echo_studies
            }
        else:
            echo_filenames_sorted = [self.echo_filenames[i] for i in echo_indices_sim_scores_sorted]
            if top_k is not None:
                echo_filenames_sorted = echo_filenames_sorted[:top_k]
            gt_echo_filenames = list(self.pairs[self.pairs['ecg_filepath'] == self.ecg_filenames[ecg_idx]]['echo_filepath'])
            return {
                'pred': echo_filenames_sorted,
                'gt': gt_echo_filenames
            }

    def find_matching_ecg_for_echo(self, echo_idx, top_k=None):
        if echo_idx >= self.num_echo:
            raise Exception
        print(f'Finding a matching ECG for echo {self.echo_filenames[echo_idx]}')
        sim_scores = self.sim_matrix[:, echo_idx]
        ecg_indices_sim_scores_sorted = torch.sort(sim_scores, descending=True).indices
        ecg_filenames_sorted = [self.ecg_filenames[i] for i in ecg_indices_sim_scores_sorted]
        if top_k is not None:
            ecg_filenames_sorted = ecg_filenames_sorted[:top_k]
        if self.cfg.model.echo.view_aggregation.use:
            gt_ecg_filenames = list(self.pairs[self.pairs['echo_study_id'] == self.echo_studies[echo_idx]]['ecg_filepath'])
        else:
            gt_ecg_filenames = list(self.pairs[self.pairs['echo_filepath'] == self.echo_filenames[echo_idx]]['ecg_filepath'])
        return {
            'pred': ecg_filenames_sorted,
            'gt': gt_ecg_filenames
        }

    def calculate_metrics(self, source_modality='ecg', target_modality='echo'):
        assert source_modality in ['ecg', 'echo'] and target_modality in ['ecg', 'echo']
        assert source_modality != target_modality

        metrics = dict()

        if source_modality == 'ecg':
            pred_src_to_target = self.pred_ecg_to_echo
            gt_src_to_target = self.gt_ecg_to_echo
            num_samples = self.num_ecg
        elif source_modality == 'echo':
            pred_src_to_target = self.pred_echo_to_ecg
            gt_src_to_target = self.gt_echo_to_ecg
            if self.cfg.model.echo.view_aggregation.use:
                num_samples = self.num_echo_study
            else:
                num_samples = self.num_echo

        # Recall@k and precision@k
        for k in [1, 3, 5, 10]:
            recall, precision = 0, 0
            top_k_pred_src_to_target = pred_src_to_target[:, :k]
            for idx in tqdm(range(num_samples)):

                top_k_pred_indices = top_k_pred_src_to_target[idx].tolist()
                gt_indices = gt_src_to_target[idx]
                predicted_gts = set(top_k_pred_indices).intersection(set(gt_indices))
                if len(gt_indices) > 0:
                    recall += len(predicted_gts) / len(gt_indices)
                precision += len(predicted_gts) / k
        
            recall /= num_samples
            precision /= num_samples

            metrics[f'{self.eval_mode}/recall@{k}_{source_modality}_to_{target_modality}'] = recall
            metrics[f'{self.eval_mode}/prec@{k}_{source_modality}_to_{target_modality}'] = precision
        
        # mAP
        ap_sum = 0
        for idx in tqdm(range(num_samples)):
            pred_indices = pred_src_to_target[idx].tolist()
            gt_indices = gt_src_to_target[idx]

            if len(gt_indices) > 0:
                num_hits = 0
                precisions = []
                for rank, pred in enumerate(pred_indices, start=1):
                    if pred in gt_indices:
                        num_hits += 1
                        precisions.append(num_hits / rank)
                if len(precisions) > 0:
                    ap_i = np.mean(precisions)
                else:
                    ap_i = 0.0
                ap_sum += ap_i

        mAP = ap_sum / num_samples

        metrics[f'{self.eval_mode}/mAP_{source_modality}_to_{target_modality}'] = mAP

        # Median & mean rank
        ranks = []
        # median_ranks, mean_ranks = [], []
        for idx in tqdm(range(num_samples)):
            pred_indices = pred_src_to_target[idx].tolist()
            gt_indices = gt_src_to_target[idx]

            rank_found = None
            for rank, pred_idx in enumerate(pred_indices):
                if pred_idx in gt_indices:
                    rank_found = rank + 1  # +1 for 1-based ranks
                    break
            if rank_found == None:
                rank_found = len(pred_indices) + 1
            
            ranks.append(rank_found)
        median_rank = np.median(ranks)
        mean_rank = np.mean(ranks)
        
        metrics[f'{self.eval_mode}/MdR_{source_modality}_to_{target_modality}'] = median_rank
        metrics[f'{self.eval_mode}/MnR_{source_modality}_to_{target_modality}'] = mean_rank

        # Alignment score (supposed to be symmetrical)
        alignment_scores = []
        for idx in tqdm(range(num_samples)):
            gt_indices = gt_src_to_target[idx]
            if source_modality == 'ecg':
                sim_scores = [self.sim_matrix[idx, gt].cpu().item() for gt in gt_indices]
            elif source_modality == 'echo':
                sim_scores = [self.sim_matrix[gt, idx].cpu().item() for gt in gt_indices]
            alignment_scores.append(np.mean(sim_scores))
        
        alignment_score = np.mean(alignment_scores)
        
        metrics[f'{self.eval_mode}/alignment_score_{source_modality}_to_{target_modality}'] = alignment_score

        return metrics
    
    def plot_3d_umap(self):
        umap_model = umap.UMAP(
            n_neighbors=self.cfg.umap_n_neighbors,
            n_components=3,
            min_dist=self.cfg.umap_min_dist,
            metric='euclidean',
            random_state=self.cfg.seed
        )
        all_embeddings = torch.cat([self.ecg_global_tokens_aligned, self.echo_global_tokens_aligned], dim=0) # (N_ecg + N_echo, D)
        all_embeddings = F.normalize(all_embeddings, dim=-1)
        all_embeddings = all_embeddings.cpu().numpy()
        X_umap_3d = umap_model.fit_transform(all_embeddings)
        df_umap_3d = pd.DataFrame(X_umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
        df_umap_3d['label'] = [0 for i in range(len(self.ecg_global_tokens_aligned))] + [1 for i in range(len(self.echo_global_tokens_aligned))]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            df_umap_3d['UMAP1'],
            df_umap_3d['UMAP2'],
            df_umap_3d['UMAP3'],
            c=df_umap_3d['label'],
            cmap='Spectral',
            s=20,
            alpha=0.8
        )

        ax.set_title('3D UMAP Projection of the ECG-Echo Dataset', pad=20)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')

        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label('Label (0: ECG, 1: Echo)')

        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)  # Free up memory

        # Convert to PIL image
        image = Image.open(buf)

        return image

