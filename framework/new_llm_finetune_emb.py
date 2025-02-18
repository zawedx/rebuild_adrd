from common.ml_frame import with_local_info, ml_frame
import torch
import numpy as np
from typing import Any, Dict
from torch import Tensor
import pickle

class V1EmbeddingLayer(torch.nn.Module):
    ''' version 1 of the embedding layer '''
    @with_local_info
    def __init__(self,
                 d_model=None, src_modalities=None, emb_path=None, device=None
                 ) -> None:
        ''' ... '''
        super().__init__()

        if emb_path is not None and emb_path.lower() not in ["", "none"]:
            with open(emb_path, 'rb') as f:
                emb_dict = pickle.load(f)
                self.emb_dict = emb_dict
                ml_frame.set_local_info('emb_dimension', emb_dict['dimension'])
                source = 'GPT 3.5' if 'gpt' in emb_path.lower() else 'OP model'
                print(f'Embedding dict loaded from: {source}')
            for k in src_modalities.keys():
                # Convert embeddings to tensors
                self.emb_dict[k]['embedding'] = torch.tensor(
                    self.emb_dict[k]['embedding'], device=device
                )
        else:
            self.emb_dict = None
            print('Simple linear embedding implemented.')

        # Embedding modules for source modalities
        self.modules_emb_src = torch.nn.ModuleDict()

        if self.emb_dict is None:
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    self.modules_emb_src[k] = torch.nn.Embedding(info['num_categories'], d_model)
                elif info['type'] == 'numerical':
                    self.modules_emb_src[k] = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(info['shape'][0]),
                        torch.nn.Linear(info['shape'][0], d_model)
                    )
                elif info['type'] == 'imaging':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    # unrecognized
                    raise ValueError('{} is an unrecognized data modality'.format(k))
        else:
            print("------------------Experiment is on GPT embeddings-----------------------")
            # 对于二元特征，创建共享的线性层
            self.binary_emb_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.emb_dict['dimension'], d_model)
                for _ in range(2)
            ])
            self.batch_norms = torch.nn.ModuleDict()
            ################# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.binary_emb_layers.requires_grad_(False)
            ###################################################
            self.cached_embeddings = {}
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    num_categories = info['num_categories']
                    if num_categories == 2:
                        # 对于二元特征，使用共享层
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                    else:
                        # 为每个类别创建一个线性层
                        self.modules_emb_src[k] = torch.nn.ModuleList([
                            torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], d_model)
                            for _ in range(num_categories)
                        ])
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                elif info['type'] == 'numerical':
                    # 为每个数值特征创建一个线性层
                    self.modules_emb_src[k] = torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], d_model)
                    # 初始化缓存为 None
                    self.cached_embeddings[k] = None
                    self.batch_norms[k] = torch.nn.BatchNorm1d(info['shape'][0])
                elif info['type'] == 'imaging':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    raise ValueError(f'{k} is an unrecognized data modality')
            #####################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.modules_emb_src.requires_grad_(False)

    def clear_cache(self):
        """Clear cached embeddings. Should be called at the start of each training epoch."""
        for k in self.cached_embeddings.keys():
            self.cached_embeddings[k] = None

    @with_local_info
    def forward(self,
                x: Dict[str, Tensor],
                mask: Dict[str, Tensor],
                skip_embedding=None, src_modalities=None, img_net=None, fusion_stage=None, d_model=None, device=None, emb_droprate=None
                ) -> Dict[str, Tensor]:
        """ ... """
        out_emb = {}
        batch_size = next(iter(x.values())).shape[0]  # Assume all inputs have the same batch size

        for k in src_modalities.keys():
            if skip_embedding is None or not skip_embedding.get(k, False):
                if "img_MRI" in k:
                    # print("img_MRI in ", k)
                    if torch.all(mask[k]):
                        if "swinunetr" in img_net.lower() and fusion_stage == 'late':
                            out_emb[k] = torch.zeros((1,768,4,4,4))
                        else:
                            out_emb[k] = torch.zeros((mask[k].shape[0], d_model)).to(device, non_blocking=True)
                        # print("mask is True, out_emb[k]: ", out_emb[k].size())
                    else:
                        out_emb[k] = self.modules_emb_src[k](x[k])

                elif src_modalities[k]['type'] == 'categorical':
                    num_categories = src_modalities[k]['num_categories']
                    if num_categories == 2: # For binary features
                        indices = x[k].long().to(device)  # Shape: [batch_size]
                        
                        # check if cache available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            # compute intermidiate embeddings and cache them
                            transformed_embeddings = torch.stack([
                                self.binary_emb_layers[i](self.emb_dict[k]['embedding']) for i in range(2)
                            ])  # [2, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=emb_droprate, training=self.training
                        )
                        # Select embeddings based on indices
                        if torch.isnan(transformed_embeddings).any():
                            print(f'Nan in {k}')
                            raise('nan')
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                    else: # For non-binary categorical features
                        # Check if cached embeddings are available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            emb = self.emb_dict[k]['embedding'].to(device)
                            transformed_embeddings = torch.stack([
                                self.modules_emb_src[k][i](emb) for i in range(num_categories)
                            ])  # [num_categories, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=emb_droprate, training=self.training
                        )
                        # Get indices
                        if torch.isnan(transformed_embeddings).any():
                            print(f'Nan in {k}')
                            raise('nan')
                        indices = x[k].long().to(device)  # Shape: [batch_size]
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                elif src_modalities[k]['type'] == 'numerical':
                    # For numerical features
                    transformed_embedding = self.cached_embeddings.get(k)
                    if transformed_embedding is None:
                        emb = self.emb_dict[k]['embedding']
                        transformed_embedding = self.modules_emb_src[k](emb)  # 形状: [d_model]
                        self.cached_embeddings[k] = transformed_embedding
                    # Dropout
                    transformed_embedding = torch.nn.functional.dropout(
                        transformed_embedding, p=emb_droprate, training=self.training
                    )
                    x[k] = self.batch_norms[k](x[k])
                    # Multiply by the normalized feature values
                    out_emb[k] = x[k] * transformed_embedding
                else:
                    out_emb[k] = x[k]
            else:
                out_emb[k] = x[k]
            # print(k)
            # print(x[k])
            assert(not torch.isnan(out_emb[k]).any())
        return out_emb