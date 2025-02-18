from common.ml_frame import with_local_info, ml_frame
from common.ml_logger import MLLogger

import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Any, Self, Type
from functools import wraps
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]

# for DistributedDataParallel
import torch.distributed as dist

from method.focal_loss import SigmoidFocalLoss
from method.dual_T_loss import dual_temperature_loss_func
from framework.transformer import Transformer

from utils import TransformerTrainingDataset, TransformerBalancedTrainingDataset, TransformerValidationDataset, TransformerTestingDataset, Transformer2ndOrderBalancedTrainingDataset
from utils.misc import ProgressBar
from utils.misc import get_metrics_multitask, print_metrics_multitask
from utils.misc import convert_args_kwargs_to_kwargs

class ADRDModel:
    """Primary model class for ADRD framework.
    """    
    @with_local_info
    def __init__(self,
                 local_rank=None, load_from_ckpt=None, tgt_modalities=None, gamma=None, label_fractions=None, src_modalities=None
    ) -> None:
        if local_rank == 0:
            ml_frame.set_local_info('logger', MLLogger())

        self._init_net()

        # initialize optimizer and scheduler
        # CHECK: build model before optimizer
        if not load_from_ckpt:
            self._init_optimizer()
        self._init_scheduler()

        # initialize the focal losses 
        loss_fn = {}
        for k in tgt_modalities:
            alpha = pow((1 - label_fractions[k]), 2)
            loss_fn[k] = SigmoidFocalLoss(
                alpha=alpha,
                gamma=gamma,
                reduction='none'
            )
        ml_frame.set_local_info('loss_fn', loss_fn)

        skip_embedding = {}
        for k, info in src_modalities.items():
            skip_embedding[k] = False
        ml_frame.set_local_info('skip_embedding', skip_embedding)

        # initialize the ranking loss
        ml_frame.set_local_info('lambda_coeff', 0.005)
        ml_frame.set_local_info('margin_loss', torch.nn.MarginRankingLoss(reduction='sum', margin=0.25))

        # initialize the range of epoch
        ml_frame.set_local_info('start_epoch', 0)

    @with_local_info
    def fit(self, x_trn, x_vld, y_trn, y_vld,
            local_rank=None, tgt_modalities=None, criterion=None, start_epoch=None, num_epochs=None, ckpt_path=None, save_intermediate_ckpts=None, verbose=None) -> Self:
        # initialize the dataloaders
        ldr_trn, ldr_vld = self._init_dataloader(x_trn, x_vld, y_trn, y_vld)
        ml_frame.set_local_info('ldr_trn', ldr_trn)
        ml_frame.set_local_info('ldr_vld', ldr_vld)
        
        # to record the best validation performance criterion
        if criterion is not None:
            best_crit = None
            best_crit_AUPR = None
        
        # training loop
        for epoch in range(start_epoch, num_epochs):
            ml_frame.set_local_info('epoch', epoch)
            
            # TODO: -> self.to_train/train()
            ml_frame.set_local_info('training_state', 'train')
            self.net_.train()
            self.train_one_epoch()
            
            ml_frame.set_local_info('training_state', 'validation')
            self.net_.eval()
            self.train_one_epoch()

            # save the model if it has the best validation performance criterion by far
            if criterion is None: continue
            if local_rank != 0: continue
            # is current criterion better than previous best?
            # TODO: save current_metric in train_one_epoch
            current_metric = ml_frame.get_local_info('current_metric')
            curr_crit = np.mean([current_metric[i][criterion] for i in range(len(tgt_modalities))])
            curr_crit_AUPR = np.mean([current_metric[i]["AUC (PR)"] for i in range(len(tgt_modalities))])
            # AUROC
            if best_crit is None or np.isnan(best_crit):
                is_better = True
            elif criterion == 'Loss' and best_crit >= curr_crit:
                is_better = True
            elif criterion != 'Loss' and best_crit <= curr_crit :
                is_better = True
            else:
                is_better = False

            # AUPR
            if best_crit_AUPR is None or np.isnan(best_crit_AUPR):
                is_better_AUPR = True
            elif best_crit_AUPR <= curr_crit_AUPR :
                is_better_AUPR = True
            else:
                is_better_AUPR = False
            # update best criterion
            if is_better_AUPR:
                best_crit_AUPR = curr_crit_AUPR
                if save_intermediate_ckpts:
                    print(f"Saving the model to {ckpt_path[:-3]}_AUPR.pt...")
                    self.save(ckpt_path[:-3]+"_AUPR.pt", epoch)
            if is_better:
                best_crit = curr_crit
                if save_intermediate_ckpts:
                    print(f"Saving the model to {ckpt_path}...")
                    self.save(ckpt_path, epoch)

            if verbose > 2 and local_rank == 0:
                print('Best {}: {}'.format(criterion, best_crit))
                print('Best {}: {}'.format('AUC (PR)', best_crit_AUPR))

        return self

    @with_local_info
    def compute_metrics_ddp(self, scores_list, y_true_list, y_mask_list, losses_list,
                            local_rank=None, tgt_modalities=None):

        # Concatenate local tensors
        scores_local = torch.cat(scores_list)
        y_true_local = torch.cat(y_true_list)
        y_mask_local = torch.cat(y_mask_list)

        # Convert tensors to numpy arrays
        scores_local_np = scores_local.numpy()
        y_true_local_np = y_true_local.numpy()
        y_mask_local_np = y_mask_local.numpy()

        # Gather numpy arrays to rank 0
        if local_rank == 0:
            scores_gathered = [None for _ in range(dist.get_world_size())]
            y_true_gathered = [None for _ in range(dist.get_world_size())]
            y_mask_gathered = [None for _ in range(dist.get_world_size())]
        else:
            scores_gathered = None
            y_true_gathered = None
            y_mask_gathered = None

        dist.gather_object(scores_local_np, scores_gathered, dst=0)
        dist.gather_object(y_true_local_np, y_true_gathered, dst=0)
        dist.gather_object(y_mask_local_np, y_mask_gathered, dst=0)

        # Gather losses to rank 0
        all_losses = None
        for i in range(len(tgt_modalities)):
            if local_rank == 0:
                losses_gathered = [None for _ in range(dist.get_world_size())]
            else:
                losses_gathered = None
            dist.gather_object(losses_list[i], losses_gathered, dst=0)

            if local_rank == 0:
                # Flatten the list of losses
                if all_losses is None:
                    all_losses = [[] for _ in range(len(tgt_modalities))]
                all_losses[i] = [loss for sublist in losses_gathered for loss in sublist]

        # Compute metrics on rank 0
        if local_rank == 0:
            # Concatenate arrays from all processes
            scores = np.concatenate(scores_gathered)
            y_true = np.concatenate(y_true_gathered)
            y_mask = np.concatenate(y_mask_gathered)
            
            TH_map = 0.5
            y_pred = (scores > TH_map).astype(int)
            y_prob = scores
            met_trn = get_metrics_multitask(
                y_true,
                y_pred,
                y_prob,
                y_mask
            )

            # Add computed losses to metrics
            for i in range(len(tgt_modalities)):
                met_trn[i]['Loss'] = np.mean(all_losses[i])
            
            ml_frame.set_local_info('current_metric', met_trn)
        else:
            return None

    @with_local_info
    def train_one_epoch(self,
                        training_state=None, epoch=None, verbose=None, local_rank=None, scheduler=None, device=None, tgt_modalities=None, loss_fn=None, clr_ratio=None, optimizer=None, logger=None):
        if training_state == 'train':
            loader = ml_frame.get_local_info('ldr_trn')
            # set self.scheduler according freezing or not
            scheduler.step(epoch)

            # set model to train mode
            # TODO: modify this part
            torch.set_grad_enabled(True)
            self.net_.train()
        elif training_state == 'validation':
            loader = ml_frame.get_local_info('ldr_vld')
            # set model to validation mode
            torch.set_grad_enabled(False)
            self.net_.eval()
        
        # progress bar for batch loops
        if verbose > 1 and local_rank == 0: 
            pbr_batch = ProgressBar(len(loader.dataset), 'Epoch {:03d} ({:3s})'.format(epoch, training_state[:3].upper()))

        scores_trn, y_true_trn, y_mask_trn = [], [], []
        losses_trn = [[] for _ in tgt_modalities]
        
        for n_iter, (x_batch, y_batch, mask, y_mask) in enumerate(loader):
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(device) for k in x_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(device) for k in y_batch}
            mask = {k: mask[k].to(device) for k in mask}
            y_mask = {k: y_mask[k].to(device) for k in y_mask}
            
            with torch.autocast(
                device_type = 'cpu' if device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if device == 'cpu' else torch.float16,
            ):
                # 2-step: pretrain and contrastive learning
                representation, outputs = self.net_(x_batch, mask)
                
                # calculate multitask loss
                loss = 0

                for i, k in enumerate(tgt_modalities):
                    loss_task = loss_fn[k](outputs[k], y_batch[k])
                    msk_loss_task = loss_task * y_mask[k]
                    msk_loss_mean = msk_loss_task.sum() / (torch.sum(torch.stack(list(y_mask.values())))+1e-12)
                    loss += msk_loss_mean
                    losses_trn[i] += msk_loss_task.detach().cpu().numpy().tolist()

                # TODO: add nce loss to loss metric dict
                nce_loss = 0
                for i, k in enumerate(tgt_modalities): 
                    rept = representation[i]
                    y_k = y_batch[k]
                    y_mask_k = y_mask[k]
                    n_ture = (y_k * y_mask_k).sum()
                    # w = (n_ture**2 + (y_mask.sum() - n_ture)**2) / y_mask_k.sum()**2
                    w = 1
                    nce_loss += w*dual_temperature_loss_func(rept, y_k, y_mask_k)
                # print(f"nce_loss: {nce_loss}")
                
            if training_state == 'train':
                # backward
                # gamma = 0 when [0, 127], 1 when [128, 255]
                gamma = clr_ratio if epoch < 128 else 0
                final_loss = (loss + gamma * nce_loss) / (gamma + 1)
                final_loss.backward()
                
                # update parameters
                # TODO: why niter != 0
                if n_iter != 0:
                    optimizer.step()
                    optimizer.zero_grad()

            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_trn.append(outputs.detach().to(torch.float).cpu())
            y_true_trn.append(y_batch.cpu())
            y_mask_trn.append(y_mask.cpu())

            # update progress bar
            if verbose > 1 and local_rank == 0:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size*dist.get_world_size(), {})
                pbr_batch.refresh() 

            # clear cuda cache
            torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if verbose > 1 and local_rank == 0:
            pbr_batch.close()

        # calculate and print training performance metrics
        # TODO: or move compute_metrics_ddp to outter, revise here -> frame.set(ddp_result)
        self.compute_metrics_ddp(
            scores_list=scores_trn,
            y_true_list=y_true_trn,
            y_mask_list=y_mask_trn,
            losses_list=losses_trn
        )
        
        if local_rank == 0:
            # log metrics to logger
            log_what = ['Loss', 'Balanced Accuracy', 'AUC (ROC)', 'AUC (PR)']
            for log_info in log_what:
                logger.info(log_info)

        if verbose > 2 and local_rank == 0:
            print_metrics_multitask(ml_frame.get_local_info('current_metric'))

    def save(self, filepath: str, epoch: int) -> None:
        """Save the model to the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param epoch: _description_
        :type epoch: int
        """
        pass
    
    def load(self, filepath: str, map_location: str = 'cpu', img_dict=None) -> None:
        """Load a model from the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param map_location: _description_, defaults to 'cpu'
        :type map_location: str, optional
        :param img_dict: _description_, defaults to None
        :type img_dict: _type_, optional
        """
        pass

    def to(self, device: str) -> Self:
        """Mount the model to the given device. 

        :param device: _description_
        :type device: str
        :return: _description_
        :rtype: Self
        """        
        self.device = device
        if hasattr(self, 'net_'): self.net_ = self.net_.to(device)
        if hasattr(self, 'img_model'): self.img_model = self.img_model.to(device)
        return self
    
    @with_local_info
    def _init_net(self,
                  device=None, local_rank=None, load_from_ckpt=None, ckpt_path=None):
        """ ... """
        sys.stderr.write(f"I'm on {device}, Rank: {local_rank}\n")
        
        if load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                self.load(ckpt_path, map_location=device)
            except:
                print("Cannot load from checkpoint. Initializing new model...")
                load_from_ckpt = False

        if not load_from_ckpt:
            self.net_ = Transformer()
            # sys.stderr.write(f"Thread {self.rank}'s net_ initialized.\n")
            
            # intialize model parameters using xavier_uniform
            for name, p in self.net_.named_parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
        

    @with_local_info
    def _init_dataloader(self, x_trn, x_vld, y_trn, y_vld,
                         img_train_trans=None, img_vld_trans=None, balanced_sampling=None, src_modalities=None, tgt_modalities=None, _dataloader_num_workers=None, batch_size=None, parallel=None):    
        # initialize dataset and dataloader
        if balanced_sampling:
            dat_trn = Transformer2ndOrderBalancedTrainingDataset(
                x_trn, y_trn,
                src_modalities,
                tgt_modalities,
                dropout_rate = 0.2,
                dropout_strategy = 'permutation',
                img_transform=img_train_trans,
            )
        else:
            dat_trn = TransformerTrainingDataset(
                x_trn, y_trn,
                src_modalities,
                tgt_modalities,
                dropout_rate = 0.2,
                dropout_strategy = 'permutation',
                img_transform=img_train_trans,
            )

        dat_vld = TransformerValidationDataset(
            x_vld, y_vld,
            src_modalities,
            tgt_modalities,
            img_transform=img_vld_trans,
        )

        self.spl_trn = DistributedSampler(dat_trn, shuffle=True) if parallel else None
        ldr_trn = DataLoader(
            dataset = dat_trn,
            batch_size = batch_size,
            shuffle = (self.spl_trn is None),
            drop_last = False,
            sampler = self.spl_trn,
            num_workers = _dataloader_num_workers,
            collate_fn = TransformerTrainingDataset.collate_fn,
            # pin_memory = True
        )

        self.spl_vld = DistributedSampler(dat_vld, shuffle=True) if parallel else None
        ldr_vld = DataLoader(
            dataset = dat_vld,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            sampler = self.spl_vld,
            num_workers = _dataloader_num_workers,
            collate_fn = TransformerValidationDataset.collate_fn,
            # pin_memory = True
        )

        return ldr_trn, ldr_vld
    
    @with_local_info
    def _init_optimizer(self,
                        lr=None, weight_decay=None):
        params = [param for param in self.net_.parameters()]
        ml_frame.set_local_info('optimizer',
            torch.optim.AdamW([
                    {'params': params, 'lr': lr}
                ],
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            )
        )
    
    @with_local_info
    def _init_scheduler(self,
                        optimizer=None, verbose=None):
        ml_frame.set_local_info('scheduler',
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=64,
                T_mult=1,
                eta_min=0,
                verbose=(verbose > 2)
            )
        )