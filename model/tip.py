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

sys.path.append('/openbayes/home/NEW/rebuild_adrd/baseline/')
from TIP.utils.clip_loss import CLIPLoss
from TIP.utils.reconstruct_loss import ReconstructionLoss

from TIP.models.Tip_utils.Tip_pretraining import Pretraining

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class TIPModel(Pretraining):
    '''
    Tabular-Imaging Pretraining
    '''
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        # Imaging
        self.initialize_imaging_encoder_and_projector()
        
        if self.hparams.imaging_pretrain_checkpoint:
            self.load_pretrained_imaging_weights()
        
        # Tabular 
        self.initialize_tabular_encoder_and_projector()

        # Multimodal
        self.initialize_multimodal_encoder_and_predictor()

        # image tabular matching 
        self.itm_head = nn.Linear(self.hparams.multimodal_embedding_dim, 2)

        # loss
        nclasses = hparams.batch_size
        self.criterion_val_itc = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
        self.criterion_train_itc = self.criterion_val_itc
        self.criterion_tr = ReconstructionLoss(num_cat=self.hparams.num_cat, cat_offsets=self.encoder_tabular.cat_offsets, num_con=self.hparams.num_con)
        self.criterion_itm = nn.CrossEntropyLoss(reduction='mean')
        
        self.initialize_classifier_and_metrics(nclasses, nclasses)

        print(f'Tabular model, multimodal: {self.encoder_tabular}\n{self.projector_tabular}')
        print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projector_imaging}')
        print(f'Multimodal model: {self.encoder_multimodal}')
        print(f'Predictor model, tabular: {self.predictor_tabular}')
    
    def cal_image_tabular_matching_loss(self, image_embeddings: torch.Tensor, tabular_embeddings: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        current_device = image_embeddings.device
        output_pos = self.forward_multimodal_feature(tabular_features=tabular_embeddings, image_features=image_embeddings)
        B = image_embeddings.shape[0]
        # get negative pairs
        with torch.no_grad():
            weights_i2t = F.softmax(logits, dim=1)+1e-4
            weights_i2t.fill_diagonal_(0)
            weights_t2i = F.softmax(logits.T, dim=1)+1e-4
            weights_t2i.fill_diagonal_(0)
        
        tabular_embeddings_neg = torch.zeros_like(tabular_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            tabular_embeddings_neg[b] = tabular_embeddings[neg_idx]

        image_embeddings_neg = torch.zeros_like(image_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeddings_neg[b] = image_embeddings[neg_idx]

        tabular_embeddings_all = torch.cat([tabular_embeddings, tabular_embeddings_neg], dim=0)
        image_embeddings_all = torch.cat([image_embeddings_neg, image_embeddings], dim=0)
        output_neg = self.forward_multimodal_feature(tabular_features=tabular_embeddings_all, image_features=image_embeddings_all)
        z = self.itm_head(torch.cat([output_pos, output_neg], dim=0))
        itm_labels = torch.cat([torch.ones(B), torch.zeros(2*B)], dim=0).long().to(logits.device)
        loss_itm = self.criterion_itm(z, itm_labels)
        return loss_itm, z, itm_labels


    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        '''
        Train
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        '''
        im_views, tab_views, y, _, original_tab = batch

        # =======================================  itc    =======================================================================
        # Augmented image and unagumented tabular
        z0, image_embeddings = self.forward_imaging(im_views[1]) 
        z1, tabular_embeddings = self.forward_tabular(tab_views[0])
        loss_itc, logits, labels = self.criterion_train_itc(z0, z1, y)
        self.log(f"multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(image_embeddings, tabular_embeddings, logits)
        self.log(f"multimodal.train.ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular 
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(tabular_features=tabular_embeddings, image_features=image_embeddings)
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2,original_tab,mask=mask)
        self.log(f"multimodal.train.TRloss", loss_tr, on_epoch=True, on_step=False)
        
        if len(im_views[0])==self.hparams.batch_size:
            self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='multimodal')
            self.calc_and_log_train_cat_embedding_acc(logits=pred_cat, labels=target_cat, mask=mask_cat, modality='multimodal')
            self.calc_and_log_train_itm_acc(logits=logits_itm, labels=labels_itm, modality='multimodal')
        
        loss = (loss_itc + loss_tr + loss_itm)/3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
        return {'loss':loss, 'embeddings': multimodal_embeddings, 'labels': y, }

    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        '''
        Validate
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        '''
        im_views, tab_views, y, original_im, original_tab = batch

        # =======================================  itc    =======================================================================
        # Unaugmented views
        z0, image_embeddings = self.forward_imaging(original_im) 
        z1, tabular_embeddings = self.forward_tabular(original_tab)
        loss_itc, logits, labels = self.criterion_val_itc(z0, z1, y)
        self.log(f"multimodal.val.ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(image_embeddings, tabular_embeddings, logits)
        self.log(f"multimodal.val.ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular 
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(tabular_features=tabular_embeddings, image_features=image_embeddings)
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2,original_tab, mask=mask)
        self.log(f"multimodal.val.TRloss", loss_tr, on_epoch=True, on_step=False)

        if len(im_views[0])==self.hparams.batch_size:
            self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='multimodal')
            self.calc_and_log_val_cat_embedding_acc(logits=pred_cat, labels=target_cat, mask=mask_cat, modality='multimodal')
            self.calc_and_log_val_itm_acc(logits=logits_itm, labels=labels_itm, modality='multimodal')
        
        loss = (loss_itc + loss_tr + loss_itm)/3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log(f"multimodal.val.loss", loss, on_epoch=True, on_step=False)
        return {'sample_augmentation': im_views[1], 'embeddings': multimodal_embeddings, 'labels': y}
    

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
        [
            {'params': self.encoder_imaging.parameters()}, 
            {'params': self.projector_imaging.parameters()},
            {'params': self.encoder_tabular.parameters()},
            {'params': self.projector_tabular.parameters()},
            {'params': self.encoder_multimodal.parameters()},
            {'params': self.predictor_tabular.parameters()},
            {'params': self.itm_head.parameters()},
        ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )
    

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

    @with_local_info
    def prepare_DDP(self,
                    local_rank=None, global_rank=None, parallel=None):
        
        from torch.nn.parallel import DistributedDataParallel as DDP
        import time
        import hashlib

        def get_model_hash(model):
            params = torch.cat([p.view(-1) for p in model.parameters()])
            return hashlib.md5(params.detach().cpu().numpy().tobytes()).hexdigest()
        
        if parallel == True:
            self.to(local_rank)
            if local_rank == 0:
                start_time = time.time()
                while True:
                    if time.time() - start_time >= 10:
                        break
                    
            sys.stderr.write(f"Rank {global_rank}: Model hash: {get_model_hash(self.net_)}\n")
                # raise("shit")
            self.net_ = DDP(self.net_, device_ids=[local_rank], output_device=local_rank)