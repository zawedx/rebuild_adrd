'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/models/pretraining.py
'''
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from sklearn.linear_model import LogisticRegression
from lightly.models.modules import SimCLRProjectionHead
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
def torchvision_ssl_encoder(model_name='resnet50', pretrained=True, output_dim=128, return_all_feature_maps=False):
    """
    替代 pl_bolts 的 torchvision_ssl_encoder 函数。
    
    参数:
        model_name (str): 模型名称，支持 'resnet18', 'resnet50' 等。
        pretrained (bool): 是否加载预训练权重。
        output_dim (int): 替换全连接层后的输出维度。
        return_all_feature_maps (bool): 是否返回所有特征图。
    
    返回:
        model (nn.Module): 修改后的模型。
    """
    # 加载 TorchVision 模型
    import torchvision.models as models
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 替换全连接层
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, output_dim)

    # 如果需要返回所有特征图，修改模型的前向传播
    if return_all_feature_maps:
        model = FeatureMapWrapper(model)

    return model

class FeatureMapWrapper(torch.nn.Module):
    """
    包装 TorchVision 模型，使其返回所有特征图。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_maps = []

        # 注册钩子以捕获特征图
        self._register_hooks()

    def _register_hooks(self):
        """
        为模型的每一层注册前向钩子，以捕获特征图。
        """
        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # 遍历模型的所有子模块
        for layer in self.model.children():
            if isinstance(layer, torch.nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, torch.nn.Conv2d) or isinstance(sub_layer, torch.nn.BatchNorm2d) or isinstance(sub_layer, torch.nn.ReLU):
                        sub_layer.register_forward_hook(hook_fn)
            elif isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.ReLU):
                layer.register_forward_hook(hook_fn)

    def forward(self, x):
        self.feature_maps = []  # 清空特征图列表
        out = self.model(x)
        return out, self.feature_maps

import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/openbayes/home/NEW/baseline/TIP')
from models.Tip_utils.Transformer import TabularTransformerEncoder, MultimodalTransformerEncoder, TabularPredictor
from models.Tip_utils.VisionTransformer_imagenet import create_vit

class Pretraining(pl.LightningModule):
    
  def __init__(self, hparams) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)

  def initialize_imaging_encoder_and_projector(self) -> None:
    """
    Selects appropriate resnet encoder
    """
    if self.hparams.model.startswith('vit'):
      self.encoder_imaging_type = 'vit'
      self.encoder_imaging =  create_vit(self.hparams)
      self.pooled_dim = self.hparams.embedding_dim
    elif self.hparams.model.startswith('resnet'):
      self.encoder_imaging_type = 'resnet'
      self.encoder_imaging = torchvision_ssl_encoder(self.hparams.model, return_all_feature_maps=True)
      self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
    self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

  def initialize_tabular_encoder_and_projector(self) -> None:
    self.field_lengths_tabular = torch.load(self.hparams.field_lengths_tabular)
    self.cat_lengths_tabular = []
    self.con_lengths_tabular = []
    for x in self.field_lengths_tabular:
      if x == 1:
        self.con_lengths_tabular.append(x) 
      else:
        self.cat_lengths_tabular.append(x)
    self.encoder_tabular = TabularTransformerEncoder(self.hparams, self.cat_lengths_tabular, self.con_lengths_tabular)
    self.projector_tabular = SimCLRProjectionHead(self.hparams.tabular_embedding_dim, self.hparams.tabular_embedding_dim, self.hparams.projection_dim)
  
  def initialize_multimodal_encoder_and_predictor(self) -> None:
    self.encoder_multimodal = MultimodalTransformerEncoder(self.hparams)
    self.predictor_tabular = TabularPredictor(self.hparams, self.cat_lengths_tabular, self.con_lengths_tabular, self.encoder_tabular.num_unique_cat)

  def initialize_classifier_and_metrics(self, nclasses_train, nclasses_val):
    """
    Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
    """
    # Classifier
    self.estimator = None

    # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
    self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train)
    self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val)

    self.top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_train)
    self.top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_val)

    n_classes_cat = self.encoder_tabular.num_unique_cat
    self.top1_acc_train_cat = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=n_classes_cat)
    self.top5_acc_train_cat = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=n_classes_cat)
    self.top1_acc_val_cat = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=n_classes_cat)
    self.top5_acc_val_cat = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=n_classes_cat)
    self.auc_train_cat = torchmetrics.AUROC(task='multiclass', num_classes=n_classes_cat)
    self.auc_val_cal = torchmetrics.AUROC(task='multiclass', num_classes=n_classes_cat)

    self.acc_train_itm = torchmetrics.Accuracy(task='binary', num_classes=2)
    self.acc_val_itm = torchmetrics.Accuracy(task='binary', num_classes=2)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

    self.classifier_acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

    self.classifier_auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)


  def load_pretrained_imaging_weights(self) -> None:
    """
    Can load imaging encoder with pretrained weights from previous checkpoint/run
    """
    loaded_chkpt = torch.load(self.hparams.imaging_pretrain_checkpoint)
    state_dict = loaded_chkpt['state_dict']
    state_dict_encoder = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
    _ = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=True)
    print("Loaded imaging weights")
    if self.hparams.pretrained_imaging_strategy == 'frozen':
      for _, param in self.encoder_imaging.named_parameters():
        param.requires_grad = False
      parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
      assert len(parameters)==0


  def forward(self, x: torch.Tensor, tabular: torch.tensor) -> torch.Tensor:
    """
    Generates encoding of multimodal data. Pick Clstoken
    """
    _, image_features = self.forward_imaging(x)
    _, tabular_features = self.forward_tabular(tabular)
    multimodal_features = self.encoder_multimodal(x=tabular_features, image_features=image_features)
    return multimodal_features[:,0,:]

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of imaging data.
    """
    y = self.encoder_imaging(x)[-1]
    if self.encoder_imaging_type == 'resnet':
      z = F.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
    elif self.encoder_imaging_type == 'vit':
      z = y[:,0,:]
    z = self.projector_imaging(z)
    return z, y

  def forward_tabular(self, x: torch.Tensor, mask: torch.Tensor=None, mask_special: torch.Tensor=None) -> torch.Tensor:
    """
    Generates projection and encoding of tabular data.
    """
    # (B,N,D)
    y = self.encoder_tabular(x, mask=mask, mask_special=mask_special)
    # (B,N1,C) and (B,N2,1)
    z = self.projector_tabular(y[:,0,:])
    # projected feature, original feature
    return z, y

  def forward_multimodal(self, tabular_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
    """
    Generates prediction of tabular data.
    """
    y = self.encoder_multimodal(x=tabular_features, image_features=image_features)
    z = self.predictor_tabular(y)
    return z, y[:,0,:]
  
  def forward_multimodal_feature(self, tabular_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
    """
    Generates feature of tabular data.
    """
    y = self.encoder_multimodal(x=tabular_features, image_features=image_features)
    return y[:,0,:]

  def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_train(logits, labels)
    self.top5_acc_train(logits, labels)
    
    self.log(f"{modality}.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
    self.log(f"{modality}.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)

  def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_val(logits, labels)
    self.top5_acc_val(logits, labels)
    
    self.log(f"{modality}.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log(f"{modality}.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)
  
  def calc_and_log_train_cat_embedding_acc(self, logits, labels, mask, modality: str) -> None:
    logits, labels = logits[mask].detach(), labels[mask].detach()
    # print(logits.shape, labels.shape)
    self.top1_acc_train_cat(logits, labels)
    self.top5_acc_train_cat(logits, labels)
    self.auc_train_cat(logits, labels)
    self.log(f"{modality}.train.categorical.top1", self.top1_acc_train_cat, on_epoch=True, on_step=False)
    self.log(f"{modality}.train.categorical.top5", self.top5_acc_train_cat, on_epoch=True, on_step=False)
    self.log(f"{modality}.train.categorical.auc", self.auc_train_cat, on_epoch=True, on_step=False)

  def calc_and_log_val_cat_embedding_acc(self, logits, labels, mask, modality: str) -> None:
    logits, labels = logits[mask].detach(), labels[mask].detach()
    self.top1_acc_val_cat(logits, labels)
    self.top5_acc_val_cat(logits, labels)
    self.auc_val_cal(logits, labels)
    self.log(f"{modality}.val.categorical.top1", self.top1_acc_val_cat, on_epoch=True, on_step=False)
    self.log(f"{modality}.val.categorical.top5", self.top5_acc_val_cat, on_epoch=True, on_step=False)
    self.log(f"{modality}.val.categorical.auc", self.auc_val_cal, on_epoch=True, on_step=False)
  
  def calc_and_log_train_itm_acc(self, logits, labels, modality: str) -> None:
    logits, labels = logits.detach(), labels.detach()
    self.acc_train_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
    self.log(f"{modality}.train.ITMacc", self.acc_train_itm, on_epoch=True, on_step=False)
  
  def calc_and_log_val_itm_acc(self, logits, labels, modality: str) -> None:
    logits, labels = logits.detach(), labels.detach()
    self.acc_val_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
    self.log(f"{modality}.val.ITMacc", self.acc_val_itm, on_epoch=True, on_step=False)

  def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
    """
    Train and log classifier
    """
    if self.current_epoch != 0 and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(train_step_outputs)
    
      self.estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(embeddings, labels)
      preds, probs = self.predict_live_estimator(embeddings)

      self.classifier_acc_train(preds, labels)
      self.classifier_auc_train(probs, labels)

      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step and calc validation classifier performance
    """
    if self.hparams.log_images:
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

    # Validate classifier
    if not self.estimator is None and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(validation_step_outputs)
      
      preds, probs = self.predict_live_estimator(embeddings)
      
      self.classifier_acc_val(preds, labels)
      self.classifier_auc_val(probs, labels)

      self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
      self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)


  def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack outputs from multiple steps
    """
    labels = outputs[0]['labels']
    embeddings = outputs[0]['embeddings']
    for i in range(1, len(outputs)):
      labels = torch.cat((labels, outputs[i]['labels']), dim=0)
      embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

    embeddings = embeddings.detach().cpu()
    labels = labels.cpu()

    return embeddings, labels

  def predict_live_estimator(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict using live estimator
    """
    preds = self.estimator.predict(embeddings)
    probs = self.estimator.predict_proba(embeddings)

    preds = torch.tensor(preds)
    probs = torch.tensor(probs)
    
    # Only need probs for positive class in binary case
    if self.hparams.num_classes == 2:
      probs = probs[:,1]

    return preds, probs


  def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.warmup_epochs, T_mult=1, eta_min=0, last_epoch=-1)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    return scheduler