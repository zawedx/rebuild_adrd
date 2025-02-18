from common.ml_frame import with_local_info

from sys import modules
from .new_llm_finetune_emb import V1EmbeddingLayer
from .new_llm_finetune_trf import TransformerLayerPT, TransformerLayerV2
import torch
import numpy as np
from typing import Any, Type
Tensor = Type[torch.Tensor]
from icecream import ic
ic.disable()

class Transformer(torch.nn.Module):
    ''' ... '''
    @with_local_info
    def __init__(self,
        d_model=None, tgt_modalities=None
    ) -> None:
        ''' ... '''
        super().__init__()

        self.embedding_layer = V1EmbeddingLayer()
        
        self.transformer_layer = TransformerLayerPT()        

        # classifiers (binary only)
        self.modules_cls = torch.nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                self.modules_cls[k] = torch.nn.Linear(d_model, 1)
            else:
                # unrecognized
                raise ValueError

    @with_local_info
    def forward(self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        # x_img: dict[str, Tensor] | Any = None,
        detach: bool = False,
        fusion_stage=None, 
    ) -> dict[str, Tensor]:
        """ ... """
        if self.training:
            self.embedding_layer.clear_cache()
        out_emb = self.embedding_layer(x, mask)
        if fusion_stage == "late":
            out_emb = {k: v for k,v in out_emb.items() if "img_MRI" not in k}
            img_out_emb = {k: v for k,v in out_emb.items() if "img_MRI" in k}
            mask_nonimg = {k: v for k,v in mask.items() if "img_MRI" not in k}
            out_trf = self.transformer_layer(out_emb, mask_nonimg)
            out_trf = torch.concatenate()
        else:
            out_trf = self.transformer_layer(out_emb, mask)
        trf_detach = out_trf.detach()
        # out_cls = self.token0_cls(out_trf[0])
        if detach:
            out_cls = self.one2one_cls(trf_detach)
        else:
            # assert(not torch.isnan(out_trf).any())
            out_cls = self.one2one_cls(out_trf)
        
        return out_trf, out_cls

    def token0_cls(self,
        out_trf: Tensor,
    ) -> dict[str, Tensor]:
        """ ... """
        tgt_iter = self.modules_cls.keys()
        out_cls = {k: self.modules_cls[k](out_trf).squeeze(1) for k in tgt_iter}
        return out_cls
    
    def one2one_cls(self,
        out_trf # Tensor[13, batchsize, d_model],
    ) -> dict[str, Tensor]:
        """ ... """
        out_cls = {}
        for i, (k, mod) in enumerate(self.modules_cls.items()):
            out_cls[k] = mod(out_trf[i]).squeeze(1)
        
        return out_cls

if __name__ == '__main__':
    ''' for testing purpose only '''
    pass
