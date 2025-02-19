import sys
import os
sys.path.append(os.path.abspath("/openbayes/home/NEW/rebuild_adrd/"))

from common.ml_frame import with_local_info, ml_frame
from common.ml_logger import MLLogger
import toml

from IPython import embed
from SimpleITK import Rank
import pandas as pd
from sympy import true
import torch
import json
import argparse
import monai
import nibabel as nib
import wandb
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.dataset_csv import CSVDataset
from model.adrd import ADRDModel#, GMModel, SCLAD , MGDA
from model.tip import TIPModel
from tqdm import tqdm
from collections import defaultdict
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.disable()
from torchvision import transforms
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Resized,
)

def parser():
    parser = argparse.ArgumentParser("Transformer pipeline", add_help=False)

    parser.add_argument('--mgda', action='store_true',
        help='Set True for MGDA_UB')
    
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_ml_frame_from_config(config):
    # TODO: temporary solution
    for key, value in config.items():
        # ml_frame.set_local_info(key, value)
        # print("update: ",key, " value: ", value)
        assert(type(value) is dict)
        for k, v in value.items():
            if isinstance(v, str) and (ml_frame.get_local_info('prefix') is not None):
                v = v.replace("${prefix}", ml_frame.get_local_info('prefix'))
            ml_frame.set_local_info(k, v)
            print("update: ",k, " value: ", v)

@with_local_info
def init_DDP(
    parallel=None
):
    if parallel:
        dist.init_process_group(backend='nccl')
        
        local_rank= int(os.environ["LOCAL_RANK"])
        ml_frame.set_local_info('local_rank', local_rank)
        ml_frame.set_local_info('global_rank', dist.get_rank())

        torch.cuda.set_device(local_rank)
        set_random_seed(42)

        if local_rank != 0:
            sys.stdout = open(os.devnull, 'w')
    else:
        local_rank = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ml_frame.set_local_info('local_rank', local_rank)
        ml_frame.set_local_info('global_rank', 0)


config = toml.load('/openbayes/home/NEW/rebuild_adrd/dev/config.toml')
args = parser()

# use args to update config
# TODO
# config.update(args)
update_ml_frame_from_config(config)

init_DDP()

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    @with_local_info
    def __init__(self, dat_type,
                 img_size=None):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.8, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                # Resized(keys=["image"], spatial_size=img_size),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        try:
            image_data = data["image"]
            check = nib.load(image_data).get_fdata()
            if len(check.shape) > 3:
                return None
            return self.transforms(data)
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
trn_filter_transform = FilterImages(dat_type='trn')
vld_filter_transform = FilterImages(dat_type='vld')

# initialize datasets
seed = 0
stripped = '_stripped_MNI'
print("Loading training dataset ... ")
dat_trn = CSVDataset(
    dat_file=ml_frame.get_local_info('train_path'), 
    cnf_file=ml_frame.get_local_info('cnf_file'), 
    mode=0, 
    img_mode=ml_frame.get_local_info('img_mode'), 
    arch=ml_frame.get_local_info('img_net'), 
    transforms=FilterImages('trn'), 
    stripped=stripped
)
print("Done.\nLoading Validation dataset ...")
dat_vld = CSVDataset(
    dat_file=ml_frame.get_local_info('vld_path'), 
    cnf_file=ml_frame.get_local_info('cnf_file'), 
    mode=1, 
    img_mode=ml_frame.get_local_info('img_mode'), 
    arch=ml_frame.get_local_info('img_net'), 
    transforms=FilterImages('vld'), 
    stripped=stripped
)
print("Done.\nLoading testing dataset ...")
dat_tst = CSVDataset(
    dat_file=ml_frame.get_local_info('test_path'), 
    cnf_file=ml_frame.get_local_info('cnf_file'), 
    mode=2, 
    img_mode=ml_frame.get_local_info('img_mode'), 
    arch=ml_frame.get_local_info('img_net'), 
    transforms=FilterImages('tst'), 
    stripped=stripped
)
# print("Done.")

label_fractions = dat_trn.label_fractions
print(label_fractions)

# df = pd.read_csv(args.data_path)
# label_distribution = {}
# for label in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
#     label_distribution[label] = dict(df[label].value_counts())
# print(label_distribution)

ml_frame.set_local_info('src_modalities', dat_trn.feature_modalities)
ml_frame.set_local_info('tgt_modalities', dat_trn.label_modalities)
ml_frame.set_local_info('label_fractions', label_fractions)
ml_frame.set_local_info('num_encoder_layers', 2)
ml_frame.set_local_info('criterion', 'AUC (ROC)')
ml_frame.set_local_info('_dataloader_num_workers', 2)
ml_frame.set_local_info('device', 'cuda')

current_model_name = 'TIP'
if current_model_name == 'TIP':
    mdl = TIPModel()
elif current_model_name == 'ADRD':
    mdl = ADRDModel()


    
# if ml_frame.get_local_info('parallel') == True:
#     local_rank = ml_frame.get_local_info('local_rank')
#     global_rank = ml_frame.get_local_info('global_rank')
#     mdl.to(local_rank)
#     if local_rank == 0:
#         start_time = time.time()
#         while True:
#             if time.time() - start_time >= 10:
#                 break
            
#     sys.stderr.write(f"Rank {global_rank}: Model hash: {get_model_hash(mdl.net_)}\n")
#         # raise("shit")
#     mdl.net_ = DDP(mdl.net_, device_ids=[local_rank], output_device=local_rank)

mdl.prepare_DDP()

# if args.img_mode == 0 or args.img_mode == 2:
#     mdl.fit(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels, img_train_trans=trn_filter_transform, img_vld_trans=vld_filter_transform, img_mode=args.img_mode)
# else:
mdl.fit(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels)

torch.distributed.destroy_process_group()