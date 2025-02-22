import pandas as pd
import numpy as np
import pickle
from icecream import ic
import toml

prefix = 'data/datasets/adni/'
ic.enable()
df = pd.read_csv(prefix + 'adni_0shot.csv')
cnf = toml.load(prefix + 'adni.toml')
ic(cnf)