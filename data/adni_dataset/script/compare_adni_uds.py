import toml
import pandas as pd
import numpy as np
import pickle
import json

config = toml.load('/openbayes/home/LMTDE/dev/data/toml_files/default_conf_new.toml')
# print(config)

# for k, v in config['feature'].items():
    # print(k)

json_data = json.load(open('/openbayes/home/LMTDE/data/datasets/example_conversion_scripts/adni_to_uds.json'))
rename_dict = {key: value for category in json_data.values() for key, value in category.items()}
# print(rename_dict)
mapping = rename_dict

# nacc = set()
# for k,v in convert_map.items():
#     if k in mapping.values():
#         nacc.add(k)
# print(nacc)
# exit()

dxad = set()
df = pd.read_csv('adni_final.csv', low_memory=False)
df = df[df['DIAGNOSIS'].notna() & (df['DIAGNOSIS'].astype(str).str.strip() != "")]
df['AD'] = (df['DXAD'] == 1.0).astype(int)
df['NC'] = (df['DIAGNOSIS'] == 1).astype(int)
df['MCI'] = (df['DIAGNOSIS'] == 2).astype(int)
print(df['NC'].sum())
print(df['MCI'].sum())
df.drop('DXAD', axis=1, inplace=True)
df.drop('DIAGNOSIS', axis=1, inplace=True)


# print(adni)

# df.rename(columns=mapping, inplace=True)
print(df.columns)

df.to_csv('adni_0shot.csv', index=False)