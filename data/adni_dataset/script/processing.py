from sklearn.exceptions import DataDimensionalityWarning
import pandas as pd
import pickle
def load_excel(file_path='adni_descriptor.xlsx'):
    data = pd.read_excel(file_path)
    data_dict = {}
    for index, row in data.iterrows():
        key = row['Variable name']
        data_dict[key] = {}
        for other_key in row.keys():
            if other_key == 'Variable name': continue
            data_dict[key].update({other_key: row[other_key]})
    return data_dict

data = load_excel()
for k, v in data.items():
    try:
        v['Descriptor'] = v['Descriptor'].split('\n')[0]
    except:
        continue
    print(f"Key:{k}, value: {v}")