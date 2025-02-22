import pandas as pd
import numpy as np

adni_merge = pd.read_csv('processed_adni_data.csv', low_memory=False)
label = pd.read_csv('adni_new_raw.csv', low_memory=False)

print(label.iloc[0])
index = 0

adni_merge['DXAD'] = None
adni_merge['DIAGNOSIS'] = None
for each in adni_merge.iterrows():
    name = each[1]['subject_id']
    count = 0
    while label.iloc[index]['subject_id'] != name:
        print(name)
        print(label.iloc[index]['subject_id'])
        index += 1
        count += 1
        if count > 200:
            print(index)
            exit()
    now = label.iloc[index]
    each[1]['DXAD'] = now['DXAD']
    each[1]['DIAGNOSIS'] = now['DIAGNOSIS']
    index += 1

adni_merge.to_csv('adni_final.csv')