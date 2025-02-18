
labels = [
    'NC', 'MCI', 'AD'
]


fea2drop = ['subject_id', 'visit']

import pandas as pd
import pickle


# output features
print('[features]')
print()
data = dict
data = pickle.load(open('/openbayes/home/LMTDE/data/embeddings/zeroshot_gpt_embedding_large.pkl', 'rb'))

for k, v in data.items():
    name = k.strip()
    if name in fea2drop: continue
    row = v
    type = row['type'].strip()
    print('\t[feature.{}]'.format(name))
    # if name == 'img_MRI_T1':
    #     print('\ttype = \"undefined\"')
    if type == 'multi':
        print('\ttype = \"multiple\"')
        print('\tnum_categories = {}'.format(len(row['embedding'])))
    elif type == 'numerical':
        print('\ttype = \"numerical\"')
        try:
            print('\tshape = [{}]'.format(1))
        except:
            print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
    elif type == 'binary':
        print('\ttype = \"binary\"')
        print('\tnum_categories = \"2\"')
    elif type == 'M':
        print('\ttype = \"imaging\"')
        try:
            print('\tshape = [{}]'.format(int(row['length'])))
        except:
            print('\tshape = \"################ TO_FILL_MANUALLY ################\"')
    print()

# output labels
print('[labels]')
print()
for name in labels:
    print('\t[label.{}]'.format(name))
    print('\ttype = \"categorical\"')
    print('\tnum_categories = 2')
    print()