import pandas as pd
import pickle
import numpy as np

# Load the mapping dictionary from .pkl
pkl_file_path = 'data/datasets/example_conversion_scripts/uds_to_model_input.pkl'
with open(pkl_file_path, 'rb') as f:
    mapping_dict = pickle.load(f)

# Read the CSV file
csv_file_path = 'renamed_adni.csv'
df = pd.read_csv(csv_file_path)
df.drop(columns='FTD', inplace=True)
# Apply mapping rules, modify column names and change values
to_keep = []
if True:
    mapping_dict['NACCALZD'] = ('AD', {8: 0})  # AD
    mapping_dict['NACCLBDE'] = ('LBD', {8: 0})  # LBD
    mapping_dict['NORMCOG'] = ('NC', None)  # NC
    mapping_dict['DEMENTED'] = ('DE', None)  # DE
    mapping_dict['NACCTMCI'] = ('MCI', {8: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # MCI
    mapping_dict['HYCEPH'] = ('NPH', {-4: None})  # Normal Pressure Hydrocephalus (NPH)
    # Traumatic Brain Injury
    mapping_dict['BRNINJ'] = ('TBI', {-4: None, 9: None})  # Brain Injury

    # labels which are already done
    
    #------------------LBD
    mapping_dict['NACCLBDS'] = ('NACCLBDS', {8: 0})  # LBD
    mapping_dict['PARK'] = ('PARK', {9: None, -4: None})  # PD should be merged in LBD
    mapping_dict['PDOTHR'] = ('PDOTHR', {9: None, -4: None})  # Other PD merged in LBD
    #-------------------VD
    mapping_dict['CVD'] = ('VD', {-4: None})  # Vascular brain injury (VBI)
    mapping_dict['CVDCOG'] = ('cvd_CVDCOG', {-4: None, 8: None}) # Cerebrovascular disease contributing to cognitive impairment
    mapping_dict['PD'] = ('his_PD', {-4: None, 9: None})
    mapping_dict['STROKE'] = ('STROKE', {-4: None, 8: None})  # Stroke merge to VD 
    #-------------------PRD
    mapping_dict['PRION'] = ('PRD', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Prion Disease
    #-------------------FTD
    mapping_dict['FTLDMO'] = ('FTD', {2: 1, 3: 0, 7: 0, 8: 0, -4: None, '-4':None})  # Frontotemporal Dementia
    mapping_dict['PPAPH'] = ('PPAPH', {2: 1, 3: 0, 7: 0, 8: 0, -4: None, '-4':None})  # Primary Progressive Aphasia
    mapping_dict['CORT'] = ('CBD', {2: 1, 3: 0, 7: 0, 8: 0, -4: None, '-4':None})  # Corticobasal Degeneration
    mapping_dict['PSP'] = ('PSP', {2: 1, 3: 0, 7: 0, 8: 0, -4: None, '-4':None})  # Progressive Supranuclear Palsy
    # SEF Subcategories-----------SEF
    mapping_dict['IMPSUB'] = ('IMPSUB', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Substance abuse
    mapping_dict['ALCDEM'] = ('ALCDEM', {8: 0})  # Alcohol
    mapping_dict['DYSILL'] = ('DYSILL', {8: None, -4: None})  # Systemic disease/medical illness
    mapping_dict['DELIR'] = ('DELIR', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Delirium
    mapping_dict['HIV'] = ('HIV', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # HIV
    mapping_dict['OTHCOG'] = ('OTHCOG', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Other infectious diseases

    # Psychiatric Conditions Subcategories---------PSY
    mapping_dict['SCHIZOP'] = ('SCHIZOP', {-4: None, 9: None})  # Schizophrenia
    mapping_dict['DEP'] = ('DEP', {-4: None, 9: None})  # Depression
    mapping_dict['BIPOLDX'] = ('BIPOLDX', {-4: None, 9: None})  # Bipolar Disorder
    mapping_dict['ANXIET'] = ('ANXIET', {-4: None, 9: None})  # Anxiety
    mapping_dict['PTSDDX'] = ('PTSDDX', {-4: None, 9: None})  # Post-traumatic Stress Disorder

    # Other Diagnoses
    mapping_dict['NEOP'] = ('NEOP', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Neoplasms
    mapping_dict['MSA'] = ('MSA', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Multiple System Atrophy
    mapping_dict['HUNT'] = ('HUNT', {2: 1, 3: 0, 7: 0, 8: 0, -4: None})  # Huntington's Disease
    mapping_dict['SEIZURES'] = ('his_SEIZURES', {2: 1, -4: None, 9: None})  # Seizures

    mapping_dict['PARK'] = ('PARK', {2: 1, -4: None, 9: None})  # Park

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")

for key, (new_name, mapping) in mapping_dict.items():
    if key in df.columns:
        print(f"Processing column '{key}' - New name: '{new_name}'")
        to_keep.append(new_name)
        # Map values and rename columns
        if mapping is not None:
            # Create a new mapping dict, replacing None with pd.NA
            new_mapping = {k: v if not pd.isna(v) else None for k, v in mapping.items()}
            # Apply mapping to the original column
            df[key] = df[key].astype('object')
            df[key] = df[key].apply(lambda x: new_mapping[x] if x in new_mapping else x)
            # df[key] = df[key].apply(lambda x: int(x) if isinstance(x, (int, float, bool, np.float64)) and not pd.isna(x) else None)
            # df[key] = df[key].apply(lambda x: None if pd.isna(x) else x)
            # df[key] = df[key].astype('Int64')
            print(f"Applied mapping for column '{key}': {new_mapping}")
        # Rename the column to the new name
        df.rename(columns={key: new_name}, inplace=True)
    else:
        print(f"Warning: Column '{key}' not found in CSV. Skipping.")

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")

required_columns = ['NACCID', 'diagnosis', 'NACCMRSA', 'VISITMO', 'VISITDAY', 'VISITYR','VD','ODE','PSY','SEF','LBD', 'PRD'] # Columns that must be present in the final DataFrame
to_keep += required_columns

# df.fillna(None, inplace=True)
df['LBD'] = df[['LBD', 'NACCLBDS', 'PARK', 'PDOTHR']].max(axis=1)
df['VD'] = df[['VD', 'STROKE']].max(axis=1)
df['FTD'] = df[['FTD', 'PPAPH', 'CBD', 'PSP']].max(axis=1)
df['SEF'] = df[['IMPSUB', 'ALCDEM', 'DYSILL', 'DELIR', 'HIV', 'OTHCOG']].max(axis=1)
df['PSY'] = df[['SCHIZOP', 'DEP', 'BIPOLDX', 'ANXIET', 'PTSDDX']].max(axis=1)
df['ODE'] = df[['NEOP', 'MSA', 'HUNT']].max(axis=1)

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")

# Keep only the mapped columns
df['diagnosis'] = ((df['DE'] == 1) | (df['MCI'] == 1)).astype(int)
to_keep = list(set(to_keep))
df = df[to_keep]

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")

# df['NACCLBDS'] = df['NACCLBDS'].fillna(0).astype(int)
# df['PARK'] = df['PARK'].fillna(0).astype(int)
# df['PDOTHR'] = df['PDOTHR'].fillna(0).astype(int)
# Ensure required columns are present

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns in DataFrame after mapping: {missing_columns}")
    exit(1)

exclude_columns = ['diagnosis', 'NACCMRSA', 'VISITMO', 'VISITDAY', 'VISITYR']
feature_columns = [col for col in df.columns if col not in exclude_columns]
df['features_available'] = df[feature_columns].notna().sum(axis=1)

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")
# Convert NACCMRSA to indicate MRI availability (1 if MRI is available, else 0)
df['NACCMRSA'] = df['NACCMRSA'].notna().astype(int)

# Create 'VISITDATE' from VISITYR, VISITMO, VISITDAY
df['VISITDATE'] = pd.to_datetime(
    dict(year=df['VISITYR'], month=df['VISITMO'], day=df['VISITDAY']),
    errors='coerce'
)

# Handle missing dates by using earliest possible date
df['VISITDATE'] = df['VISITDATE'].fillna(pd.Timestamp('1900-01-01'))

print("Selecting best visit per patient...")

# Sort the DataFrame according to the priorities
df_sorted = df.sort_values(
    by=['NACCID', 'diagnosis', 'NACCMRSA', 'features_available', 'VISITDATE'],
    ascending=[True, False, False, False, False]
)

# Select the best visit per patient
selected_visits = df_sorted.drop_duplicates(subset='NACCID', keep='first').reset_index(drop=True)

# Keep only the desired columns
selected_visits = selected_visits[to_keep]

# Drop auxiliary columns if not needed
selected_visits = selected_visits.drop(columns=exclude_columns + ['VISITDATE'], errors='ignore')

duplicate_columns = df.columns[df.columns.duplicated()].tolist()
if duplicate_columns:
    raise(f"Duplicate columns found: {duplicate_columns}")
# Save the processed CSV file
output_csv_path = 'adni_nacc.csv'
selected_visits.to_csv(output_csv_path, index=False)

print("Modified DataFrame preview:")
print(selected_visits.head())