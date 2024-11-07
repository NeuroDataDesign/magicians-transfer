import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Nov12_LifelongTreeple import NaiveTransferRandomForest, LifeLongForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#Load Data
df = pd.read_excel('Human.parcellated_thickness.xlsx')
df.head()

df_sex = pd.read_excel('subjects_age_sex_data_MRI.xlsx')
df_sex.head()

X1_human = []
X2_human = []
X_human = []
y_human = []
IDs = set(df['sid'])
ref_IDs = set(df_sex['ID'])

for subject in tqdm(IDs):
    if subject in ref_IDs:
        features = np.array(df[df['sid'] == subject]).reshape(-1)[2:]
        gender = list(df_sex[df_sex['ID'] == subject]['Sex'])
        sex = int(gender[0] == 'FEMALE')

        X1_human.append(list(features[:182]))
        X2_human.append(list(features[182:]))
        X_human.extend(features)
        y_human.append(sex)

X1_human = np.array(X1_human)
X2_human = np.array(X2_human)
y_human = np.array(y_human)

df = pd.read_excel('Macaque.parcellated_thickness.xlsx')
df.head()
df_sex = pd.read_csv('uwmadison.csv')
df_sex.head()
X1_macaque = []
X2_macaque = []
X_macaque = []
y_macaque = []
IDs = set(df['participant_id'])
ref_IDs = set(df_sex['participant_id'])

for subject in tqdm(IDs):
    if subject in ref_IDs:
        features = np.array(df[df['participant_id'] == subject]).reshape(-1)[4:]
        gender = list(df_sex[df_sex['participant_id'] == subject]['sex'])
        sex = int(gender[0] == 'F')

        X1_macaque.append(list(features[:182]))
        X2_macaque.append(list(features[182:]))
        X_macaque.extend(features)
        y_macaque.append(sex)

X1_macaque = np.array(X1_macaque)
X2_macaque = np.array(X2_macaque)
y_macaque = np.array(y_macaque)
print('X1_macaque_shape:', X1_macaque.shape)
print('X2_macaque_shape:', X1_macaque.shape)
valid_indices = ~np.isnan(X1_human).any(axis=1) & ~np.isnan(X2_human).any(axis=1)
X1_human = X1_human[valid_indices]
X2_human = X2_human[valid_indices]
y_human= np.array(y_human)[valid_indices]
print('X1_human_shape:', X1_human.shape)
print('X2_human_shape:', X1_human.shape)
valid_indices = ~np.isnan(X1_macaque).any(axis=1) & ~np.isnan(X2_macaque).any(axis=1)
X1_macaque = X1_macaque[valid_indices]
X2_macaque = X2_macaque[valid_indices]
y_macaque= np.array(y_macaque)[valid_indices]
print('X1_macaque_shape:', X1_macaque.shape)
print('X2_macaque_shape:', X1_macaque.shape)

reps = 5
accuracy = 0.0
accuracy_iter = 0.0

for ii in tqdm(range(reps)):
    x_train, x_test, y_train, y_test = train_test_split(
                    X1_macaque, y_macaque, train_size=0.8, random_state=ii, stratify=y_macaque)
    forest = LifeLongForest()
    forest.add_task(n_estimators=1000)
    forest.add_task(n_estimators=1000)
    forest.fit(X1_human, y_human, task_idx=0)
    forest.fit(x_train, y_train, task_idx=1)

    accuracy_iter = forest.decode(x_test, y_test)
    accuracy += accuracy_iter
    print(f"Iteration {ii+1} Accuracy: {accuracy_iter}")

print('Accuracy is ',accuracy/reps)

reps = 5
accuracy = 0.0
accuracy_iter = 0.0

for ii in tqdm(range(reps)):
    x_train, x_test, y_train, y_test = train_test_split(
                    X1_macaque, y_macaque, train_size=0.8, random_state=ii, stratify=y_macaque)
    clf = NaiveTransferRandomForest(n_estimators=1000, n_jobs=-1)
    clf.pre_train(X1_human,y_human)
    clf.transfer_train(x_train, y_train)

    accuracy_iter = clf.predict(x_test, y_test)
    accuracy += accuracy_iter
    print(f"Iteration {ii+1} Accuracy: {accuracy_iter}")
print('Accuracy is ',accuracy/reps)