import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image


# for dataset in ('APL', 'AWA', 'BCT', 'DOG', 'FLW', 'FNG', 'MED_LF'):
for dataset in ['ACT_410', 'PRT', 'PLT_DOC', 'FNG', 'PNU', 'RSICB', 'TEX_DTD']:
    path = os.path.join('Data', dataset)
    data = pd.read_csv(os.path.join(path, 'labels.csv'))

    images = []
    for idx, row in data.iterrows():
        if row.FILE_NAME == '.DS_Store':
            continue
        img_path = os.path.join(path, 'images', row.FILE_NAME)
        image = (np.array(Image.open(img_path)) / 255.)
        images.append(image)
    
    images = np.array(images).reshape(-1, 3)
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    with open(os.path.join(path, 'stats.pkl'), 'wb') as file:
        pickle.dump({'mean': mean, 'std': std}, file)
    print(mean, std)