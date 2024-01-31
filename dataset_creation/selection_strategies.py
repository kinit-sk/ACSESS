
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
from torchvision import transforms
import pickle
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import shutil

def random_select(dataset, shots=10, seed=1337):
    classes = dataset['CATEGORY']
    classes = np.unique(np.array(classes))
    
    torch.manual_seed(seed)
    selected_samples = []
    for cls in classes:
        cls_data = dataset[dataset['CATEGORY'] == cls]
        indices = torch.randperm(cls_data.shape[0])[:shots]
        cls_data = cls_data.iloc[indices]
        selected_samples.append(cls_data)
    
    new_data = pd.concat(selected_samples)
    return new_data


def similarity_select(dataset, shots=10, seed=1337):
    classes = dataset['CATEGORY']
    classes = np.unique(np.array(classes))
    
    torch.manual_seed(seed)
    selected_samples = []
    for cls in classes:
        cls_data = dataset[dataset['CATEGORY'] == cls]
        features = cls_data.features.tolist()
        features = np.array(features)
        indices = torch.randperm(cls_data.shape[0])[:1].tolist()

        for _ in range(shots - 1):
            sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()[::-1]
            for index in sim:
                if index not in indices:
                    indices.append(index)
                    break
        cls_data = cls_data.iloc[indices]
        selected_samples.append(cls_data)
    
    new_data = pd.concat(selected_samples)
    return new_data.drop('features', axis=1)

def diversity_select(dataset, shots=10, seed=1337):
    classes = dataset['CATEGORY']
    classes = np.unique(np.array(classes))
    
    torch.manual_seed(seed)
    selected_samples = []
    for cls in classes:
        cls_data = dataset[dataset['CATEGORY'] == cls]
        features = cls_data.features.tolist()
        features = np.array(features)
        indices = torch.randperm(cls_data.shape[0])[:1].tolist()

        for _ in range(shots - 1):
            sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()
            for index in sim:
                if index not in indices:
                    indices.append(index)
                    break
        cls_data = cls_data.iloc[indices]
        selected_samples.append(cls_data)
    
    new_data = pd.concat(selected_samples)
    return new_data.drop('features', axis=1)

def generate_data_representation(dataset, path, save_path):
    images = []
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")

    for idx, row in dataset.iterrows():
        img_path = os.path.join(path, 'images', row.FILE_NAME)
        image = transform(Image.open(img_path))
        images.append(image)
    

    print('Prepared images')

    inputs = feature_extractor(images, return_tensors="pt")['pixel_values']
    print('Prepared inputs')
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda()
    layer = model._modules.get('avgpool')
    model.eval()
    batch_size = 256
    embeddings = None
    
    # for input in inputs:
        # input = input.unsqueeze(0).cuda()
        # my_embedding = torch.zeros(512)
        # def copy_data(m, i, o):
            # my_embedding = o.data.detach().cpu().numpy().reshape(-1, 512)
            # print(my_embedding)
            # print(my_embedding.shape)

            # my_embedding.copy_(o.data)
        # h = layer.register_forward_hook(copy_data)
        # model(input)
        # h.remove()
        # embeddings.append(my_embedding)
        # break

    for i in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[i:i + batch_size].cuda() 
        my_embedding = torch.zeros(batch_inputs.shape[0], 512)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.detach().cpu().reshape(-1, 512))
        h = layer.register_forward_hook(copy_data)
        model(batch_inputs)
        h.remove()
        if embeddings is None:
            embeddings = my_embedding
        else:
            embeddings = torch.cat((embeddings, my_embedding))
    # for idx, row in dataset.iterrows():
        # img_path = os.path.join(path, 'images', row.FILE_NAME)
        # image = transform(Image.open(img_path)).unsqueeze(0)
        # print(image.shape)
        
        # images.append(image)

    # Load the pretrained model
    # Use the model object to select the desired layer
    print('Finished preprocess')
    embeddings = np.array(embeddings)
    with open(os.path.join(save_path, 'features.pkl'), 'wb') as file:
        pickle.dump(embeddings, file)
    dataset['features'] = embeddings
    return embeddings


MAPPING = {
    'random': random_select,
    'similarity': similarity_select,
    'diversity': diversity_select,
}

SUB_STRATEGY = 'cartography'
# SUB_STRATEGY = None
# for dataset in ('APL', 'AWA', 'BCT', 'DOG', 'FLW', 'FNG', 'MED_LF'):
for dataset in ['ACT_410', 'PRT', 'PNU', 'TEX_DTD', 'FLW']:
# for dataset in ['DOG', 'AWA', 'APL']:
    path = os.path.join('Data', dataset)
    if SUB_STRATEGY is not None:
        new_path = os.path.join(path, SUB_STRATEGY)
    else:
        new_path = path
    train = pd.read_csv(os.path.join(new_path, 'train.csv'))
    # train = pd.read_csv(os.path.join(path, 'train.csv'))
    if not os.path.exists(os.path.join(new_path, 'features.pkl')):
    # if not os.path.exists(os.path.join(path, 'features.pkl')):
        train['features'] = generate_data_representation(train, path, new_path).tolist()
        # train['features'] = generate_data_representation(train, path, path).tolist()
    else:
        with open(os.path.join(new_path, 'features.pkl'), 'rb') as file:
        # with open(os.path.join(path, 'features.pkl'), 'rb') as file:
            features = pickle.load(file)
        train['features'] = features.tolist()
    # for strategy in ['random']:
    path = new_path
    for strategy in ['random', 'diversity', 'similarity']:
        strategy_path = os.path.join(path, strategy)
        if not os.path.exists(strategy_path):
            os.makedirs(strategy_path)
        random.seed(1337)
        seeds = [random.randint(0, 100000) for _ in range(10)]
        for idx, seed in enumerate(seeds):
            selection = MAPPING[strategy](train, seed=seed)
            selection.to_csv(os.path.join(strategy_path, f'{strategy}_{idx}.csv'))
            # random_selection = random_select(train, seed=seed)
            # random_selection.to_csv(os.path.join(strategy_path, f'random_{idx}.csv'))

for TO_SELECT in [1]:
# SUB_STRATEGY = 'cartography'
    SUB_STRATEGY = None
    # for dataset in ('APL', 'AWA', 'BCT', 'DOG', 'FLW', 'FNG', 'MED_LF'):
    # for dataset in ['DOG', 'AWA', 'APL']:
    for dataset in ['ACT_410', 'TEX_DTD', 'PRT', 'PNU', 'FLW']:
        path = os.path.join('Data', dataset)
        if SUB_STRATEGY is not None:
            new_path = os.path.join(path, SUB_STRATEGY)
            train = pd.read_csv(os.path.join(new_path, 'train.csv'))
            path = new_path
        else:
            train = pd.read_csv(os.path.join(path, 'train.csv'))
        # for strategy in ['random']:
        for strategy in ['random']:
            strategy_path = os.path.join(path, f'{strategy}_size_change')
            if not os.path.exists(strategy_path):
                os.makedirs(strategy_path)
            random.seed(1337)
            seeds = [random.randint(0, 100000) for _ in range(10)]
            for idx, to_select in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]):
                selection = random_select(train, to_select, 1337)
                selection.to_csv(os.path.join(strategy_path, f'{to_select}.csv'))
                print(selection.shape)
                # random_selection = random_select(train, seed=seed)
                # random_selection.to_csv(os.path.join(strategy_path, f'random_{idx}.csv'))