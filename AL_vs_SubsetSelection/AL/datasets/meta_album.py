import os
import pickle
from torchvision import datasets, transforms
import pickle
import numpy as np

def DOG(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 120

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'DOG_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'DOG_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test


def AWA(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 50

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'AWA_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'AWA_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced

    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def APL(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 21

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'APL_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'APL_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced

    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def ACT_410(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 29

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'ACT_410_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'ACT_410_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced

    print(dst_train.scores)
    print(dst_train.scores_balanced)
    # dst_train.scores_balanced = []
    # dst_train.scores = np.array([])


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def TEX_DTD(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 47

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'TEX_DTD_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'TEX_DTD_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced
    print(scores_balanced)
    # dst_train.scores_balanced = []
    # dst_train.scores = np.array([])


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def PRT(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 21

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'PRT_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'PRT_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced
    # dst_train.scores_balanced = []
    # dst_train.scores = np.array([])


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def PNU(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 19

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'PNU_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'PNU_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced
    # dst_train.scores_balanced = []
    # dst_train.scores = np.array([])


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test

def FLW(args, size=32):
    path = args.data_path
    channel = 3
    # im_size = (128, 128)
    if size == 32:
        im_size = (32, 32)
    else:
        im_size = (224, 224)
    num_classes = 102

    with open(os.path.join(path, 'stats.pkl'), 'rb') as file:
        stats = pickle.load(file)
    mean = stats['mean']
    std = stats['std']
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if args.strategy == 'basic':
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', 'train'), transform=transform)
    else:
        dst_train = datasets.ImageFolder(root=os.path.join(path, 'images', f'{args.strategy}_subset', 'train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(path, 'images', 'test'), transform=transform)

    with open(os.path.join(path, 'FLW_scores_for_selection.pkl'), 'rb') as file:
        scores = pickle.load(file)
    dst_train.scores = np.array(scores)

    with open(os.path.join(path, 'FLW_scores_for_selection_balanced.pkl'), 'rb') as file:
        scores_balanced = pickle.load(file)
    dst_train.scores_balanced = scores_balanced
    # dst_train.scores_balanced = []
    # dst_train.scores = np.array([])


    class_names = dst_train.classes
    print(class_names)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train, dst_test