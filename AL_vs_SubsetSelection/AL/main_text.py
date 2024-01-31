# Python
import os
import random
import pickle

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T

# Utils
from tqdm import tqdm
from utils import *
from methods.utils import CustomSubset, CustomTextSubset, TextDataset

# Custom
from arguments import parser
from ptflops import get_model_complexity_info
import nets
import datasets as datasets
import methods as methods
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

# Seed
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:'+str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    print("args: ", args)

    with open(os.path.join(args.data_path, '..', 'weights.pkl'), 'rb') as file:
        weights = pickle.load(file)

    dataset_to_len_mapping = {
            'NewsCategory': 20,
            '20News': 100,
            'News20': 100,
            'SNLI': 40,
            'MNLI': 70,
            'MRPC': 50,
            'atis': 20,
            'facebook': 20,
            'snips': 20,
            'liu': 20,

        }
    MODEL_NAME = args.model
    MAX_LEN = dataset_to_len_mapping[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_dict=False)

    for repeat in range(3):
        num_classes, class_names, dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args, tokenizer, MAX_LEN)
        args.num_classes, args.class_names = num_classes, class_names

        # Initialize Unlabeled Set & Labeled Set
        if args.balance:
            to_select = int(args.n_query / args.num_classes)
            scores_balanced = dst_train.scores_balanced
            indices = []
            for c in range(args.num_classes):
                class_index = np.argwhere(np.array(dst_train.targets) == c).reshape(-1)
                if 'ACSESS' in args.uncertainty:
                    if 'random' in args.uncertainty:
                        class_scores = weights[args.dataset]['full']['other'] * np.array(scores_balanced[c]) + weights[args.dataset]['full']['random'] * np.random.uniform(0, 1, size=class_index.shape)
                    else:
                        class_scores = np.array(scores_balanced[c])
                    sorted_indices = np.argsort(-class_scores).reshape(-1)[:to_select]
                    indices.extend(class_index[sorted_indices])
                else:
                    random.shuffle(class_index)
                    indices.extend(class_index[:to_select])
            labeled_set = indices
            unlabeled_set = [idx for idx in range(len(dst_train)) if idx not in indices]
        else:
            indices = list(range(len(dst_train)))
            random.shuffle(indices)

            labeled_set = indices[:args.n_query]
            unlabeled_set = indices[args.n_query:]

        # dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
        dst_subset = CustomTextSubset(dst_train, labeled_set)
        print("Initial set size: ", len(dst_subset))

        save_path = os.path.join('results', args.dataset, args.method)
        if args.strategy != 'basic':
            save_path = os.path.join(save_path, args.strategy)
        if 'Uncertainty' in args.method:
            save_path = os.path.join(save_path, f'{args.uncertainty}_{args.resolution}_{args.epochs}{"_pretrained" if args.pretrained else ""}')
        else:
            save_path = os.path.join(save_path, f'{args.resolution}_{args.epochs}{"_pretrained" if args.pretrained else ""}')

        save_path = os.path.join(save_path, str(repeat))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, 'indices_before_cycle.pkl'), 'wb') as file:
            pickle.dump(labeled_set, file)


        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        # Get Model
        print("| Training on model %s" % args.model)
        network = get_text_model(args, MODEL_NAME)

        # Active learning cycles
        logs = []
        for cycle in range(args.cycle):
            print("====================Cycle: {}====================".format(cycle+1))

            # Get optim configurations for Distrubted SGD
            criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)

            # Training
            print("==========Start Training==========")
            for epoch in range(args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

            acc = test(test_loader, network, criterion, epoch, args, rec)
            print('Cycle {}/{} || Label set size {}: Test acc {}'.format(cycle + 1, args.cycle, len(labeled_set), acc))
            
            # save logs
            logs.append([acc])
            if cycle == args.cycle-1:
                break

            # AL Query Sampling
            print("==========Start Querying==========")

            selection_args = dict(selection_method=args.uncertainty,
                                balance=args.balance,
                                greedy=args.submodular_greedy,
                                function=args.submodular,
                                # weights=weights,
                                weights=[],
                                )
            ALmethod = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
            Q_indices, Q_scores = ALmethod.select()

            # Update the labeled dataset and the unlabeled dataset, respectively
            for idx in Q_indices:
                labeled_set.append(idx)
                unlabeled_set.remove(idx)

            print("# of Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(unlabeled_set)))
            assert len(labeled_set) == len(list(set(labeled_set))) and len(unlabeled_set) == len(list(set(unlabeled_set)))
            
            # Re-Configure Training of the Next Cycle
            # network = get_model(args, nets, args.model, args.pretrained)
            network = get_text_model(args, MODEL_NAME)

            # dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
            dst_subset = CustomTextSubset(dst_train, labeled_set)
            if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            else:
                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            with open(os.path.join(save_path, f'indices_cycle_{cycle}.pkl'), 'wb') as file:
                pickle.dump(labeled_set, file)
            with open(os.path.join(save_path, f'scores_cycle_{cycle}.pkl'), 'wb') as file:
                pickle.dump(Q_scores.tolist() if type(Q_scores) != list else Q_scores, file)
        print("Final acc logs")
        logs = np.array(logs).reshape((-1, 1))
        print(logs, flush=True)
        with open(os.path.join(save_path, 'indices_after_cycle.pkl'), 'wb') as file:
            pickle.dump(labeled_set, file)
        print(labeled_set)

