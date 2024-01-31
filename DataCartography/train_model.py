import os
from datasets import DOG, AWA, APL, ACT, TEX_DTD, PRT, PNU, FLW
from nets.resnet import * 
from torch import nn
import torch
import numpy as np
import json
import argparse

EPOCHS = 100
BATCH_SIZE = 64
# DATASET = 'APL'
DATASET = 'AWA'
# DATASET = 'DOG'
REPEATS = 10

parser = argparse.ArgumentParser(description='Parameter Processing')

# Basic arguments
parser.add_argument('--dataset', type=str, default='DOG', help='dataset')
args = parser.parse_args()

DATASET = args.dataset
print(f'Running dataset {DATASET}')

dataset_path = os.path.join('/', 'home', 'bpecher', 'sample_choice_experiments', 'dataset_creation', 'Data', DATASET)

for repeat in range(REPEATS):
    print(f'Running repeat {repeat}')

    save_path = os.path.join('results', DATASET, 'training_dynamics', f'Adam_{EPOCHS}_pr_{repeat}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if os.path.exists(os.path.join('results', DATASET, 'training_dynamics', f'Adam_{EPOCHS}_pr_{repeat + 1}')) or (repeat == 9 and os.path.exists(os.path.join(save_path, f'dynamics_epoch_{EPOCHS - 1}.jsonl'))):
        print('Skipping!')
        continue

    dataset_constr = {
        'AWA': AWA,
        'DOG': DOG,
        'APL': APL,
        'ACT_410': ACT,
        'TEX_DTD': TEX_DTD,
        'PRT': PRT,
        'PNU': PNU,
        'FLW': FLW,
    }


    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = DOG(dataset_path, 224)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = dataset_constr[DATASET](dataset_path, 224)
    pretrained = True

    model = ResNet18(channel, num_classes, im_size=im_size, pretrained=pretrained)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__()

    # model_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        train_indices = np.arange(len(dst_train))
        model.train()
        trainset_permutation_inds = np.random.permutation(train_indices)

        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=BATCH_SIZE, drop_last=False)
        # trainset_permutation_inds = list(batch_sampler)

        loader = torch.utils.data.DataLoader(dst_train, shuffle=False, batch_sampler=batch_sampler, num_workers=1, pin_memory=False)

        dynamics = []
        correct = 0
        all = len(trainset_permutation_inds)

        for i, data in enumerate(loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()

            model_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss = loss.mean()

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()

            loss.backward()
            model_optimizer.step()

        print(f'Epoch: {epoch} - Train Acc: {100. * correct/all}; Loss: {loss.item()}')

        model.eval()
        eval_loader = torch.utils.data.DataLoader(dst_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=False)
        for batch_ids, data in enumerate(eval_loader):
            input, target = data
            output = model(input.cuda())
            cpu_outputs = output.detach().cpu().numpy().tolist()
            cpu_targets = target.detach().cpu().numpy().tolist()

            for idx, output in enumerate(cpu_outputs):
                guid = trainset_permutation_inds[i * BATCH_SIZE + idx]
                guid = dst_train.samples[trainset_permutation_inds[i * BATCH_SIZE + idx]][0].split('/')[-1].split('.')[0]
                dynamics.append({
                    'guid': int(guid),
                    f'logits_epoch_{epoch}': output,
                    'gold': cpu_targets[idx]
                })

        with open(os.path.join(save_path, f'dynamics_epoch_{epoch}.jsonl'), 'w') as file:
            json.dump(dynamics, file)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=64, shuffle=False, num_workers=1, pin_memory=False)

        correct = 0.
        total = 0.

        for batch_ids, data in enumerate(test_loader):
            input, target = data
            output = model(input.cuda())

            loss = criterion(output, target.cuda()).sum()

            _, predicted = torch.max(output.data, 1)
            correct += predicted.detach().cpu().eq(target).sum().item()
            total += target.size(0)

        acc = 100. * correct / total
        print(f'Test Acc: {acc}')


