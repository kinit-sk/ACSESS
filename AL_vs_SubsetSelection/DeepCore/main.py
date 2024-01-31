import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import pickle
import numpy as np
import random

# custom
from arguments import parser
from ptflops import get_model_complexity_info

def main():
    # parse arguments
    args = parser.parse_args()
    gpus = ""
    for i, g in enumerate(args.gpu):
        gpus = gpus+str(g)
        if i != len(args.gpu)-1:
            gpus = gpus+","
            
    state = {k: v for k, v in args._get_kwargs()}
    if args.dataset == 'ImageNet':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = 'cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu'

    args, checkpoint, start_exp, start_epoch = get_more_args(args)

    print(args.pretrained)

    save_path = os.path.join('result', args.dataset)
    if args.strategy != 'basic':
        save_path = os.path.join(save_path, args.strategy)
    save_path = os.path.join(save_path, args.selection, f'{args.resolution}_{args.selection_epochs}{"_pretrained" if args.pretrained else ""}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for exp in range(start_exp, args.num_exp):

        exp = exp+1 #TOFIXTOFIXTOFIX
        if os.path.exists(os.path.join(save_path, f'indices_{exp}.pkl')):
            continue

        # Get checkpoint if have
        if args.save_path != "":
            checkpoint_name = "{dst}_{net}_{mtd}_exp{exp}_se{se}_{dat}_fr{fr}_".format(dst=args.dataset,
                                                                                         net=args.model,
                                                                                         mtd=args.selection,
                                                                                         dat=datetime.now(),
                                                                                         exp=exp,
                                                                                         se=args.selection_epochs,
                                                                                         fr=args.fraction)

        print('\n================== Exp %d ==================' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](args, args.resolution)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        seeds = [42, 1337, 15435, 43543435, 2154864, 345433434, 5435434]
        torch.random.manual_seed(seeds[exp]) # Should change this for changing seed
        np.random.seed(seeds[exp])
        random.seed(seeds[exp])

        # Core-set Selection
        if "subset" in checkpoint.keys():
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            selection_args = dict(epochs=args.selection_epochs,
                                  selection_method=args.uncertainty,
                                  balance=args.balance,
                                  greedy=args.submodular_greedy,
                                  function=args.submodular,
                                  dst_test = dst_test,
                                  torchvision_pretrain=args.pretrained
                                  )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            start_time = time.time()
            ##### Main Function #####
            subset, warmup_test_acc = method.select()
            print("(should be unordered) subset[:10]:", subset["indices"][:10])

            core_selection_time = time.time() - start_time
            print("Elapsed Time: ", core_selection_time)


        print(subset['indices'])
        print(len(subset['indices']))
        with open(os.path.join(save_path, f'indices_{exp}.pkl'), 'wb') as file:
            pickle.dump(subset['indices'], file)

        print(subset['scores'])
        with open(os.path.join(save_path, f'scores_{exp}.pkl'), 'wb') as file:
            pickle.dump(subset['scores'].tolist() if type(subset['scores']) != list else subset['scores'], file)

        continue
        # Handle weighted subset
        if_weighted = "weights" in subset.keys()
        #if if_weighted:
        #    dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
        #else:
        dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.train_batch, shuffle=True,
                                       num_workers=args.workers, pin_memory=False)
            test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                      num_workers=args.workers, pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=False)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=False)

        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        # Model Training
        for model in models:
            print("| Training on model %s" % model)

            # Get configurations for Distrubted SGD
            network, criterion, optimizer, scheduler, rec = get_configuration(args, nets, model, checkpoint, train_loader, start_epoch)
            print("Main Model: {}".format(args.model))
            macs, params = get_model_complexity_info(network, (3, args.im_size[0], args.im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
            print('{:<30}  {:<8}'.format('MACs: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

            # Save the checkpont with only the susbet.
            if args.save_path != "" and args.resume == "":
                save_checkpoint({"exp": exp,
                                 "subset": subset,
                                 "sel_args": selection_args},
                                os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
                                             + "_") + "unknown.ckpt"), 0, 0.)

            ##### Training #####
            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                             "epoch": epoch + 1,
                                             #"state_dict": network.state_dict(),
                                             #"opt_dict": optimizer.state_dict(),
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset,
                                             "elapsed_time": core_selection_time,
                                             "sel_args": selection_args},
                                            os.path.join(args.save_path, checkpoint_name + (
                                                "" if model == args.model else model + "_") + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

            # Prepare for the next checkpoint
            if args.save_path != "":
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
                                     "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
                                     ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
                except:
                    save_checkpoint({"exp": exp,
                                     "epoch": args.epochs,
                                     #"state_dict": network.state_dict(),
                                     #"opt_dict": optimizer.state_dict(),
                                     "best_acc1": best_prec1,
                                     "rec": rec,
                                     "subset": subset,
                                     "sel_args": selection_args},
                                    os.path.join(args.save_path, checkpoint_name +
                                                 ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            print("len(subset): ", len(subset["indices"]))
            start_epoch = 0
            checkpoint = {}
            sleep(2)

if __name__ == '__main__':
    main()