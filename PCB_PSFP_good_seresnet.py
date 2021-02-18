from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers_partloss import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from utils import ParameterType
import model_adapters
from prune_utils import prune_fc_like, get_prune_index_ratio, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, \
    gng_criterion, init_from_pretrained, create_conv_tensor, create_new_bn, zeroed_criterion

import functools

def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                    transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    return dataset, num_classes, train_loader, query_loader, gallery_loader


def  main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset,  args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 )


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes,cut_at_pooling=False, FCN=True)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
#        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    model = model.cuda()


    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader,  dataset.query, dataset.gallery)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if hasattr(model, 'base'):
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    #model = torch.load("all_saves/resnet50_duke_59").cuda()


    # Start training


    # PSFP§

    model_adapter = model_adapters.ResNetAdapter()

    target_prune = 0.5 #50%
    hoel_magic_value = 0.147  # D always at 1/8
    k = -np.log(hoel_magic_value) / args.epochs
    a = target_prune / (np.exp(-k * args.epochs) - 1)

    def num_remain_from_expo(original_c, k, a, epoch):
        decay = a * np.exp(-k * epoch) - a
        num_weak = decay * original_c
        num_remain = original_c - num_weak
        return num_remain

    decay_rates_c = {}
    original_c = {}

    for name, parameters in model.base.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            #decay_rates_c[name] = (np.log(parameters.shape[0]) - np.log(target_prune[name])) / args.epochs
            original_c[name] = parameters.shape[0]
        original_c['local_conv'] = model.local_conv.out_channels

    finished_list = False
    type_list = []
    forced_remove = False
    removed_filters_total = 0
    removed_parameters_total = 0
    is_last = False
    zero_initializer = functools.partial(torch.nn.init.constant_, val=0)

    for epoch in range(1, args.epochs + 1):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        prune_index_dict, _ = L2_criterion(model=model.base, model_adapter=model_adapter)
        out_channels_keep_indexes = []
        in_channels_keep_indexes = []
        reset_indexes = []
        first_fc = False
        removed_filters_total_epoch = 0

        for name, parameters in model.base.named_parameters():
            #print(name)
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if not finished_list:
                type_list.append(param_type)

            if layer_index == -1 or ('seresnet' in args.arch and  layer_index == 0):
                # Handling CNN and BN before Resnet
                if param_type == ParameterType.CNN_WEIGHTS:
                    sorted_filters_index = prune_index_dict[name]
                    conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                    original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                    # prune_target = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                    prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                    keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                     sorted_filters_index, forced_remove)

                    if len(reset_indexes) != 0:
                        conv_tensor.weight.data[:, reset_indexes[-1], :, :] = zero_initializer(
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                    conv_tensor.weight.data[reset_index] = zero_initializer(conv_tensor.weight.data[reset_index])
                    if conv_tensor.bias is not None:
                        conv_tensor.bias.data[reset_index] = zero_initializer(conv_tensor.bias.data[reset_index])
                    removed_filters_total_epoch += reset_index.shape[0]

                    in_c = 3
                    if len(reset_indexes) != 0:
                        in_c = reset_indexes[-1].shape[0]
                    removed_parameters_total = + reset_index.shape[0] * in_c * parameters.shape[2:].numel()

                    start_index = (keep_index.sort()[0], reset_index)
                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    else:
                        in_channels_keep_indexes.append(None)
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.CNN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    bn_tensor.running_var.data[reset_index] = zero_initializer(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = zero_initializer(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = zero_initializer(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = zero_initializer(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.FC_WEIGHTS and first_fc == False:
                    fc_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                    reset_index = reset_indexes[-1]

                    reshaped_fc = fc_tensor.weight.data.view(fc_tensor.weight.data.shape[0], original_out_channels, -1)
                    reshaped_fc[:, reset_index, :] = zero_initializer(reshaped_fc[:, reset_index, :])
                    fc_tensor.weight.data = reshaped_fc.view(fc_tensor.weight.data.shape[0], -1)

                    first_fc = True
                    finished_list = True

            else:
                if param_type == ParameterType.CNN_WEIGHTS:

                    if tensor_index == 1:
                        sorted_filters_index = prune_index_dict[name]
                        conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                        original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                        prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                        keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                         sorted_filters_index, forced_remove)

                        if len(reset_indexes) != 0:
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :] = zero_initializer(
                                conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                        conv_tensor.weight.data[reset_index] = zero_initializer(conv_tensor.weight.data[reset_index])
                        if conv_tensor.bias is not None:
                            conv_tensor.bias.data[reset_index] = zero_initializer(conv_tensor.bias.data[reset_index])

                        in_c = conv_tensor.in_channels
                        if len(out_channels_keep_indexes) != 0:
                            in_c = reset_indexes[-1].shape[0]
                        removed_filters_total_epoch += reset_index.shape[0]
                        removed_parameters_total = + reset_index.shape[0] * in_c * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        else:
                            in_channels_keep_indexes.append(None)
                        out_channels_keep_indexes.append(keep_index.sort()[0])

                    if tensor_index == 2:
                        sorted_filters_index = prune_index_dict[name]
                        conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                        original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                        prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                        keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                         sorted_filters_index, forced_remove)

                        if len(reset_indexes) != 0:
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :] = zero_initializer(
                                conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                        conv_tensor.weight.data[reset_index] = zero_initializer(conv_tensor.weight.data[reset_index])
                        if conv_tensor.bias is not None:
                            conv_tensor.bias.data[reset_index] = zero_initializer(conv_tensor.bias.data[reset_index])

                        in_c = conv_tensor.in_channels
                        if len(out_channels_keep_indexes) != 0:
                            in_c = reset_indexes[-1].shape[0]
                        removed_filters_total_epoch += reset_index.shape[0]
                        removed_parameters_total = + reset_index.shape[0] * in_c * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        else:
                            in_channels_keep_indexes.append(None)
                        out_channels_keep_indexes.append(keep_index.sort()[0])

                    elif tensor_index == 3:

                        downsample_cnn, d_name = model_adapter.get_downsample(model.base, layer_index, block_index)
                        if downsample_cnn is not None:

                            sorted_filters_index = prune_index_dict[d_name]
                            original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                            last_keep_index, last_reset_index = start_index
                            prune_target = original_c[name]
                            # prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                            # prune_target = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                            # keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                            #                                                  sorted_filters_index, forced_remove)
                            keep_index = torch.range(0, original_out_channels - 1, dtype=torch.long)

                            downsample_cnn.weight.data[:, last_reset_index, :, :] = zero_initializer(
                                downsample_cnn.weight.data[:, last_reset_index, :, :])

                            removed_filters_total_epoch += reset_index.shape[0]
                            removed_parameters_total += reset_index.shape[0] * last_reset_index.shape[
                                0] * parameters.shape[2:].numel()
                            start_index = (keep_index.sort()[0], reset_index)


                        original_out_channels = parameters.shape[0]
                        conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                        keep_index, reset_index = start_index

                        if len(reset_indexes) != 0:
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :] = zero_initializer(
                                conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                        conv_tensor.weight.data[reset_index] = zero_initializer(conv_tensor.weight.data[reset_index])
                        if conv_tensor.bias is not None:
                            conv_tensor.bias.data[reset_index] = zero_initializer(conv_tensor.bias.data[reset_index])

                        removed_filters_total_epoch += reset_index.shape[0]
                        removed_parameters_total = + reset_index.shape[0] * reset_indexes[-1].shape[
                            0] * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        out_channels_keep_indexes.append(keep_index.sort()[0])

                        # se_cnn, _ = model_adapter.get_se(model.base, layer_index, block_index)
                        # if se_cnn is not None:
                        #     se_cnn.weight.data[:, reset_indexes[-1], :, :] = zero_initializer(
                        #     se_cnn.weight.data[:, reset_indexes[-1], :, :])


                elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS:

                    keep_index, reset_index = start_index
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(last_keep_index.sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    bn_tensor.running_var.data[reset_index] = zero_initializer(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = zero_initializer(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = zero_initializer(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = zero_initializer(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.DOWNSAMPLE_BN_W:

                    bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                    keep_index, reset_index = start_index

                    bn_tensor.running_var.data[reset_index] = zero_initializer(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = zero_initializer(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = zero_initializer(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = zero_initializer(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.DOWNSAMPLE_BN_B:
                    keep_index, reset_index = start_index
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])


                elif param_type == ParameterType.CNN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

        #Prune local conv
        local_conv_tensor = model._modules['local_conv']
        original_out_channels = local_conv_tensor.out_channels
        filters_L2 = local_conv_tensor.weight.data.view(original_out_channels, -1).norm(dim=1, p=2)
        _, local_conv_sorted_filters_index = filters_L2.sort()
        prune_target = num_remain_from_expo(original_c['local_conv'], k, a, epoch)
        keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                         local_conv_sorted_filters_index, forced_remove)
        local_conv_tensor.weight.data[reset_index] = zero_initializer(local_conv_tensor.weight.data[reset_index])
        if local_conv_tensor.bias is not None:
            local_conv_tensor.bias.data[reset_index] = zero_initializer(local_conv_tensor.bias.data[reset_index])
        local_bn_tensor = model._modules['feat_bn2d']
        local_bn_tensor.running_var.data[reset_index] = zero_initializer(local_bn_tensor.running_var.data[reset_index])
        local_bn_tensor.running_mean.data[reset_index] = zero_initializer(local_bn_tensor.running_mean.data[reset_index])
        local_bn_tensor.weight.data[reset_index] = zero_initializer(local_bn_tensor.weight.data[reset_index])
        local_bn_tensor.bias.data[reset_index] = zero_initializer(local_bn_tensor.bias.data[reset_index])




        # print("Total: {}".format(removed_filters_total))
        removed_filters_total = removed_filters_total_epoch
        if removed_filters_total - removed_filters_total_epoch == 0:
            forced_remove = True
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)

    # Final removal

    zero_indexes, remain_indexes = zeroed_criterion(model=model.base, model_adapter=model_adapter)
    out_channels_keep_indexes = []
    in_channels_keep_indexes = []
    reset_indexes = []
    first_fc = False
    keep_index = 0
    original_out_channels = 0
    start_index = 0


    for name, parameters in model.base.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if not finished_list:
            type_list.append(param_type)

        if layer_index == -1  or ('seresnet' in args.arch and  layer_index == 0):
            # Handling CNN and BN before Resnet
            if param_type == ParameterType.CNN_WEIGHTS:
                only_zero_filters_index = zero_indexes[name]
                keep_index = remain_indexes[name]
                conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                original_out_channels = parameters.shape[0]

                new_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, zero_initializer, keep_index, []).cuda()
                model_adapter.set_layer(model.base, param_type, new_tensor, tensor_index, layer_index, block_index)

                start_index = (keep_index.sort()[0], reset_index)

                if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                else:
                    in_channels_keep_indexes.append(None)
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.CNN_BIAS:
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

            elif param_type == ParameterType.BN_WEIGHT:
                bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                keep_index = out_channels_keep_indexes[-1]

                n_bn = create_new_bn(bn_tensor, keep_index, []).cuda()
                model_adapter.set_layer(model.base, param_type, n_bn, tensor_index, layer_index, block_index)

                if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_BIAS:
                if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.FC_WEIGHTS and first_fc == False:
                fc_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                new_fc_weight = prune_fc_like(fc_tensor.weight.data, out_channels_keep_indexes[-1],
                                              original_out_channels)
                new_fc_bias = None
                if fc_tensor.bias is not None:
                    new_fc_bias = fc_tensor.bias.data
                new_fc_tensor = nn.Linear(new_fc_weight.shape[1], new_fc_weight.shape[0],
                                          bias=new_fc_bias is not None).cuda()
                new_fc_tensor.weight.data = new_fc_weight
                if fc_tensor.bias is not None:
                    new_fc_tensor.bias.data = new_fc_bias
                model_adapter.set_layer(model.base, param_type, new_fc_tensor, tensor_index, layer_index, block_index)

                first_fc = True
                finished_list = True

        else:
            if param_type == ParameterType.CNN_WEIGHTS:

                if tensor_index == 1:
                    only_zero_filters_index = zero_indexes[name]
                    keep_index = remain_indexes[name]
                    conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index,
                                                          block_index)
                    original_out_channels = parameters.shape[0]

                    new_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, zero_initializer,
                                                    keep_index, []).cuda()
                    model_adapter.set_layer(model.base, param_type, new_tensor, tensor_index, layer_index,
                                            block_index)

                    if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    else:
                        in_channels_keep_indexes.append(None)
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                if tensor_index == 2:
                    only_zero_filters_index = zero_indexes[name]
                    keep_index = remain_indexes[name]
                    conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index,
                                                          block_index)
                    original_out_channels = parameters.shape[0]

                    new_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, zero_initializer,
                                                    keep_index, []).cuda()
                    model_adapter.set_layer(model.base, param_type, new_tensor, tensor_index, layer_index,
                                            block_index)

                    if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    else:
                        in_channels_keep_indexes.append(None)
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif tensor_index == 3:

                    downsample_cnn, d_name = model_adapter.get_downsample(model.base, layer_index, block_index)
                    if downsample_cnn is not None:

                        last_keep_index, _ = start_index
                        only_zero_filters_index = zero_indexes[d_name]
                        original_out_channels = parameters.shape[0]
                        keep_index = torch.range(0, original_out_channels - 1, dtype=torch.long)

                        last_start_conv = create_conv_tensor(downsample_cnn, [last_keep_index], zero_initializer,
                                                        keep_index, []).cuda()
                        last_start_conv = [last_start_conv, 0, layer_index, block_index]

                        start_index = (keep_index.sort()[0], [])

                    original_out_channels = parameters.shape[0]
                    conv_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)
                    keep_index, reset_index = start_index
                    new_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, zero_initializer,
                                                    keep_index, []).cuda()
                    model_adapter.set_layer(model.base, param_type, new_tensor, tensor_index, layer_index,
                                            block_index)


                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                    # if se_cnn is not None:
                    #     se_start_conv = create_conv_tensor(se_cnn, out_channels_keep_indexes, zero_initializer,
                    #                                        torch.range(0, se_cnn.out_channels - 1, dtype=torch.long), []).cuda()
                    #     model_adapter.set_se_pre_layer(model.base, se_start_conv,
                    #                                    layer_index,
                    #                                    block_index)



            elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS:

                last_start_conv, tensor_index, layer_index, block_index = last_start_conv
                model_adapter.set_layer(model.base, ParameterType.DOWNSAMPLE_WEIGHTS, last_start_conv, tensor_index,
                                        layer_index,
                                        block_index)

                in_channels_keep_indexes.append(last_keep_index.sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_WEIGHT:
                bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                keep_index = out_channels_keep_indexes[-1]

                n_bn = create_new_bn(bn_tensor, keep_index, []).cuda()
                model_adapter.set_layer(model.base, param_type, n_bn, tensor_index, layer_index, block_index)

                if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_BIAS:
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.DOWNSAMPLE_BN_W:

                bn_tensor = model_adapter.get_layer(model.base, param_type, tensor_index, layer_index, block_index)

                keep_index, reset_index = start_index

                if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.DOWNSAMPLE_BN_B:
                keep_index, reset_index = start_index
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])


            elif param_type == ParameterType.CNN_BIAS:
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

    #fINAL REMOVE LOCAL CONV
    local_conv_tensor = model._modules['local_conv']
    original_out_channels = local_conv_tensor.out_channels
    filters_L2 = local_conv_tensor.weight.data.view(original_out_channels, -1).norm(dim=1, p=2)
    zero_indexes_local_conv  = (filters_L2 != 0).nonzero().squeeze()
    keep_indexs_local_conv = filters_L2.nonzero().squeeze()
    new_tensor = create_conv_tensor(local_conv_tensor, [], zero_initializer, keep_indexs_local_conv, []).cuda()
    model._modules['local_conv'] = new_tensor

    local_bn_tensor = model._modules['feat_bn2d']
    n_bn = create_new_bn(local_bn_tensor, keep_indexs_local_conv, []).cuda()
    model._modules['feat_bn2d'] = n_bn

    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance0')
    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance1')
    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance2')
    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance3')
    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance4')
    prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, 'instance5')

    trainer.train(epoch + 1, train_loader, optimizer)
    is_best = True
    save_checkpoint({
        'state_dict': model.state_dict(),
        'epoch': epoch + 1,
        'best_top1': best_top1,
    }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    torch.save(model, "all_saves/PSPF_{}_{}_{}.p".format(args.arch, args.dataset, epoch + 1))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)


def prune_instances_fc(keep_indexs_local_conv, model, original_out_channels, instance_name):
    ifc = model._modules[instance_name]
    new_fc_weight = prune_fc_like(ifc.weight.data, keep_indexs_local_conv,
                                  original_out_channels)
    new_fc_bias = None
    if ifc.bias is not None:
        new_fc_bias = ifc.bias.data
    new_fc_tensor = nn.Linear(new_fc_weight.shape[1], new_fc_weight.shape[0],
                              bias=new_fc_bias is not None).cuda()
    new_fc_tensor.weight.data = new_fc_weight
    if ifc.bias is not None:
        new_fc_tensor.bias.data = new_fc_bias
    model._modules[instance_name] = new_fc_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
