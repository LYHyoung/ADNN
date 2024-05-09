import sys
sys.path.append("./ADNNs/")
sys.path.append("./ADNNs/SDnet/")
sys.path.append("./ADNNs/BlockDrop/")
sys.path.append("./ADNNs/BlockDrop/models/")

import os
import argparse

import Branchynet as Branchynet
import SDnet.network_architectures as arcs
from RANet import RANet as RAnet
from Msdnet import MSDNet as MSDnet
import BlockDrop.utils as utils
import SkipNet as Skipnet

def select_model(model_name):
    if model_name == "BranchyNet":
        return Branchynet.B_AlexNet()

    elif model_name == "SDNet":
        sdn_model, sdn_params = arcs.load_model("./CheckPoints/SDNet/CIFAR10",'test',epoch=-1)
        confidence_thresholds = [0.9] # set for the confidence threshold for early exits
        sdn_model.forward = sdn_model.early_exit
        sdn_model.confidence_threshold = confidence_thresholds[0]
        return sdn_model

    elif model_name == "RANet" :
        args = argparse.ArgumentParser(description="Early Exit CLI")
        args.nBlocks = 2
        #args.Block_base = 2
        args.step = 4
        args.stepmode ='even'
        args.compress_factor = 0.25
        args.nChannels = 16
        args.data = 'cifar10'
        args.growthRate = 6
        args.block_step = 2
        args.grFactor = '4-2-1'
        args.bnFactor = '4-2-1'
        args.scale_list = '1-2-3'
        args.reduction = 0.5
        args.use_valid = True
        args.grFactor = list(map(int, args.grFactor.split('-')))
        args.bnFactor = list(map(int, args.bnFactor.split('-')))
        args.scale_list = list(map(int, args.scale_list.split('-')))
        args.nScales = len(args.grFactor)
        args.splits = ['train', 'val', 'test']
        args.num_classes = 10
        RANet = RAnet(args)
        return RANet

    elif model_name == "MSDNet" :
        args = argparse.ArgumentParser(description="Early Exit CLI")
        args.nBlocks = 3
        args.base = 4
        args.step = 2
        args.stepmode ='even'
        args.compress_factor = 0.25
        args.nChannels = 16
        args.data = 'cifar10'
        args.growthRate = 6
        args.block_step = 2
        args.prune = 'max'
        args.bottleneck =True
        args.grFactor = '1-2-4'
        args.bnFactor = '1-2-4'
        args.reduction = 0.5
        args.use_valid = True
        args.grFactor = list(map(int, args.grFactor.split('-')))
        args.bnFactor = list(map(int, args.bnFactor.split('-')))
        args.nScales = len(args.grFactor)
        args.splits = ['train', 'val', 'test']
        args.num_classes = 10
        MSDNet = MSDnet(args)
        return MSDNet

    elif model_name == "BlockDrop" :
        rnet, agent = utils.get_model('R110_C10')
        return rnet, agent

    elif model_name == "SkipNet" :
        args = argparse.ArgumentParser()
        model_path = os.path.join("./CheckPoints/SkipNet/CIFAR10",'skipnet_10.pth.tar')
        #args.model = 'cifar10_rnn_gate_110'
        args.model = 'cifar10_rnn_gate_38'
        args.dataset = 'cifar10'
        args.resume = model_path
        args.cmd = 'test'
        #args.arch = 'cifar10_rnn_gate_110'
        args.arch = 'cifar10_rnn_gate_38'
        args.batch_size = 1
        args.pretrained = ('pretrained','store_true')
        SkipNet = Skipnet.__dict__[args.arch](args.pretrained)
        return SkipNet

    else:
        print("Model Name Error : [BranchNet, SDNet, RANet, MSDNet, BlockDrop, SkipNet]")
        return
         