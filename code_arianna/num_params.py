import torch
import torchvision
import numpy as np
import os, pickle
from preprocessing import alexnet_transform, vggnet_transform,inception_transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import timm


parser = argparse.ArgumentParser(description= "Evaluation folder")
parser.add_argument('model')
parser.add_argument('fitting')

NUM_CLASSES = 1

if __name__ == "__main__":
    args = parser.parse_args()

    # methods from pytorch available
    if args.model == 'alexnet':
        if args.fitting == 'untrained':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        else:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
        # Modify the Alexnet model to predict for the two classes in our dataset
        model.classifier[6] = torch.nn.Linear(4096, 1)
        #Print a quick summary of the model
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    if args.model == 'vggnet':
        if args.fitting == 'untrained':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
        else:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
        #Modify the vgg model to predict for the two classes in our dataset
        model.classifier[6] = torch.nn.Linear(4096, 1)
        #Print a quick summary of the model
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    # methods from hugging face
    if args.model == 'inception_v3':
        if args.fitting == 'untrained':
            model = timm.create_model('inception_v3', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('inception_v3', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'fc.weight' or name == 'fc.bias':
                    para.requires_grad = True
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    if args.model == 'inception_v4':
        if args.fitting == 'untrained':
            model = timm.create_model('inception_v4', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('inception_v4', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'last_linear.weight' or name == 'last_linear.bias':
                    para.requires_grad = True
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    if args.model == 'inception_res':
        if args.fitting == 'untrained':
            model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'classif.weight' or name == 'classif.bias':
                    para.requires_grad = True
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    if args.model == 'inception_adv':
        if args.fitting == 'untrained':
            model = timm.create_model('adv_inception_v3', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('adv_inception_v3', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'fc.weight' or name == 'fc.bias':
                    para.requires_grad = True
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)

    if args.model == 'inception_ens':
        if args.fitting == 'untrained':
            model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'classif.weight' or name == 'classif.bias':
                    para.requires_grad = True
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(args.model, 'with', args.fitting, 'total params', model_total_params)
    