"""
Module for performing the baseline benchmarks with state-of-the art models,
i.e. alexnet and VGGNet
"""
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


parser = argparse.ArgumentParser(description= "Baseline training options")
parser.add_argument('model')
parser.add_argument('fitting')
parser.add_argument('num_epochs')
parser.add_argument('lr')


args = parser.parse_args()
print(args)

BATCH_SIZE = 16
IMAGE_SIZE = 208
NUM_CLASSES = 1
CHANNELS = 3
DROPOUT = 0.1
EMB_DROPOUT = 0.1
NUM_EPOCHS = 30
DIM = 1024

class BaselineTrainer:
    """
    Object-oriented handler for training the models necessary for the 
    study baselines. Specifically, alexnet and VGGnet (but the general functionality
    is the same)
    """

    def __init__(self, model, train_dataset, test_dataset, batch_size, loss_fn, optimizer, num_epochs,
                 lr, result_dir, device, top_k=5, k_metric='val'):
        """
        Arguments:
            model: Model to be trained
            train_dataset: Dataset for training, should be an instance of the ImageFolder dataloader for classification tasks
            test_dataset: Dataset for testing, same format requirements as train_dataset
            batch_size: The batch size to use when training
            loss_fn: The loss function to use for backpropagation
            optimizer: The optimizer to use
            num_epochs: The number of epochs to train for
            lr: Learning rate for model training
            result_dir: Directory to save results from training to
            device: The device to run everything on. Defaults to 
        """
        self.model = model
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.train_loader = DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
        self.validation_loader = DataLoader(self.test_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
        self.loss_fn = loss_fn
        self.optim = optimizer
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        #Send the model to the GPU device
        self.model.to(self.device)
        self.result_dir = result_dir
        self.writer = SummaryWriter(log_dir = self.result_dir)
        #Not doing any learning rate scheduling right now
        self.lr_scheduler = None
        #top k
        self.top_k = top_k
    
    def _train_loop(self, epoch):
        self.model.train()
        tot_loss = 0
        n_accurate = 0
        n_total = 0
        for ibatch, (x, y) in enumerate(self.train_loader):
            self.optim.zero_grad()
            inner_step = int(( epoch * len(self.train_loader)) + ibatch)
            #Unpack input and output
            inp, labels = x, y
            inp = inp.to(self.device)
            labels = labels.to(self.device)
            #Send the data one batch at a time to the GPU
            #Should not matter where loss is computed as long as all quantities are consistent
            out = self.model(inp)
            loss = self.loss_fn(out.squeeze(-1), labels.float())
            #print(out)
            loss.backward()
            self.writer.add_scalar("Step Learning Rate", self.optim.param_groups[0]['lr'], inner_step)
            self.optim.step()
            #Step the learning rate scheduler too based on the current optimizer step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if (ibatch % 100 == 99):
                print(f"Epoch: {epoch}\tBatch:{ibatch}\tTrain Loss:{loss.item()}")
            preds = torch.round(torch.nn.Sigmoid()(out.detach().squeeze(-1)))
            n_accurate += torch.sum(preds == labels.data)
            n_total += len(labels)
            tot_loss += loss.item()
            self.writer.add_scalar("Training Step Loss", loss.item(), inner_step)

        self.writer.add_scalar("Avg. Epoch Train Loss", tot_loss / len(self.train_loader), epoch)
        print(f"Epoch: {epoch}\tAverage Epoch Training Loss:{tot_loss / len(self.train_loader)}\t")
        self.writer.add_scalar("Epoch Training Accuracy", n_accurate / n_total, epoch)
        print(f"Epoch: {epoch}\tEpoch Training Accuracy:, {n_accurate / n_total}")
#        return tot_loss / len(self.train_loader)
        return n_accurate/n_total

    def _validation_loop(self, epoch):
        self.model.eval()
        tot_loss = 0
        n_accurate = 0
        n_total = 0
        with torch.no_grad():
            for ibatch, (x, y) in enumerate(self.validation_loader):
                inner_step = int(epoch * len(self.validation_loader) + ibatch)
                inp, labels = x, y
                inp = inp.to(self.device)
                labels = labels.to(self.device)
                #Consistent device migration for data
                out = self.model(inp)
                loss = self.loss_fn(out.squeeze(-1), labels.float())
                #print(out.shape,labels.shape)
                preds = torch.round(torch.nn.Sigmoid()(out.detach().squeeze(-1)))
                n_accurate += torch.sum(preds == labels.data)
                n_total += len(labels)
                if (ibatch % 100 == 99):
                    print(f"Epoch: {epoch}\tBatch: {ibatch}\tValidation Loss:{loss.item()}")
                tot_loss += loss.item()
                self.writer.add_scalar("Validation Step Loss", loss.item(), inner_step)
            self.writer.add_scalar("Avg. Epoch Validation Loss", tot_loss / len(self.validation_loader), epoch)
            print(f"Epoch: {epoch}\tAverage Epoch Validation Loss: {tot_loss / len(self.validation_loader)}\t")
            self.writer.add_scalar("Validation Accuracy", n_accurate / n_total, epoch)
            print(f"Epoch: {epoch}\tEpoch Validation Accuracy: {n_accurate / n_total}")
#            return tot_loss / len(self.validation_loader)
            return n_accurate/n_total
        
    def _save_models(self, tag = ''):
        """
        Saves the model using the torch.save() functionality
        """
        string_model = f'mod_info_{tag}.pt'
        torch.save({"model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optim.state_dict()},
                    self.result_dir + '//' + string_model)

    def _save_dataloaders(self, tag=''):
        string_train = f'train_data_loader{tag}.pth'
        string_val = f'val_data_loader{tag}.pth'
        torch.save(self.train_loader, os.path.join(self.result_dir, string_train))
        torch.save(self.validation_loader, os.path.join(self.result_dir, string_val))

    def _get_curr_saved_models(self):
        all_files = os.listdir(self.result_dir)
        return list(filter(lambda x : 'mod_info' in x, all_files))

    def _clean_up_worst_model(self, target):
        """
        Removes the worst performing model from the result directory with a given
        target in the filename
        """
        target_file = list(filter(lambda x: target == x, os.listdir(self.result_dir)))
        #Not asserting that only one target file exists because it is possible for multiple files to have
        #   the same tags due to rounding in losses.
#        assert(len(target_file) > 0)
        os.remove(os.path.join(self.result_dir, target_file[0]))

    def run(self):
        self._save_dataloaders()
        top_accs = []
        saved_mods = []

        self.optim = self.optim(self.model.parameters(), lr=self.lr)
        train_accs, val_accs = [], []
        for epoch in range(self.num_epochs):
            start = time.time()

            train_acc = self._train_loop(epoch)
            val_acc = self._validation_loop(epoch)
            train_accs.append(train_acc)
            val_accs.append(val_accs)

            cur_acc = val_acc
            if cur_acc in top_accs:
                continue
            if len(top_accs) < self.top_k:
                #Save the model if less than 10 models available or saving every model checkpoint
                #   (self.top_k = -1)
                tag = f"val_acc={cur_acc}"
                top_accs.append(cur_acc)
                top_accs = sorted(top_accs, reverse=True)
                self._save_models(tag=tag)
                saved_mods = self._get_curr_saved_models()
            else:
                #Replace the highest loss model
                worst_acc = sorted(top_accs, reverse=True)[-1]
                if cur_acc > worst_acc:
                    remove_target = f"val_acc={worst_acc}"
                    remove_string = 'mod_info_' + remove_target + '.pt'
                    print(remove_target)
                    #Update the top_losses list
                    top_accs.append(cur_acc)
                    top_accs = sorted(top_accs, reverse=True)[:self.top_k]

                    self._clean_up_worst_model(remove_string)
                    save_tag = f"val_acc={cur_acc}"
                    self._save_models(tag=save_tag)
                    saved_mods = sorted(self._get_curr_saved_models(), reverse=True)[:self.top_k]
            end = time.time()
            print(saved_mods)
            print('finished epoch in: ', end - start)
        #Flush and close the summary writer
        self.writer.flush()
        self.writer.close()
        final_val_acc = self._validation_loop(epoch)
        print(f"Final test loss is {final_val_acc}")
        self._save_models()
        print(f"The top {self.top_k} losses are:")
        print(top_accs)
        print(f"The saved top {self.top_k} models are:")
        print(saved_mods)

if __name__ == "__main__":
    args = parser.parse_args()
    #Initialize the dataset using imagefolder
    test_root = "test"
    train_root = "train"
    val_root = "val"
    #Load the training data
    transform_functions = {
        'alexnet':alexnet_transform,
        'vggnet':vggnet_transform,
        'inception_v3':inception_transform,
        'inception_v4':inception_transform,
        'inception_res':inception_transform,
        'inception_adv':inception_transform,
        'inception_ens':inception_transform
        # 'resnet_18':resnet_transform,
        # 'resnet_50':resnet_transform,
        # 'resnet_152':resnet_transform
    }
    train_data = torchvision.datasets.ImageFolder(train_root, transform=transform_functions[args.model])
    val_data = torchvision.datasets.ImageFolder(val_root, transform=transform_functions[args.model])

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
        print(model)
        trainer = BaselineTrainer(model=model,
                                 train_dataset=train_data,
                                 test_dataset=val_data,
                                 batch_size=16,
                                 loss_fn=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.SGD,
                                 num_epochs=int(args.num_epochs),
                                 lr=float(args.lr),
                                 result_dir='alex_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                                 device=torch.device('cuda:0')) #Try using GPU resources if available

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
        print(model)
        trainer = BaselineTrainer(model=model,
                                 train_dataset=train_data,
                                 test_dataset=val_data,
                                 batch_size=16,
                                 loss_fn=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.SGD,
                                 num_epochs=int(args.num_epochs),
                                 lr=float(args.lr),
                                 result_dir='vgg_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                                 device=torch.device('cuda:0')) #Try using GPU resources if available

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
        print(model)
        trainer = BaselineTrainer(model=model,
                                 train_dataset=train_data,
                                 test_dataset=val_data,
                                 batch_size=16,
                                 loss_fn=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.SGD,
                                 num_epochs=int(args.num_epochs),
                                 lr=float(args.lr),
                                 result_dir='v3_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                                 device=torch.device('cuda:0')) #Try using GPU resources if available

    if args.model == 'inception_v4':
        if args.fitting == 'untrained':
            model = timm.create_model('inception_v4', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('inception_v4', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'last_linear.weight' or name == 'last_linear.bias':
                    para.requires_grad = True
        print(model)
        trainer = BaselineTrainer(model=model,
                         train_dataset=train_data,
                         test_dataset=val_data,
                         batch_size=16,
                         loss_fn=torch.nn.BCEWithLogitsLoss(),
                         optimizer=torch.optim.SGD,
                         num_epochs=int(args.num_epochs),
                         lr=float(args.lr),
                         result_dir='v4_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                         device=torch.device('cuda:0'))

    if args.model == 'inception_res':
        if args.fitting == 'untrained':
            model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'classif.weight' or name == 'classif.bias':
                    para.requires_grad = True
        print(model)
        trainer = BaselineTrainer(model=model,
                         train_dataset=train_data,
                         test_dataset=val_data,
                         batch_size=16,
                         loss_fn=torch.nn.BCEWithLogitsLoss(),
                         optimizer=torch.optim.SGD,
                         num_epochs=int(args.num_epochs),
                         lr=float(args.lr),
                         result_dir='res_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                         device=torch.device('cuda:0'))

    if args.model == 'inception_adv':
        if args.fitting == 'untrained':
            model = timm.create_model('adv_inception_v3', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('adv_inception_v3', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'fc.weight' or name == 'fc.bias':
                    para.requires_grad = True
        print(model)
        trainer = BaselineTrainer(model=model,
                         train_dataset=train_data,
                         test_dataset=val_data,
                         batch_size=16,
                         loss_fn=torch.nn.BCEWithLogitsLoss(),
                         optimizer=torch.optim.SGD,
                         num_epochs=int(args.num_epochs),
                         lr=float(args.lr),
                         result_dir='adv_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                         device=torch.device('cuda:0'))

    if args.model == 'inception_ens':
        if args.fitting == 'untrained':
            model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=False, num_classes=NUM_CLASSES)
        else:
            model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)
            for name, para in model.named_parameters():
                para.requires_grad = args.fitting == 'partial'
                if name == 'classif.weight' or name == 'classif.bias':
                    para.requires_grad = True
        print(model)
        trainer = BaselineTrainer(model=model,
                         train_dataset=train_data,
                         test_dataset=val_data,
                         batch_size=16,
                         loss_fn=torch.nn.BCEWithLogitsLoss(),
                         optimizer=torch.optim.SGD,
                         num_epochs=int(args.num_epochs),
                         lr=float(args.lr),
                         result_dir='ens_{}_lr{}_epochs{}'.format(args.fitting, args.lr, args.num_epochs),
                         device=torch.device('cuda:0'))

    trainer.run()
