import numpy as np
import torch, os
from torch.utils.data import DataLoader
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from tqdm import tqdm
import argparse
from preprocessing import alexnet_transform, vggnet_transform,inception_transform
import timm

transform_functions = {
    'alex': alexnet_transform,
    'vgg': vggnet_transform,
    'v3': inception_transform,
    'v4': inception_transform,
    'res': inception_transform,
    'adv': inception_transform,
    'ens': inception_transform
}

"""
Module that performs evaluation of a trained model over a standardized test set
"""
parser = argparse.ArgumentParser(description= "Evaluation folder")
parser.add_argument('model')
parser.add_argument('fitting')
parser.add_argument('epochs')

NUM_CLASSES = 1

def load_models_from_dir(model_base, checkpoint_directory, load_to_gpu=True):
    """
    Loads all the models saved as checkpoints in checkpoint_directory

    model_base: Base example of model that is compatible with the model.load_state_dict() method
    checkpoint_directory: The directory that contains all of the model checkpoints to evaluate
    load_to_gpu: Maps the loaded model to the GPU for faster inference. Defaults to True

    Note: model checkpoints should have the .pt file extension!
    """
    constructed_models = []
    all_files = os.listdir(checkpoint_directory)
    checkpoint_files = list(filter(lambda x: 'mod_info_val_acc' in x, all_files))
    print(checkpoint_files)
    for chkpt in checkpoint_files:
        full_path = os.path.join(checkpoint_directory, chkpt)
        model_checkpoint = torch.load(full_path, map_location=torch.device('cuda:0') if load_to_gpu else torch.device('cpu'))
        print(chkpt, ' loaded')
        assert ('model_state_dict' in model_checkpoint)
        assert ('optimizer_state_dict' in model_checkpoint)
        mod_template = deepcopy(model_base)
        if load_to_gpu:
            mod_template.to(torch.device('cuda:0'))
        mod_template.load_state_dict(model_checkpoint['model_state_dict'])
        print(chkpt, ' copied')
        constructed_models.append(mod_template)
    return constructed_models


def visualize_confusion_matrix(confusion_matrix, labels, save_path, include_cbar=True):
    """
    Generates a seaborn heatmap viualization of a confusion matrix

    confusion_matrix: Numpy array representation of the confusion matrix. For binary classification,
        a 2x2 matrix
    labels: The labels for the rows and columns of the confusion matrix
    save_path: Full file path for saving the generated confusion matrix
    include_cbar: Whether the colorbar should be included for the heatmap. Defaults to False.

    Note: Make sure the given labels match up with the row ordering of the given confusion matrix
    """
    df_cfm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    plt.figure()
    cfm_plot = sns.heatmap(df_cfm, annot=True, cbar=include_cbar, fmt='d')
    cfm_plot.tick_params(left=True, bottom=False, top=True, labelleft=True, labeltop=True,
                         labelbottom=False)
    cfm_plot.figure.savefig(save_path)
    plt.show()


class ModelEval:

    def __init__(self, models, test_data, gpu_inference=True):
        """
        models: A list of model checkpoints to be used for evaluation
        test_data: The data used for testing the model's performance separately
            from the training or validation data
        gpu_inference: Flag indicating if inference is done on a GPU. Defaults to True
        """
        self.models = models
        self.test_loader = DataLoader(test_data, batch_size=1)
        if gpu_inference:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self._get_predictions()

    def _get_predictions(self):
        true_labels = []
        predictions = []
        for _, (x, y) in tqdm(enumerate(self.test_loader)):
            inp, label = x, y
            true_labels.append(label.item())
            predicted_labels = []
            # Generate a prediction from each model
            for model in self.models:
                model.eval()
                out = model(inp.to(self.device))
                pred_label = torch.round(torch.nn.Sigmoid()(out.detach().squeeze(-1)))
                predicted_labels.append(pred_label.item())
            # Generate the final prediction by averaging over each individual model
            #   and rounding the final answer
            mean_label = np.mean(predicted_labels)
            if mean_label >= 0.5:
                predictions.append(1)
            elif mean_label < 0.5:
                predictions.append(0)
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        self.true_labels = true_labels
        self.predictions = predictions
        return true_labels, predictions

    def generate_confusion_matrix(self):
        """
        Should be called after generating predictions
        """
        unique_labels = len(np.unique(self.true_labels))
        print('making matrix')
        confusion_matrix = np.zeros((unique_labels, unique_labels))
        assert (len(self.true_labels) == len(self.predictions))
        np.add.at(confusion_matrix, (self.true_labels, self.predictions), 1)
        confusion_matrix = confusion_matrix.astype(np.int32)
        return confusion_matrix


if __name__ == "__main__":
    """
    Note that all the code here is specifically geared for ViT evaluation. However,
    you can change this as needed for other model architectures/types
    """
    args = parser.parse_args()

    test_root = "test"
    transform = transform_functions[args.model]
    test_data_raw = torchvision.datasets.ImageFolder(test_root, transform=transform)

    # Create instance of base model
    print("Loading models...")

    # Create all models for a specific directory
    folder = args.model + '_' + args.fitting + '_lr1e-4_epochs' + args.epochs
    tst_dir = folder


    if args.model == 'alex':
        model_base = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model_base.classifier[6] = torch.nn.Linear(4096, 1)

    if args.model == 'vgg':
        model_base = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        model_base.classifier[6] = torch.nn.Linear(4096, 1)

    if args.model == 'v3':
        model_base = timm.create_model('inception_v3', pretrained=True, num_classes=NUM_CLASSES)

    if args.model == 'v4':
        model_base = timm.create_model('inception_v4', pretrained=True, num_classes=NUM_CLASSES)

    if args.model == 'res':
        model_base = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)

    if args.model == 'ens':
        model_base = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True, num_classes=NUM_CLASSES)

    if args.model == 'adv':
        model_base = timm.create_model('adv_inception_v3', pretrained=True, num_classes=NUM_CLASSES)

    all_models = load_models_from_dir(model_base, tst_dir)

    # Initialize an instance of model evaluation which generates predictions
    print("Generating predictions...")
    eval_handler = ModelEval(all_models, test_data_raw)
    confusion_matrix = eval_handler.generate_confusion_matrix()

    # Visualize confusion matrix
    print("Visualizing confusion matrices...")
    save_file = args.model + '_' + args.fitting + '_tst_cfm.png'
    visualize_confusion_matrix(confusion_matrix, ['Non Demented', 'Demented'], save_file)
    print("Done")

