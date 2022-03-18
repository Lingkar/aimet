from decimal import Decimal

import onnx
import torch
from aimet_torch.onnx_utils import OnnxSaver
from torch import nn
from torchvision import models
import os

from onnx2pytorch import ConvertModel

# imports required for spatial SVD
from aimet_torch.compress import ModelCompressor
from aimet_torch.defs import SpatialSvdParameters
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters
from torchvision import datasets, models, transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './datasets/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    """
    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    model.eval()
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if preds[j] == labels[j]:
                    correct_count += 1
                total_count += 1
    return correct_count / total_count


# f = open("./thesis/models/resnetBeeAnt.onnx")
model = torch.load("./models/resnetBeeAnt")
print(model)
input_shape = (1, 3, 224, 244)

torch.onnx.export(model, torch.rand(input_shape).cuda(), './models/resnetBeeAnt.onnx')
OnnxSaver.set_node_names('./models/resnetBeeAnt.onnx', model, torch.rand(input_shape))

# Compress the model to 80% (so with 20%)
greed_params = GreedySelectionParameters(target_comp_ratio=0.8)
auto_params = SpatialSvdParameters.AutoModeParams(greed_params)
spatial_svd_params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                          params=auto_params)

comp_model, stats = ModelCompressor.compress_model(model,
                                                   input_shape=input_shape,
                                                   eval_callback=evaluate_model, eval_iterations=None,
                                                   compress_scheme=CompressionScheme.spatial_svd,
                                                   cost_metric=CostMetric.mac,
                                                   parameters=spatial_svd_params,
                                                   visualization_url="http://localhost:5006/")

print("New model:")
print(comp_model)
print(stats)
torch.save(comp_model, './models/compressedBeeAnts/spatial_svd_model_1')

torch.onnx.export(comp_model, torch.rand(input_shape).cuda(), './models/compressedBeeAnts/spatial_svd_model_1.onnx')
OnnxSaver.set_node_names('./models/compressedBeeAnts/spatial_svd_model_1.onnx', comp_model, torch.rand(input_shape))

# Fine tune afterwards
