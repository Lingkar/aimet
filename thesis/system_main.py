# For s3 dataLoader
import os
from typing import Tuple, List, Optional
import shutil

import torch
from aimet_torch.compress import ModelCompressor
from aimet_torch.defs import SpatialSvdParameters
from aimet_common.defs import GreedySelectionParameters, CompressionScheme, CostMetric

# On model compression start ->
# 1. Spatial svd eval mapping
# 2.a Choose set of svd compressed models. (90%, 80%, 70%, 60%, 50%)
# 2.b (Optional) retrain svd compressed models for ~10 epochs
# 3.a Channel pruning eval map on most compressed svd model.
# 3.b Channel pruning eval map on least compressed svd model. Check if results are always similar.
# 3.c Choose set of fully compressed models. Based on MAC's. As percentages will become weird.
# 3.d Retrain fully compressed models for ~10 epochs
# 4. Will possibly have multiple models with similar resource usage, ones with the best accuracy should be preferred.
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class ClassificationModelForCompression:
    def __init__(self, model_name: str, input_shape: Tuple, data_loader_train: DataLoader,
                 data_loader_val: DataLoader,
                 eval_iterations, targets: List[float]):
        self.model_name = model_name
        self.model_path = os.path.join('./models', model_name)
        self.dir_names = ['evalScores', 'svdModels', 'compressedModels']

        self.model = torch.load(os.path.join(self.model_path, 'base'))
        self.input_shape = input_shape
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        # Can be used to speed up the AIMET search
        self.eval_iterations = eval_iterations

        self.targets = targets

        self.eval_scores_svd: Optional[str] = None
        self.eval_scores_cp: Optional[str] = None

        self.createModelDirectories()
        self.svdModels = {}

    def createModelDirectories(self):
        for dir_name in self.dir_names:
            path = os.path.join(self.model_path, dir_name)
            if not os.path.exists(path):
                os.makedirs(path)

    def generate_spatial_svd(self):
        greed_params = GreedySelectionParameters(target_comp_ratio=self.targets[0], num_comp_ratio_candidates=10,
                                                 use_monotonic_fit=True, saved_eval_scores_dict=self.eval_scores_svd)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greed_params, modules_to_ignore=None)
        spatial_svd_params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                                  params=auto_params)

        model, _ = \
            ModelCompressor.compress_model(self.model,
                                           input_shape=self.input_shape,
                                           eval_callback=self.evaluate_model_cb,
                                           eval_iterations=self.eval_iterations,
                                           compress_scheme=CompressionScheme.spatial_svd,
                                           cost_metric=CostMetric.mac,
                                           parameters=spatial_svd_params,
                                           visualization_url='http://localhost:5006/')

        # Save the first target untrained svd model
        torch.save(model, os.path.join(self.model_path, self.dir_names[1] + '_untrained_' + str(
            self.evaluate_model_cb(model, None, True))))
        shutil.copy2('./data/greedy_selection_eval_scores_dict.pkl',
                     os.path.join(self.model_path, self.dir_names[0], 'eval_scores_svd.pkl'))

    def evaluate_model_cb(self, model: torch.nn.Module, eval_iterations, use_cuda: bool = False) -> float:
        """
        Callback function to evaluate a given model.
        :param model: Model to evaluate
        :param eval_iterations: Number of iterations to use for evaluation.
                None for entire epoch.
        :param use_cuda: If true, evaluate using gpu acceleration
        :return: single float number (accuracy) representing model's performance
        """
        if use_cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = 'cpu'

        model.eval()
        total_count = 0
        correct_count = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.data_loader_val):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    if preds[j] == labels[j]:
                        correct_count += 1
                    total_count += 1
        return correct_count / total_count


# def spatial_svd(base_model: ModelForCompression):

if __name__ == '__main__':
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
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    model_compressor = ClassificationModelForCompression('beeAnts', (1, 3, 224, 244),
                                                         data_loaders['train'],
                                                         data_loaders['val'], None, [0.9, 0.8, 0.7, 0.6, 0.5])
    model_compressor.generate_spatial_svd()
