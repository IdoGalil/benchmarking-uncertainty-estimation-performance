import sys
import torch

from utils import log_utils
import numpy as np
import sklearn.model_selection
import csv
import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import argparse
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader, Subset
from utils.uncertainty_metrics import *
from utils.temperature_scaling import ModelWithTemperature
import timm
from timm.data import resolve_data_config, create_transform


config_parser = parser = argparse.ArgumentParser(description='Benchmarking uncertainty performance config', add_help=False)
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')

parser.add_argument('--models', nargs='+', type=str, help='a list of model names available on the timm repo')


def create_model_and_transforms(model_name):
    model = timm.create_model(model_name, pretrained=True).eval().cuda()
    # Creating the model specific data transformation
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def metrics_calculations(samples_certainties, num_bins_ece=15):
    # Note: we assume here the certainty scores in samples_certainties are probabilities.
    results = {}
    results['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    results['AUROC'] = AUROC(samples_certainties)
    results['Coverage_for_Accuracy_99'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.99, start_index=200)
    ece, mce = ECE_calc(samples_certainties, num_bins=num_bins_ece)
    results[f'ECE_{num_bins_ece}'] = ece
    results['AURC'] = AURC_calc(samples_certainties)
    return results


def extract_model_info(model, dataloader, pbar_name='Extracting data for model'):
    num_batches = len(dataloader.batch_sampler)
    total_correct = 0
    total_samples = 0
    # samples_certainties holds a tensor of size (N, 2) of N samples, for each its certainty and whether it was a
    # correct prediction.
    # Position 0 is the confidences and 1 is the correctness
    samples_certainties = torch.empty((0, 2))
    timer_start = timer()
    with torch.no_grad():
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=sys.stdout) as pbar:
            dl_iter = iter(dataloader)
            for batch_idx in range(num_batches):
                x, y = next(dl_iter)
                x = x.cuda()
                y = y.cuda()

                y_scores = model.forward(x)
                y_scores = torch.softmax(y_scores, dim=1)
                y_pred = torch.max(y_scores, dim=1)
                certainties = y_pred[0]
                correct = y_pred[1] == y
                total_correct += correct.sum().item()
                total_samples += x.shape[0]
                accuracy = (total_correct / total_samples) * 100

                samples_info = torch.stack((certainties.cpu(), correct.cpu()))
                samples_certainties = torch.vstack((samples_certainties, samples_info.transpose(0, 1)))

                pbar.set_description(f'{pbar_name}. accuracy:{accuracy:.3f}% (Elapsed time:{timer() - timer_start:.3f} sec)')
                pbar.update()

            indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
            samples_certainties = samples_certainties[indices_sorting_by_confidence]
            results = metrics_calculations(samples_certainties)
            return results


def extract_temperature_scaled_metrics(model, transform, valid_size=5000, model_name=None):
    assert valid_size > 0
    dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
    test_indices, valid_indices = sklearn.model_selection.train_test_split(np.arange(len(dataset)),
                                                                           train_size=len(dataset) - valid_size,
                                                                           stratify=dataset.targets)
    valid_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))
    test_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(test_indices))
    model = ModelWithTemperature(model)
    print(f'Performing temperature scaling')
    model.set_temperature(valid_loader)
    if model_name:
        pbar_name = f'Extracting data for {model_name} after temperature scaling'
    else:
        pbar_name = f'Extracting data for model after temperature scaling'
    model_results_TS = extract_model_info(model, test_loader, pbar_name=pbar_name)
    # To make sure all temperature scaled metrics have a distinct name, add _TS at its end
    model_results_TS = {f'{key}_TS': value for key, value in model_results_TS.items()}
    return model_results_TS


def models_comparison(models_names: list, file_name='./results.csv'):
    headers = ['Architecture', 'Accuracy', 'AUROC', 'AUROC_TS', 'ECE_15', 'ECE_15_TS', 'Coverage_for_Accuracy_99',
               'Coverage_for_Accuracy_99_TS', 'AURC', 'AURC_TS']
    logger = log_utils.Logger(file_name=file_name, headers=headers, overwrite=False)
    for model_name in models_names:
        model, transform = create_model_and_transforms(model_name)
        dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model_results = extract_model_info(model, dataloader, pbar_name=f'Extracting data for {model_name}')
        # Temperature scaling
        temperature_scaled_model_results = extract_temperature_scaled_metrics(model, transform, model_name=model_name)
        model_results = {**model_results, **temperature_scaled_model_results}
        model_results['Architecture'] = model_name
        # Log results
        logger.log(model_results)
    x = 1
    # for key, value in zip(model_results.keys(), model_results.values()):



if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    args = parser.parse_args()
    models_comparison(args.models, file_name='./results.csv')