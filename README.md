# Benchmarking the uncertainty estimation performance of ImageNet classifiers
<p align="center">
  <img src="https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance/blob/main/Risk-Coverage_curve_comparison.png">
</p>
This repository is the official implementation of "What can we learn from the selective prediction and uncertainty estimation performance of 523 ImageNet classifiers?", published in ICLR 2023:
https://openreview.net/pdf?id=p66AzKi6Xim


## Usage
Use benchmark_models.py to automatically benchmark the uncertainty estimation performance of any ImageNet-1k classifier from the excellent 'timm' (PyTorch Image Models) repo:
https://github.com/rwightman/pytorch-image-models

This will print and create a csv file with the following uncertainty metrics:

1. SAC (Selective Accuracy Constraint): a selective classification metric (first introduced by our paper) that measures the maximal coverage a model is able to provide given a constraint on the required accuracy. For example, we discovered a ViT model can achieve an unprecedented 99% top-1 selective accuracy on ImageNet at 47% coverage (and 95% top-1 accuracy at 80%), i.e., if we allow the model to reject 53% of its least confident predictions, its accuracy on the predictions it did not reject would be 99%.

2. AUROC (Area Under the Receiver Operating Characteristic): a measure of ranking performance insensitive to accuracy. See Equation 1 in the paper for more information.

3. ECE (Expected Calibration Error): A metric evaluating calibration performance.

4. AURC (Area Under the Risk-Coverage curve ,Geifman et al., 2018). In essence, this metric is equal the mean selective risk over all possible coverage values.

All models are also evaluated after using 'Temperature Scaling', which tends to improve the performance over all metrics.


To use it, simply run:

```
python benchmark_models.py --data_dir #path to the ImageNet validation folder# -b #batchsize# --models #models to evaluate#
```
For example, to evaluate two models, resnet18 and resnet34:
```example
python benchmark_models.py --data_dir #path to the ImageNet validation folder# -b 128 --models resnet18 resnet34
```
