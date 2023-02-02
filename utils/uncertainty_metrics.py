import torch
import numpy as np

'''
NOTE: All following methods get samples_certainties, which is a tensor of size (N,2) of N samples, for each 
its certainty and whether or not it was a correct prediction. Position 0 in each sample is its confidence, and 
position 1 is its correctness (True \ 1 for correct samples and False \ 0 for incorrect ones).

If samples_certainties is sorted in a descending order, set sort=False to avoid sorting it again.

Example: samples_certainties[0][0] is the confidence score of the first sample.
samples_certainties[0][1] is the correctness (True \ False) of the first sample.
'''

def confidence_variance(samples_certainties):
    return torch.var(samples_certainties.transpose(0, 1)[0], unbiased=True).item()


def confidence_mean(samples_certainties):
    return torch.mean(samples_certainties.transpose(0, 1)[0]).item()


def confidence_median(samples_certainties):
    return torch.median(samples_certainties.transpose(0, 1)[0]).item()


def gini(samples_certainties):
    array = samples_certainties.transpose(0, 1)[0]
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if torch.amin(array) < 0:
        # Values cannot be negative:
        array -= torch.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = torch.sort(array)[0]
    # Index per array element:
    index = torch.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((torch.sum((2 * index - n  - 1) * array)) / (n * torch.sum(array))).item()


def AUROC(samples_certainties, sort=True):
    # Calculating AUROC in a similar way gamma correlation is calculated. The function can easily return both.
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    incorrect_after_i = np.zeros((total_samples))

    for i in range(total_samples - 1, -1, -1):
        if i == total_samples - 1:
            incorrect_after_i[i] = 0
        else:
            incorrect_after_i[i] = incorrect_after_i[i+1] + (1 - int(samples_certainties[i+1][1]))
            # Note: samples_certainties[i+1][1] is the correctness label for sample i+1

    n_d = 0  # amount of different pairs of ordering
    n_s = 0  # amount of pairs with same ordering
    incorrect_before_i = 0
    for i in range(total_samples):
        if i != 0:
            incorrect_before_i += (1 - int(samples_certainties[i-1][1]))
        if samples_certainties[i][1]:
            # if i is a correct prediction, i's ranking 'agrees' with all the incorrect that are to come
            n_s += incorrect_after_i[i]
            # i's ranking 'disagrees' with all incorrect predictions that preceed it (i.e., ranked more confident)
            n_d += incorrect_before_i
        else:
            # else i is an incorrect prediction, so i's ranking 'disagrees' with all the correct predictions after
            n_d += (total_samples - i - 1) - incorrect_after_i[i]  # (total_samples - i - 1) = all samples after i
            # i's ranking 'agrees' with all correct predictions that preceed it (i.e., ranked more confident)
            n_s += i - incorrect_before_i

    return n_s / (n_s + n_d)


def AURC_calc(samples_certainties, sort=True):
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    AURC = 0
    incorrect_num = 0
    for i in range(total_samples):
        incorrect_num += 1 - int(samples_certainties[i][1])
        AURC += (incorrect_num / (i + 1))
    AURC = AURC / total_samples
    return AURC


def EAURC_calc(AURC, accuracy):
    risk = 1 - (accuracy / 100)
    # From https://arxiv.org/abs/1805.08206 :
    optimal_AURC = risk + (1-risk)*np.log(1-risk)
    return AURC - optimal_AURC


def ECE_calc(samples_certainties, num_bins=15, bin_boundaries_scheme=None):
    indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=False)  # Notice the reverse sorting
    samples_certainties = samples_certainties[indices_sorting_by_confidence]
    samples_certainties = samples_certainties.transpose(0, 1)
    if bin_boundaries_scheme is None:  # Normal ECE, make it evenly spaced
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    else:  # Use scheme to determine bin limits
        bin_boundaries = bin_boundaries_scheme(samples_certainties, num_bins=num_bins)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    max_calibration_error = -1
    bins_accumulated_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_indices = torch.logical_and(samples_certainties[0] <= bin_upper, samples_certainties[0] > bin_lower)
        if bin_indices.sum() == 0:
            continue  # This is an empty bin
        bin_confidences = samples_certainties[0][bin_indices]
        bin_accuracies = samples_certainties[1][bin_indices]
        bin_avg_confidence = bin_confidences.mean()
        bin_avg_accuracy = bin_accuracies.mean()
        bin_error = torch.abs(bin_avg_confidence - bin_avg_accuracy)
        if bin_error > max_calibration_error:
            max_calibration_error = bin_error
        bins_accumulated_error += bin_error * bin_confidences.shape[0]

    expected_calibration_error = bins_accumulated_error / samples_certainties.shape[1]
    return expected_calibration_error, max_calibration_error


def calc_adaptive_bin_size(samples_certainties, num_bins=10):
    certainties = samples_certainties[0]
    N = certainties.shape[0]
    step_size = int(N / (num_bins - 1))
    bin_boundaries = [certainties[i].item() for i in range(0, certainties.shape[0], step_size)]
    bin_boundaries[0] = 0
    bin_boundaries[-1] = certainties[-1]
    return torch.tensor(bin_boundaries)


def coverage_for_desired_accuracy(samples_certainties, accuracy=0.95, sort=False, start_index=200):
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    correctness = samples_certainties.transpose(0, 1)[1]
    cumsum_correctness = torch.cumsum(correctness, dim=0)
    num_samples = cumsum_correctness.shape[0] + 1
    cummean_correctness = cumsum_correctness / torch.arange(1, num_samples)
    # note: We use numpy's argmax since it returns the first occurrence. torch doesn't!
    coverage_for_accuracy = np.argmax(cummean_correctness < accuracy).item()

    # To ignore statistical noise, start measuring at an index greater than 0
    coverage_for_accuracy_nonstrict = np.argmax(cummean_correctness[start_index:] < accuracy).item() + start_index
    if coverage_for_accuracy_nonstrict > start_index:
        # If they were the same, even the first non-noisy measurement didn't satisfy the risk, so its coverage is undue,
        # use the original index. Otherwise, use the non-strict to diffuse noisiness.
        coverage_for_accuracy = coverage_for_accuracy_nonstrict

    coverage_for_accuracy = coverage_for_accuracy / num_samples
    return coverage_for_accuracy