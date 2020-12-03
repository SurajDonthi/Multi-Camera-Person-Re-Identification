import math
import numpy as np
import torch

from mtmct_reid.utils import standardize

# Finding the best matches during testing:
# Step 1: From a given dataset, compute it's Spatial-Temporal Distribution. ✔✔
#         Requires: cam_ids, targets(labels), frames, MODEL is not required!
# Step 2: Compute it's Gaussian smoothed ST-Distribution. ✔✔
#         Requires: cam_ids, targets(labels), frames, MODEL is not required!
# Step 3: Compute the L2-Normed features ✔✔
#         Requires: Features - Performed once training is finished!
# Step 4: Compute the Joint Scores. ✔✔
#         Requires: Smoothed Distribution & L2-Normed Features, cam_ids, frames
# Step 5: Optionally perform Re-ranking of the Generated scores. ✔✔
#         Requires: Joint Scores
# Step 6: Compute mAP & CMC for each query. ✔✔
#         Requires: Reranked/Joint Scores, (query labels & cams),
#                   (gallery labels & cams)


def st_distribution(camera_ids, targets, frames, max_hist=3000,
                    eps=1e-7, interval=100.0):
    """To calculate Special Temporal Distribution of the number of targets on \
    the number of cameras."""

    # num_samples = len(camera_ids)
    num_cams = len(np.unique(camera_ids))
    num_classes = len(np.unique(targets))
    spatial_temporal_sum = np.zeros((num_classes, num_cams))
    spatial_temporal_count = np.zeros((num_classes, num_cams))

    # for i in range(num_samples):
    for cam, target, frame in zip(camera_ids, targets, frames):
        cam = cam - 1   # Camera ids are supposed to start from 0 & not 1.
        # For each identity (target) appearing on a cam,
        # store the sum of their time of appearance
        spatial_temporal_sum[target][cam] += frame  # frame -> time
        spatial_temporal_count[target][cam] += 1
    # spatial_temporal_avg: (num_classes ids, num_cams) -> center point
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)

    # Implementation of eq. (2) i.e, spatial-temporal histogram
    distribution = np.zeros((num_cams, num_cams, max_hist))
    for i in range(num_classes):
        for j in range(num_cams-1):  # j -> l in eq. (2)
            for k in range(j+1, num_cams):
                if spatial_temporal_count[i][j] == 0 or \
                        spatial_temporal_count[i][k] == 0:
                    continue
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                diff = np.abs(st_ij-st_ik)
                # binning the difference on a scale of 0-max_hist
                hist_ = int(diff/interval)
                # [big][small]
                distribution[j][k][hist_] += 1

    # Adjusted Percentage Calculation
    distribution = distribution / \
        (distribution.sum(axis=2, keepdims=True) + eps)
    # End of eq. (2) implementation

    # [to][from], to xxx camera, from xxx camera
    return distribution


def gaussian_func(x, mu, std):
    if (std == 0):
        raise ValueError('In gaussian, std shouldn\'t be equal to zero')
    temp1 = 1.0 / (std * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - mu, 2)) / (2 * math.pow(std, 2))
    return temp1 * np.exp(temp2)


def gaussian_kernel(length, mu=0, std=50):
    # vect = np.expand_dims(np.arange(length), 1)
    approximate_delta = 3 * std
    gaussian_vector = gaussian_func(np.arange(length), mu=mu, std=std)
    kernel = np.zeros((length, length))
    for i in range(length):
        k = 0
        for j in range(i, length):
            if k > approximate_delta:
                continue
            kernel[i][j] = gaussian_vector[j-i]
            k = k+1
    kernel += kernel.transpose()
    for i in range(length):
        kernel[i][i] /= 2
    return kernel

# Step 2


def smooth_st_distribution(camera_ids, targets, frames, num_cams,
                           max_hist=3000, eps=1e-7, interval=100.0):

    distribution = st_distribution(
        camera_ids, targets, frames, max_hist, eps, interval)

    matrix = gaussian_kernel(distribution.shape[-1])
    # distribution = np.tensordot(distribution, matrix, 1)
    # Apparently the below implementation is faster because of
    # smaller camera sizes!
    for i in range(num_cams):
        for j in range(num_cams):
            distribution[i][j][:] = np.dot(matrix, distribution[i][j][:])

    # Adjusted Percentage calculation
    distribution = distribution / \
        (distribution.sum(axis=2, keepdims=True) + eps)

    return distribution

# Step 3


def joint_scores(query_features, query_cams, query_frames,
                 gallery_features,
                 gallery_cams, gallery_frames, distribution, alpha=5,
                 interval=100):

    query_features, gallery_features = standardize(query_features,
                                                   gallery_features)

    scores = torch.Tensor()

    for feature, cam, frame in zip(query_features, query_cams, query_frames):
        # n: Number of Gallery instances
        # (n, 1228*6) * 2048*6  -> n
        # Visual Feature Stream
        feature_score = torch.matmul(gallery_features, feature)

        # Size: n
        gallery_frames = gallery_frames
        gallery_cams = gallery_cams

        diff = torch.abs(gallery_frames - frame)
        hist_ = (diff/interval).type(torch.int16)
        # Size: n
        st_score = distribution[cam.type(torch.int16).tolist(
        )-1][(gallery_cams - 1).type(torch.int16).tolist(), hist_.tolist()]
        st_score = torch.tensor(st_score)

        # score -> probabilities; This must be a formula from the paper!
        # Size: n
        score = 1/(1+torch.exp(-alpha*feature_score)) * \
            1/(1+2*torch.exp(-alpha*st_score))

        scores = torch.cat(
            [scores, torch.unsqueeze(score, dim=0)])   # all_scores

    # Size: k * n; k -> Num. of Query Instansces
    return scores

# Step 4


def AP_CMC(score, query_target, query_cam, gallery_targets,
           gallery_cams):
    """
    Compute CMC & mAP
    """
    # region I N D I C E S
    # Order by scores
    index = np.argsort(score)
    # index = index[::-1]

    # good index
    # All indices where there are same identities
    query_indices = np.argwhere(gallery_targets == query_target)
    # All indices where there are same cameras
    camera_indices = np.argwhere(gallery_cams == query_cam)

    # All indices in query_indices that are not in camera_indices
    good_index = np.setdiff1d(
        query_indices, camera_indices, assume_unique=True)
    junk_index1 = np.argwhere(gallery_targets == -1)
    # All indices in query_indices that are in camera_indices
    junk_index2 = np.intersect1d(query_indices, camera_indices)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    # endregion

    average_precision = 0

    # Cumulated Matching Charactesitics
    cmc = torch.IntTensor(len(index)).zero_()

    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return average_precision, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)

    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    if rows_good.size == 0:
        cmc[0] = -1
        return average_precision, cmc

    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0/ngood

        precision = (i+1)*1.0/(rows_good[i]+1)

        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        average_precision += d_recall*(old_precision + precision)/2

    return average_precision, cmc


def mAP(scores, query_targets, query_cams, gallery_targets, gallery_cams):

    num_queries = len(query_targets)

    CMC = torch.IntTensor(len(gallery_targets)).zero_()

    ap = 0.0
    # print(query_label)
    for score, query_target, query_cam in zip(scores, query_targets, query_cams):
        ap_tmp, CMC_tmp = AP_CMC(score, query_target, query_cam,
                                 gallery_targets, gallery_cams)

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    mean_ap = ap / num_queries

    CMC = CMC.float()
    CMC = CMC / num_queries  # average CMC

    return mean_ap, CMC
