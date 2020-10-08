import matplotlib.pyplot as plt
# import numpy as np
import torch


def get_ids(img_paths: list, dataset: str) -> tuple:
    camera_ids = []
    labels = []
    frames = []

    if dataset == 'market':
        dict_cam_seq_max = {
            11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346,
            17: 0, 18: 0, 21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0,
            26: 0, 27: 0, 28: 0, 31: 161708, 32: 161769, 33: 104469, 34: 0,
            35: 0, 36: 0, 37: 0, 38: 0, 41: 72107, 42: 72373, 43: 74810,
            44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0, 51: 161095,
            52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
            61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0,
            67: 0, 68: 0}

    for i, path in enumerate(img_paths):
        # File excluding the extension (.jpg)
        filename = path.split('\\')[-1][:-4]

        if dataset == 'market':
            label, camera_seq, frame, _ = filename.split('_')
        else:
            label, camera_seq, frame = filename.split('_')
            frame = frame[1:]

        camera_id = int(camera_seq[1])
        frame = int(frame)

        if dataset == 'market':
            seq = int(camera_seq[3])
            re = 0
            for j in range(1, seq):
                re = re + dict_cam_seq_max[int(str(camera_id) + str(j))]
            frame += re

        labels.append(label)
        camera_ids.append(int(camera_id))
        frames.append(frame)
    # (list, list, list)
    return camera_ids, labels, frames


def fliplr(imgs: torch.Tensor, device=None) -> torch.Tensor:
    """
    Horizontal Flip
    """
    assert len(imgs.shape) == 4, f'Expected imgs of 4 dimension but got \
                                    {len(imgs.shape)}'
    inv_idx = torch.arange(imgs.size(3)-1, -1, -1,
                           device=device).long()  # N x C x H x W
    img_flip = imgs.index_select(3, inv_idx)
    return img_flip


def l2_norm_standardize(features) -> torch.Tensor:
    """
    This function standarizes the features by their L2-Norm.
    """

    # L2-Norm Sum of Squares of the output all_features
    fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
    # Divide by L2-Norm (equivalent std.) -> Standardization
    standardized_features = features.div(fnorm.expand_as(features))
    standardized_features = standardized_features.view(
        standardized_features.size(0), -1)

    return standardized_features


def standardize(*args) -> list:
    """
    Standardize the input features

    :return: Standardized features
    :rtype: List[torch.Tensors]
    """
    new_args = []
    for arg in args:
        arg = arg.T / (torch.sum(arg**2, dim=1)**0.5)
        arg = arg.T
        new_args.append(arg)

    return new_args


def plot_distributions(st_distribution):

    nrows, ncols = st_distribution.shape[0], st_distribution.shape[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 25))

    for i, ax_i in enumerate(axes):
        for j, ax_j in enumerate(ax_i):
            ax_j.plot(st_distribution[i][j])

    return fig
