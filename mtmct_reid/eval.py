from argparse import ArgumentParser

import joblib
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from .data import ReIDDataset
from .metrics import joint_scores, mAP
from .model import PCB
from .re_ranking import re_ranking
from .utils import fliplr, l2_norm_standardize


def generate_features(model, dataloader):
    all_features = torch.tensor()
    all_targets = torch.Tensor()
    all_cam_ids = torch.Tensor()
    all_frames = torch.Tensor()

    for batch in dataloader:
        x, targets, cam_ids, frames = batch

        features = model(x).detach().cpu()
        features += model(fliplr(x, x.device)).detach.cpu()

        all_features = torch.cat([all_features, features])
        all_targets = torch.cat([all_targets, targets])
        all_cam_ids = torch.cat([all_cam_ids, cam_ids])
        all_frames = torch.cat([all_frames, frames])

    all_features = l2_norm_standardize(all_features)

    return all_features, all_targets, all_cam_ids, all_frames


def main(args):
    # Load the data
    transform = transforms.Compose([
        transforms.Resize(size=(384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    query_data = ReIDDataset(data_dir=args.query_data_dir, transform=transform)
    query_dataloader = DataLoader(query_data, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    gal_data = ReIDDataset(data_dir=args.gallery_data_dir, transform=transform)
    gal_dataloader = DataLoader(gal_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

    # Load the model
    model = PCB(num_classes=len(query_data.num_classes))
    model.load_state_dict(args.model_path)
    model.eval()

    # Generate the feature vectors
    q_features, q_targets, q_cam_ids, q_frames = generate_features(
        model, query_dataloader)
    g_features, g_targets, g_cam_ids, g_frames = generate_features(
        model, gal_dataloader)

    # Load Spatial-Temporal Distribution
    st_distribution = joblib.load(args.st_distribution_path)

    scores = joint_scores(q_features, q_cam_ids, q_frames,
                          g_features, g_cam_ids, g_frames,
                          st_distribution)

    if args.re_rank:
        scores = re_ranking(scores)

    mean_ap, cmc = mAP(scores, q_targets,
                       q_cam_ids,
                       g_targets,
                       g_cam_ids)

    print_result = \
        f"""
    x-------------x TEST RESULT x-------------x
        mAP: {mean_ap},
        Rank-1: {cmc[0]},
        Rank-5: {cmc[4]},
        Rank-10: {cmc[9]}
    x-----------------------------------------x
    """
    print(print_result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str,
                        required=True, help="Path to the model")
    parser.add_argument('-qd', '--query_data_dir', type=str,
                        required=True, help="Path to load the query dataset")
    parser.add_argument('-d', '--gallery_data_dir', type=str,
                        required=True, help="Path to load the gallery dataset")
    parser.add_argument('-s', '--st_distribution_path', type=str,
                        required=True, help="Path to the sptial-temporal distribution of the data")
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=64, help="Batch size for evaluating the model")
    parser.add_argument('-n', '--num_workers', type=int,
                        required=False, default=4, help="Number of workers to use while loading the data")
    parser.add_argument('-r', '--re_rank', type=bool,
                        required=False, default=False, help="Choose whether to perform re-ranking of the metric scores")
    args = parser.parse_args()

    main(args)
