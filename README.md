# Multi-Camera Person Re-Identification

This repository is inspired by the paper [Spatial-Temporal Reidentification (ST-ReID)](https://arxiv.org/pdf/1812.03282.pdf)[1]. The state-of-the-art for Person Re-identification tasks.

This repository has been trained & tested on [DukeMTMTC-reID](https://megapixels.cc/duke_mtmc/) and [Market-1501 datasets](https://www.kaggle.com/pengcw1/market-1501). The model can be easily trained on any new datasets.

Below are the metrics on the various datasets.

| Model                | Size | Dataset | mAP  | CMC: Top1 | CMC: Top5 |
| -------------------- | ---- | ------- | ---- | --------- | --------- |
| `resnet50-PCB+rerank`|      | Market  | 95.5 |   98.0    |   98.9    |
| `resnet50-PCB+rerank`|      | Duke    | 92.7 |   94.5    |   96.8    |
| `resnet50-ft_dense`  |      | Market  |      |           |           |

## Dependencies

This project requires `pytorch>=1.3`, `pytorch-lightning=0.9.0`, `torchvision=0.7.0`, `pathlib2`, `joblib` and other common packages like `numpy`, `matplotlib` and `csv`

## Getting Started

1. Clone this repo & cd into it: `git clone https://github.com/SurajDonthi/MTMCT-Person-Re-Identification; cd MTMCT-Person-Re-Identification`
2. Download the datasets. (By default you can download & unzip them to `data/raw/` directory)

You can get started by training this model. Trained models will be available soon!

#### Train with your own dataset

```sh
python mtmct_reid/main.py --data_dir path/to/dataset/ --save_distribution path/to/dataset/st_distribution.pkl --gpus 1 --max_epochs 60
```

For a detailed list of arguments you can pass, refer to `hparams.csv`.

References:

[1] - [Spatial-Temporal Reidentification(ST-ReID)](https://arxiv.org/pdf/1812.03282.pdf)

[2] - [Beyond Parts Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/pdf/1711.09349)
