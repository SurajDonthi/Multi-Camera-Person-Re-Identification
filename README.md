# Multi-Camera Person Re-Identification

This repository is inspired by the paper [Spatial-Temporal Reidentification (ST-ReID)](https://arxiv.org/pdf/1812.03282.pdf)[1]. The state-of-the-art for Person Re-identification tasks.

This repository has been trained & tested on DukeMTMTC-reID and Market-1501 datasets. The model can be easily trained on any new datasets.

Below are the metrics on the various datasets.

| Model               | Size | mAP  | CMC: Top1 | CMC: Top5 |
| ------------------- | ---- | ---- | --------- | --------- |
| `resnet50-PCB`      |      | 95.1 | 89.8      | 93.2      |
| `resnet50-ft_dense` |      |      |           |           |

## Training with your own dataset

```sh
python -W ignore::UserWarning::0 mtmct_reid/main.py --data_dir path_to_data/ --save_distribution path_to_data/st_distribution.pkl --gpus 1 --max_epochs 60 --early_stop_callback 'Train' --precision 16
```

For a detailed list of arguments you can pass, refer to `hparams.csv`.

References:

[1] - [Spatial-Temporal Reidentification(ST-ReID)](https://arxiv.org/pdf/1812.03282.pdf)

[2] - [Beyond Parts Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/pdf/1711.09349)
