# Multi-Camera Person Re-Identification

This repository is inspired by the paper [Spatial-Temporal Reidentification (ST-ReID)](https://arxiv.org/abs/1812.03282v1)[1]. The state-of-the-art for Person Re-identification tasks. This repository offers a flexible, and easy to understand clean implementation of the model architecture, training and evaluation.

This repository has been trained & tested on [DukeMTMTC-reID](https://megapixels.cc/duke_mtmc/) and [Market-1501 datasets](https://www.kaggle.com/pengcw1/market-1501). The model can be easily trained on any new datasets with a few tweaks to parse the files!

Below are the metrics on the various datasets.

| Model                 | Size | Dataset | mAP  | CMC: Top1 | CMC: Top5 |
| --------------------- | ---- | ------- | ---- | --------- | --------- |
| `resnet50-PCB+rerank` |      | Market  | 95.5 | 98.0      | 98.9      |
| `resnet50-PCB+rerank` |      | Duke    | 92.7 | 94.5      | 96.8      |
<!---| `resnet50-ft_dense`   |      | Market  |      |           |           |--->


## Model Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/SurajDonthi/Clean-ST-ReID-Multi-Target-Multi-Camera-Tracking/master/imgs/model-architecture.png" width=800 alt="MTMCT ST-ReID Model Architecture">
  <br>
  <i>Source: <a href="https://arxiv.org/pdf/1812.03282.pdf">Spatial-Temporal Reidentification(ST-ReID)</a></i>
</p>

1. A pre-trained ResNet-50 backbone model with layers up until Adaptive Average Pooling(excluded) is used

**During Training**

> 1. The last Convolutional layer is broken into 6 (Final output size: 6 x 1) parts and separately used for predicting the person label.
> 2. The total loss of the 6 part predictions are calculated for backpropagation & weights update.

**During Testing/Evaluation/Deployment**

> 1. Only the visual feature stream up until Adaptive Average Pooling is used.
> 2. The feature vector of the query image is compared against all the feature vectors of the gallery images using a simple dot product & normalization.
> 3. The Spatio-Temporal distribution is used to calculate their spatio-temporal scores.
> 4. The joint score is then calculated from the feature score and the spatio-temporal scores.
> 5. The Cumulated Matching Score is used to find the best matching for person from the gallet set.

## Getting Started
Run the below commands in the shell.

1. Clone this repo, cd into it & install setup.py: 
```sh
git clone https://github.com/SurajDonthi/MTMCT-Person-Re-Identification

cd MTMCT-Person-Re-Identification

pip install -r requirements.txt
```
2. Download the datasets. (By default you can download & unzip them to `data/raw/` directory)

You can get started by training this model. Trained models will be available soon!

*Dependencies*

This project requires `pytorch>=1.5.0`, `torchvision>=0.6.0`, `pytorch-lightning=0.9.0`, `tensorboard=2.2.0`, `pathlib2`, `joblib` and other common packages like `numpy`, `matplotlib` and `csv`.

NOTE: This project uses [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html) which is a high-level interface to abstract away repeating Pytorch code. It helps achieve clean, & easy to maintain code with hardly any learning curve!

#### Train with your own dataset

Run the below command in the shell.

```sh
mtmct_reid --data_dir path/to/dataset/ --save_distribution path/to/dataset/st_distribution.pkl --gpus 1 --max_epochs 60
```

For a detailed list of arguments you can pass, refer to [`hparams.csv`](https://github.com/SurajDonthi/MTMCT-Person-Re-Identification/blob/master/hparams.csv)

#### Monitor the training on Tensorboard

Log files are created to track the training in a new folder `lightning_logs`. To monitor the training, run the below command in the shell

```sh
tensorboard --logdir lightning_logs/
```

## Metrics

The evaluation metrics used are mAP (mean Average Precision) & CMC (Cumulated Matching Characteristics)

Finding the best matches during testing:

 Step 1: From a given dataset, compute it's Spatial-Temporal Distribution.
 
 >> Requires: cam_ids, targets(labels), frames, MODEL is not required!
         
 Step 2: Compute it's Gaussian smoothed ST-Distribution.
 
>> Requires: cam_ids, targets(labels), frames, MODEL is not required!
         
 Step 3: Compute the L2-Normed features that is generated from the model.
 
>> Requires: Features - Performed once training is finished!
 
 Step 4: Compute the Joint Scores.
 
>> Requires: Smoothed Distribution & L2-Normed Features, cam_ids, frames
 
 Step 5: Optionally perform Re-ranking of the Generated scores.

>> Requires: Joint Scores
 
 Step 6: Compute mAP & CMC (Cumulated Matching Characteristics; for Rank-1,Rank-5, Rank-10) for each query.
 
>> Requires: Reranked/Joint Scores, (query labels & cams),
                   (gallery labels & cams)


**References:**

[1] - [Spatial-Temporal Reidentification(ST-ReID)](https://arxiv.org/pdf/1812.03282.pdf)

[2] - [Beyond Parts Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/pdf/1711.09349)

**Related repos:**

The model logic is mainly based on this [repository](https://github.com/Wanggcong/Spatial-Temporal-Re-identification).
