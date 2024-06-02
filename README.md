# Diffusion Language-Shapelets for Semi-supervised Time-Series Classification

This is the training code for our paper "Diffusion Language-Shapelets for Semi-supervised Time-Series Classification" (AAAI-24).

## Abstract

Semi-supervised time-series classification could effectively alleviate the issue of lacking labeled data. However,
existing approaches usually ignore model interpretability, making it difficult for humans to understand the principles
behind the predictions of a model. Shapelets are a set of discriminative subsequences that show high interpretability in
time series classification tasks. Shapelet learning-based methods have demonstrated promising classification
performance. Unfortunately, without enough labeled data, the shapelets learned by existing methods are often poorly
discriminative, and even dissimilar to any subsequence of the original time series. To address this issue, we propose
the Diffusion Language-Shapelets model (DiffShape) for semi-supervised time series classification. In DiffShape, a
self-supervised diffusion learning mechanism is designed, which uses real subsequences as a condition. This helps to
increase the similarity between the learned shapelets and real subsequences by using a large amount of unlabeled data.
Furthermore, we introduce a contrastive language-shapelets learning strategy that improves the discriminability of the
learned shapelets by incorporating the natural language descriptions of the time series.
Experiments have been conducted on the UCR time series archive, and the results reveal that the proposed DiffShape
method achieves state-of-the-art performance and exhibits superior interpretability over baselines.

## Datasets

### UCR archive time series datasets

* [UCR time series archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip)

In accordance with the recommendation provided by the creators of the UCR archive and TS-TFC, we implemented a
restriction to maintain an average
of at least 30 samples per class within each dataset. This measure was taken to enhance the stability of the
classification test results. As a result,
we employed a total of 106 datasets from the initial pool of 128 UCR datasets for our experimental analysis.

Please refer to **page 13** of the [PDF](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/BriefingDocument2018.pdf) document for the password to access the zipped file of the UCR archive.

## Usage (Our Model)

To train a DiffShape model on a dataset for semi-supervised time series classification, run

```bash
python diffshape_ssc/main_diffshape.py --dataset [name of the dataset you want to train]  ...
```

For detailed options and examples, please refer to parser setup in ```diffshape_ssc/main_diffshape.py```

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{liu2024diffusion,
  title={Diffusion language-shapelets for semi-supervised time-series classification},
  author={Liu, Zhen and Pei, Wenbin and Lan, Disen and Ma, Qianli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14079--14087},
  year={2024}
}
```


