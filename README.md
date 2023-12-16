# Diffusion Language-Shapelets for Semi-supervised Time-series Classification

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

