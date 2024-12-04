# MatsVD: Boosting Statement-Level Vulnerability Detection via Dependency-Based Attention

## Introduction
This is the official implementation of [MatsVD](https://dl.acm.org/doi/10.1145/3671016.3674807). MatsVD transform source code into PDGs, and then selectively mask partial attention in the Transformer based on the connectivity of the nodes in the PDGs.

## Dataset
https://zenodo.org/doi/10.5281/zenodo.11506329

## Requirements
joern=2.0.156
torch=1.13.1+cu116
transformers=4.27.3

## Run MatsVD
To retrain the model, run the following script(Training + Inference):
```shell
./train.sh
```
To reproduce the result, run the following script(Only Inference):
```shell
./test.sh
```



