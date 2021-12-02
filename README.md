# Fake news ll-nlp

Folowing repository contains code for paper *Set title here*.

## Repo organisation

Experiments have several stages:

* training of extraction models, and storing weights or features (scripts: precompute_features.py, train_bert.py, train_beto.py)
* collecting model predictions (scripts: classify.py, predict_bert.py, predict_beto.py)
* creating and evaluation of ensemble algorithms (scripts: test_ensemble.py, test_concatenated_extractions.py)
* statiscial evaluation of results (scripts: statistical_tests.py)

There are also several other minor scripts, but they are not of primary importance. 
After first and second step complete results are stored as numpy array (.npy file extensions) in folders extracted_features and predictions.
Evaluation results with accuracy for each fold are stored in predictions (also in .npy files).

### Datasets

Before running any experimetns download kaggle fake news datasets https://www.kaggle.com/c/fake-news/ into datasets folder.

```
wget -o data.csv https://www.kaggle.com/c/fake-news/data?select=train.csv
mv data.csv datasets/bs_detector/
```


### Environments

There are 5 types of base models, namely tf-idf, lda, bert multilanguage, bert english, beto. 
Lda utilize gensim, bert are used from tensorflow, beto is avaliable in huggingface (with pytorch).
For this reason there are 3 conda environments defined in .yml files in this repo: 
* fake-news, file: environemnt.yml - gensim
* fake-news-gpu, file: environemnt_gpu.yml - tensorflow
* fake-news-gpu-pytorch, file: environemnt_gpu_pytorch.yml - pytorch
Both environemnt_gpu.yml and environemnt_gpu_pytorch.yml assume the gpu is avaliable with CUDA version >= 11.3 installed.

## Results reproduction

To run all experiments run following bash scripts (with following conda env sourced):

* ```run_all.sh``` (fake-news, fake-news-gpu, fake-news-gpu-pytorch)
* ```run_all_pred.sh``` (fake-news, fake-news-gpu, fake-news-gpu-pytorch)
* ```run_all_ensembles.sh``` (fake-news)
* ```run_all_concatenated.sh``` (fake-news)

In case of of multiple envrionments please make sure, that you run only scripts that proper for currently source conda env.