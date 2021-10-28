#! /bin/bash

# python precompute_features.py --dataset_name=esp_fake --attribute=text --extraction_method=lda
# python precompute_features.py --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
# python precompute_features.py --dataset_name=esp_fake --attribute=title --extraction_method=lda
# python precompute_features.py --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf

# python precompute_features.py --dataset_name=bs_detector --attribute=title --extraction_method=lda
# python precompute_features.py --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf

python classify.py --dataset_name=esp_fake --attribute=text --extraction_method=lda
python classify.py --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
python classify.py --dataset_name=esp_fake --attribute=title --extraction_method=lda
python classify.py --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf

python classify.py --dataset_name=bs_detector --attribute=title --extraction_method=lda
python classify.py --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf

# python train_bert.py --dataset_name=bs_detector --language=eng
# python train_bert.py --dataset_name=bs_detector --language=multi

# python train_beto.py ## beto is spanish bert --dataset_name=esp_fake --language=spanish
# python train_bert.py --dataset_name=esp_fake --language=multi