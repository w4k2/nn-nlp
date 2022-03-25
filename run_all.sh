#! /bin/bash


# python precompute_features.py --dataset_name=esp_fake --attribute=text --extraction_method=lda
# python precompute_features.py --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
# python precompute_features.py --dataset_name=esp_fake --attribute=title --extraction_method=lda
# python precompute_features.py --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf

# python classify.py --train_dataset_name=esp_fake --dataset_name=esp_fake --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=bs_detector --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=mixed --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=esp_fake --dataset_name=bs_detector --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=esp_fake --dataset_name=mixed --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=esp_fake --dataset_name=esp_fake --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=bs_detector --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=mixed --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=esp_fake --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=esp_fake --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=esp_fake --dataset_name=mixed --attribute=title --extraction_method=tf_idf

# BETO is BERT pretrained on spanish corpus
# NOTE: train_beto requires different conda env than train_bert 
python train_beto.py --dataset_name=esp_fake --attribute=text 
# python train_bert.py --dataset_name=esp_fake --language=multi --attribute=text
python train_beto.py --dataset_name=esp_fake --attribute=title 
# python train_bert.py --dataset_name=esp_fake --dataset_name=esp_fake --language=multi --attribute=title


# python precompute_features.py --dataset_name=bs_detector --attribute=text --extraction_method=lda
# python precompute_features.py --dataset_name=bs_detector --attribute=text --extraction_method=tf_idf
# python precompute_features.py --dataset_name=bs_detector --attribute=title --extraction_method=lda
# python precompute_features.py --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf

# python classify.py --train_dataset_name=bs_detector --dataset_name=esp_fake --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=bs_detector --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=mixed --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=bs_detector --dataset_name=bs_detector --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=bs_detector --dataset_name=mixed --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=bs_detector --dataset_name=esp_fake --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=bs_detector --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=mixed --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=bs_detector --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=bs_detector --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=bs_detector --dataset_name=mixed --attribute=title --extraction_method=tf_idf

# python train_bert.py --dataset_name=bs_detector --language=eng --attribute=text
# python train_bert.py --dataset_name=bs_detector --language=multi --attribute=text
# python train_bert.py --dataset_name=bs_detector --language=eng --attribute=title
# python train_bert.py --dataset_name=bs_detector --language=multi --attribute=title


# python precompute_features.py --dataset_name=mixed --attribute=text --extraction_method=lda
# python precompute_features.py --dataset_name=mixed --attribute=text --extraction_method=tf_idf
# python precompute_features.py --dataset_name=mixed --attribute=title --extraction_method=lda
# python precompute_features.py --dataset_name=mixed --attribute=title --extraction_method=tf_idf

# python classify.py --train_dataset_name=mixed --dataset_name=esp_fake --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=bs_detector --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=mixed --attribute=text --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=esp_fake --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=mixed --dataset_name=bs_detector --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=mixed --dataset_name=mixed --attribute=text --extraction_method=tf_idf
# python classify.py --train_dataset_name=mixed --dataset_name=esp_fake --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=bs_detector --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=mixed --attribute=title --extraction_method=lda
# python classify.py --train_dataset_name=mixed --dataset_name=esp_fake --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=mixed --dataset_name=bs_detector --attribute=title --extraction_method=tf_idf
# python classify.py --train_dataset_name=mixed --dataset_name=mixed --attribute=title --extraction_method=tf_idf

# python train_bert.py --dataset_name=mixed --language=eng --attribute=text
# python train_bert.py --dataset_name=mixed --language=multi --attribute=text
# python train_bert.py --dataset_name=mixed --language=eng --attribute=title
# python train_bert.py --dataset_name=mixed --language=multi --attribute=title
python train_beto.py --dataset_name=mixed --attribute=text 
python train_beto.py --dataset_name=mixed --attribute=title 
