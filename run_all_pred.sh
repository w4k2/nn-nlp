#! /bin/bash

# python train_beto.py --dataset_name=esp_fake --attribute=text 
python train_bert.py --dataset_name=esp_fake --language=multi --attribute=text
# python train_beto.py --dataset_name=esp_fake --attribute=title 
python train_bert.py --dataset_name=esp_fake --dataset_name=esp_fake --language=multi --attribute=title

python train_bert.py --dataset_name=bs_detector --language=eng --attribute=text
python train_bert.py --dataset_name=bs_detector --language=multi --attribute=text
python train_bert.py --dataset_name=bs_detector --language=eng --attribute=title
python train_bert.py --dataset_name=bs_detector --language=multi --attribute=title

python train_bert.py --dataset_name=mixed --language=eng --attribute=text
python train_bert.py --dataset_name=mixed --language=multi --attribute=text
python train_bert.py --dataset_name=mixed --language=eng --attribute=title
python train_bert.py --dataset_name=mixed --language=multi --attribute=title
# python train_beto.py --dataset_name=mixed --attribute=text 
# python train_beto.py --dataset_name=mixed --attribute=title 
