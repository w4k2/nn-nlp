#! /bin/bash


python train_bert.py --dataset_name=liar --language=eng
python train_bert.py --dataset_name=liar --language=multi

# python train_bert.py --dataset_name=esp_fake --language=eng
python train_bert.py --dataset_name=esp_fake --language=multi