#! /bin/bash


python test_ensemble.py --dataset_name=esp_fake --attribute=text
python test_ensemble.py --dataset_name=esp_fake --attribute=title
# python test_ensemble.py --dataset_name=bs_detector --attribute=text
# python test_ensemble.py --dataset_name=bs_detector --attribute=title
# python test_ensemble.py --dataset_name=mixed --attribute=text
# python test_ensemble.py --dataset_name=mixed --attribute=title
