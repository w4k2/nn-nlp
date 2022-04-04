#! /bin/bash


python test_ensemble.py --dataset_name=esp_fake --attribute=text --mode=3M
python test_ensemble.py --dataset_name=esp_fake --attribute=title --mode=3M
python test_ensemble.py --dataset_name=bs_detector --attribute=text --mode=3M
python test_ensemble.py --dataset_name=bs_detector --attribute=title --mode=3M
python test_ensemble.py --dataset_name=mixed --attribute=text --mode=3M
python test_ensemble.py --dataset_name=mixed --attribute=title --mode=3M

#python test_ensemble.py --dataset_name=esp_fake --attribute=text --mode=4M
#python test_ensemble.py --dataset_name=esp_fake --attribute=title --mode=4M
#python test_ensemble.py --dataset_name=bs_detector --attribute=text --mode=4M
#python test_ensemble.py --dataset_name=bs_detector --attribute=title --mode=4M
#python test_ensemble.py --dataset_name=mixed --attribute=text --mode=4M
#python test_ensemble.py --dataset_name=mixed --attribute=title --mode=4M

#python test_ensemble.py --dataset_name=esp_fake --attribute=text --mode=12M
#python test_ensemble.py --dataset_name=esp_fake --attribute=title --mode=12M
#python test_ensemble.py --dataset_name=bs_detector --attribute=text --mode=12M
#python test_ensemble.py --dataset_name=bs_detector --attribute=title --mode=12M
#python test_ensemble.py --dataset_name=mixed --attribute=text --mode=12M
#python test_ensemble.py --dataset_name=mixed --attribute=title --mode=12M
