#! /bin/bash

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=mutual_info --mode=3M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=mutual_info --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=mutual_info --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=mutual_info --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=mutual_info --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=mutual_info --mode=3M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=mutual_info --mode=4M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=mutual_info --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=mutual_info --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=mutual_info --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=mutual_info --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=mutual_info --mode=4M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=mutual_info --mode=12M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=mutual_info --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=mutual_info --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=mutual_info --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=mutual_info --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=mutual_info --mode=12M


python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=anova --mode=3M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=anova --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=anova --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=anova --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=anova --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=anova --mode=3M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=anova --mode=4M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=anova --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=anova --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=anova --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=anova --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=anova --mode=4M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=anova --mode=12M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=anova --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=anova --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=anova --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=anova --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=anova --mode=12M



python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=pca --mode=3M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=pca --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=pca --mode=3M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=pca --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=pca --mode=3M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=pca --mode=3M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=pca --mode=4M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=pca --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=pca --mode=4M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=pca --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=pca --mode=4M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=pca --mode=4M

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=pca --mode=12M
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=pca --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=pca --mode=12M
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=pca --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=pca --mode=12M
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=pca --mode=12M