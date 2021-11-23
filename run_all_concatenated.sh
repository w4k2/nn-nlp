#! /bin/bash

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=mutual_info
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=mutual_info
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=mutual_info
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=mutual_info
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=mutual_info
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=mutual_info

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=anova
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=anova
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=anova
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=anova
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=anova
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=anova

python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=text --feature_selection=pca
python test_concatenated_extractions.py --dataset_name=esp_fake --attribute=title --feature_selection=pca
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=text --feature_selection=pca
python test_concatenated_extractions.py --dataset_name=bs_detector --attribute=title --feature_selection=pca
python test_concatenated_extractions.py --dataset_name=mixed --attribute=text --feature_selection=pca
python test_concatenated_extractions.py --dataset_name=mixed --attribute=title --feature_selection=pca
