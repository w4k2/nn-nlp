#! /bin/bash

# python predict_beto.py --dataset_name=esp_fake --train_dataset_name=esp_fake --attribute=text 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=bs_detector --language=eng --attribute=text 
# python predict_beto.py --dataset_name=esp_fake --train_dataset_name=mixed --attribute=text 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=mixed --language=eng --attribute=text 

python predict_bert.py --dataset_name=esp_fake --train_dataset_name=esp_fake --language=multi --attribute=text 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=bs_detector --language=multi --attribute=text 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=mixed --language=multi --attribute=text 

# python predict_beto.py --dataset_name=esp_fake --train_dataset_name=esp_fake --attribute=title 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=bs_detector --language=eng --attribute=title 
# python predict_beto.py --dataset_name=esp_fake --train_dataset_name=mixed --attribute=title 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=mixed --language=eng --attribute=title 

python predict_bert.py --dataset_name=esp_fake --train_dataset_name=esp_fake --language=multi --attribute=title 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=bs_detector --language=multi --attribute=title 
python predict_bert.py --dataset_name=esp_fake --train_dataset_name=mixed --language=multi --attribute=title 


python predict_bert.py --dataset_name=bs_detector --train_dataset_name=bs_detector --language=eng --attribute=text 
# python predict_beto.py --dataset_name=bs_detector --train_dataset_name=esp_fake --attribute=text 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=mixed --language=eng --attribute=text 
# python predict_beto.py --dataset_name=bs_detector --train_dataset_name=mixed --attribute=text 

python predict_bert.py --dataset_name=bs_detector --train_dataset_name=bs_detector --language=multi --attribute=text 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=esp_fake --language=multi --attribute=text 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=mixed --language=multi --attribute=text 

python predict_bert.py --dataset_name=bs_detector --train_dataset_name=bs_detector --language=eng --attribute=title 
# python predict_beto.py --dataset_name=bs_detector --train_dataset_name=esp_fake --attribute=title 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=mixed --language=eng --attribute=title 
# python predict_beto.py --dataset_name=bs_detector --train_dataset_name=mixed --attribute=title 

python predict_bert.py --dataset_name=bs_detector --train_dataset_name=bs_detector --language=multi --attribute=title 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=esp_fake --language=multi --attribute=title 
python predict_bert.py --dataset_name=bs_detector --train_dataset_name=mixed --language=multi --attribute=title 


python predict_bert.py --dataset_name=mixed --train_dataset_name=mixed --language=eng --attribute=text 
# python predict_beto.py --dataset_name=mixed --train_dataset_name=mixed --attribute=text 
# python predict_beto.py --dataset_name=mixed --train_dataset_name=esp_fake --attribute=text 
python predict_bert.py --dataset_name=mixed --train_dataset_name=bs_detector --language=eng --attribute=text 

python predict_bert.py --dataset_name=mixed --train_dataset_name=mixed --language=multi --attribute=text 
python predict_bert.py --dataset_name=mixed --train_dataset_name=esp_fake --language=multi --attribute=text 
python predict_bert.py --dataset_name=mixed --train_dataset_name=bs_detector --language=multi --attribute=text 

python predict_bert.py --dataset_name=mixed --train_dataset_name=mixed --language=eng --attribute=title 
# python predict_beto.py --dataset_name=mixed --train_dataset_name=mixed --attribute=title 
# python predict_beto.py --dataset_name=mixed --train_dataset_name=esp_fake --attribute=title 
python predict_bert.py --dataset_name=mixed --train_dataset_name=bs_detector --language=eng --attribute=title 

python predict_bert.py --dataset_name=mixed --train_dataset_name=mixed --language=multi --attribute=title 
python predict_bert.py --dataset_name=mixed --train_dataset_name=esp_fake --language=multi --attribute=title 
python predict_bert.py --dataset_name=mixed --train_dataset_name=bs_detector --language=multi --attribute=title 