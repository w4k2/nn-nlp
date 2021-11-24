import datasets
import numpy as np 

for dataset_name in ('esp_fake', 'bs_detector', 'mixed'):
    _, labels = datasets.load_dataset(dataset_name, attribute='text')
    if dataset_name == 'mixed':
        labels[np.argwhere(labels == 2).flatten()] = 0
        labels[np.argwhere(labels == 3).flatten()] = 1
    zeros = sum(labels == 0)
    ones = sum(labels == 1)
    print(f'{dataset_name}: class 0 count = {zeros}, class 1 count = {ones}, dataset count = {len(labels)}')

