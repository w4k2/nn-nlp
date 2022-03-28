import argparse
import torch
import torch.nn as nn
import datasets
import os
import pathlib
import numpy as np
import sklearn.model_selection
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, AutoModelForSequenceClassification


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    args = parse_args()
    print(args)
    dataset_docs, dataset_labels = datasets.load_dataset(args.dataset_name)

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(dataset_docs, dataset_labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        torch.cuda.empty_cache()

        dataset_docs_train = [dataset_docs[i] for i in train_idx]
        dataset_docs_test = [dataset_docs[i] for i in test_idx]
        y_train, y_test = dataset_labels[train_idx], dataset_labels[test_idx]

        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case=False)
        train_encodings = tokenizer(dataset_docs_train, truncation=True, padding=True)
        test_encodings = tokenizer(dataset_docs_test, truncation=True, padding=True)
        train_dataset = TextDataset(train_encodings, y_train)
        test_dataset = TextDataset(test_encodings, y_test)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=64, num_workers=4)
        val_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)

        device = torch.device("cuda")
        num_labels = 4 if args.train_dataset_name == 'mixed' else 2
        model = AutoModelForSequenceClassification.from_pretrained(f'./weights/beto/{args.train_dataset_name}/{args.attribute}/fold_{fold_idx}/', num_labels=num_labels)
        model.to(device)

        test_preds, _ = collect_predictions(model, val_dataloader, device)
        pred_filename = f'./predictions/beto/{args.train_dataset_name}_{args.dataset_name}/{args.attribute}/fold_{fold_idx}/predictions.npy'
        os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
        np.save(pred_filename, test_preds)

        model_features = AutoModelForSequenceClassification.from_pretrained(f'./weights/beto/{args.train_dataset_name}/{args.attribute}/fold_{fold_idx}/', num_labels=num_labels)
        model_features.dropout = nn.Identity()
        model_features.classifier = nn.Identity()
        model_features.to(device)

        output_path = pathlib.Path(f'./extracted_features/{args.train_dataset_name}_beto_{args.dataset_name}/')
        os.makedirs(output_path, exist_ok=True)
        train_features, train_labels = collect_predictions(model_features, train_dataloader, device)
        np.save(output_path / f'fold_{fold_idx}_X_train_{args.attribute}.npy', train_features)
        np.save(output_path / f'fold_{fold_idx}_y_train_{args.attribute}.npy', train_labels)
        test_features, test_labels = collect_predictions(model_features, val_dataloader, device)
        np.save(output_path / f'fold_{fold_idx}_X_test_{args.attribute}.npy', test_features)
        np.save(output_path / f'fold_{fold_idx}_y_test_{args.attribute}.npy', test_labels)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_name', choices=('esp_fake', 'mixed'), required=True, help='dataset model was trained on')
    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'bs_detector', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


def collect_predictions(model, dataloader, device):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            y_pred = outputs.logits.softmax(dim=1)
            test_preds.append(y_pred.cpu().numpy())
            test_labels.append(batch['labels'].numpy())
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return test_preds, test_labels


if __name__ == '__main__':
    # model = AutoModelForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    # print(model)
    # model.save_pretrained(f'./weights/beto/fold_{0}/')
    # model_new = AutoModelForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    # # model_new.from_pretrained(f'./weights/beto/fold_{0}/')
    # for (name, param), (_, new_param) in zip(model.named_modules(), model_new.named_modules()):
    #     if name not in ['', 'bert', 'bert.embeddings',
    #                     'bert.embeddings.word_embeddings',
    #                     'bert.embeddings.position_embeddings',
    #                     'bert.embeddings.token_type_embeddings',
    #                     'bert.embeddings.LayerNorm',
    #                     'bert.embeddings.dropout',
    #                     'bert.encoder', 'bert.pooler.dense.weight', 'classifier.weight', 'classifier.bias', 'bert.pooler.dense.bias']:
    #         print(name)
    #         assert param == new_param
    # assert model == model_new

    main()
