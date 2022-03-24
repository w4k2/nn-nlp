import argparse
import torch
import datasets
import pathlib
import os
import numpy as np
import sklearn.model_selection
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler, AdamW
from sklearn.metrics import accuracy_score


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

    dataset_docs, dataset_labels = datasets.load_dataset(args.dataset_name)

    acc_all = []

    k_fold = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    pbar = tqdm(enumerate(k_fold.split(dataset_docs, dataset_labels)), desc='Fold feature extraction', total=10)
    for fold_idx, (train_idx, test_idx) in pbar:
        torch.cuda.empty_cache()

        dataset_docs_train = [dataset_docs[i] for i in train_idx]
        dataset_docs_test = [dataset_docs[i] for i in test_idx]
        y_train, y_test = dataset_labels[train_idx], dataset_labels[test_idx]
        if args.dataset_name == 'mixed':
            y_train[np.argwhere(y_train == 2).flatten()] = 0
            y_train[np.argwhere(y_train == 3).flatten()] = 1
            y_test[np.argwhere(y_test == 2).flatten()] = 0
            y_test[np.argwhere(y_test == 3).flatten()] = 1

        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case=False)

        train_encodings = tokenizer(dataset_docs_train, truncation=True, padding=True)
        test_encodings = tokenizer(dataset_docs_test, truncation=True, padding=True)

        train_dataset = TextDataset(train_encodings, y_train)
        test_dataset = TextDataset(test_encodings, y_test)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        val_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

        device = torch.device("cuda")
        # model = BertForMaskedLM.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=3e-5)

        num_epochs = 5
        num_training_steps = num_epochs * len(train_dataloader)
        num_warmup_steps = int(0.1*num_training_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for _ in range(num_epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                y_pred = outputs.logits.softmax(dim=1).argmax(dim=1)
                acc = torch.sum(y_pred == labels) / len(labels)
                progress_bar.set_description(f"acc = {acc}")

        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                y_pred = outputs.logits.softmax(dim=1).argmax(dim=1)
                test_labels.extend(batch['labels'].numpy())
                test_preds.extend(y_pred.cpu().numpy())

        accuracy = accuracy_score(test_labels, test_preds)
        print(f'fold {fold_idx} = {accuracy}')
        acc_all.append(accuracy)

        model.save_pretrained(f'./weights/beto/{args.dataset_name}/{args.attribute}/fold_{fold_idx}/')

    output_path = pathlib.Path('results/')
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / f'{args.dataset_name}_beto_{args.attribute}.npy', acc_all)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=('esp_fake', 'mixed'))
    parser.add_argument('--attribute', choices=('text', 'title'), required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # model = AutoModelForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
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

    # dataset_docs, dataset_labels = datasets.load_dataset('bs_detector')
    # # dataset_docs, dataset_labels = datasets.load_dataset('esp_fake')

    # dataset_docs = list(dataset_docs)
    # # print(type(dataset_docs))
    # # exit()
    # tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case=False)
    # # print(tokenizer.vocab)
    # print(tokenizer.ids_to_tokens[0])
    # print(tokenizer.ids_to_tokens[1])
    # print(tokenizer.ids_to_tokens[1100])
    # print(tokenizer.ids_to_tokens[4963])
    # print(dataset_docs[0])

    # train_encodings = tokenizer(dataset_docs[0], truncation=True, padding=True)
    # # print(train_encodings)

    # reverse_mapping = ""
    # for token_id in train_encodings['input_ids']:
    #     word = tokenizer.ids_to_tokens[token_id]
    #     reverse_mapping += f" {word}"
    # print(reverse_mapping)
