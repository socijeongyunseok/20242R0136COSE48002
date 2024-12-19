# main_revised

# -*- coding: utf-8 -*-

import argparse
import os
import sys
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Dialog Retrieval using CALIOPER')
    parser.add_argument('--train_data', help='Training data', default='simsimi', type=str)
    parser.add_argument('--test_data', help='Test data', default=None, type=str)
    parser.add_argument('--speaker_tags', help='Speaker tags', default='A', type=str)
    parser.add_argument('--utterance_separator', help='Utterance separator token', default='[SEP]', type=str)
    parser.add_argument('--main_encoder', help='Main encoder name', default='bert-base-uncased', type=str)
    parser.add_argument('--context_encoder', help='Context encoder name', default='bert-base-uncased', type=str)
    parser.add_argument('--use_only_last', help='Use only the last utterance', action='store_true')
    parser.add_argument('--epochs', help='Epochs', default=10, type=int)
    parser.add_argument('--batch_size', help='Batch size', default=32, type=int)
    parser.add_argument('--gpus', help='GPUs to use', default='0', type=str)
    parser.add_argument('--repeat', help='Repeat', default=1, type=int)
    parser.add_argument('--use_context_info', help='Use date and additional context', action='store_true')
    args = parser.parse_args()
    return args

class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, full_dialog, previous_utterance_representations, labels, tokenizer, max_len, max_utterance):
        self.full_dialog = full_dialog
        self.previous_utterance_representations = previous_utterance_representations
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_utterance = max_utterance
        self.labels = labels.values.tolist()

    def __len__(self):
        return len(self.full_dialog)

    def __getitem__(self, item):
        full_dialog = self.full_dialog[item]
        previous_utterance_representations = self.previous_utterance_representations[item]
        label_data = self.labels[item]

        label = torch.tensor(label_data, dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            full_dialog,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if len(previous_utterance_representations) == 0:
            previous_utterance_representations = torch.zeros((self.max_utterance, 768))
        else:
            previous_utterance_representations = torch.stack(previous_utterance_representations)
            if len(previous_utterance_representations) < self.max_utterance:
                pad_size = self.max_utterance - len(previous_utterance_representations)
                previous_utterance_representations = torch.cat(
                    (previous_utterance_representations, torch.zeros((pad_size, 768))))
            else:
                previous_utterance_representations = previous_utterance_representations[:self.max_utterance]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'previous_utterance_representations': previous_utterance_representations,
            'labels': label
        }

class CALIOPER(nn.Module):
    def __init__(self, bert_model, bert_hidden_size, sentence_bert_hidden_size, dropout, num_labels=6):
        super(CALIOPER, self).__init__()
        self.bert_model = bert_model
        self.bert_hidden_size = bert_hidden_size
        self.sentence_bert_hidden_size = sentence_bert_hidden_size
        self.bert_output_transform = nn.Linear(self.bert_hidden_size, self.sentence_bert_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(bert_hidden_size + sentence_bert_hidden_size, bert_hidden_size)
        self.classifier = nn.Linear(bert_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, previous_utterance_representations):
        # 현재 발화의 [CLS] 토큰 출력
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output[1]  # [CLS] representation

        # 변환 레이어
        bert_output_transformed = self.bert_output_transform(bert_output)

        # Attention Mechanism
        attention = torch.bmm(previous_utterance_representations, bert_output_transformed.unsqueeze(2)).squeeze(2)
        attention = F.softmax(attention, dim=1)

        # 가중합
        weighted_sum = torch.bmm(previous_utterance_representations.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        # 결합 및 분류
        output = torch.cat((bert_output, weighted_sum), dim=1)
        output = self.dropout(output)
        output = self.hidden_layer(output)
        output = F.relu(output)
        logits = self.classifier(output)

        return logits

def load_data(name, split):
    if name == 'simsimi':
        data = load_simsimi(split)
    elif name == 'fm_yunseok':
        data = load_fm_yunseok(split)
    else:
        raise ValueError('Invalid dataset name')
    return data

def load_simsimi(split):
    if split == 'train':
        file_path = 'data/train_not_U.tsv'
    elif split == 'test':
        file_path = 'data/test_not_U.tsv'
    else:
        raise ValueError('Invalid split')

    data = pd.read_csv(file_path, sep='\t', index_col=0)
    data['all_utterances'] = data['previous_utterance'].apply(lambda x: eval(x))

    label_columns = [
        'offensive', 'context_dependent',
        'offensiveness_disabled', 'context_offensiveness_disabled',
        'hate_speech_disabled', 'context_hate_speech_disabled'
    ]

    for col in label_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 1 if x == 'Y' else 0)

    # 날짜와 추가 컨텍스트 로드
    if 'date' in data.columns:
        data['date'] = data['date']
    else:
        data['date'] = ''

    if 'additional_context' in data.columns:
        data['additional_context'] = data['additional_context']
    else:
        data['additional_context'] = ''

    return data

def load_fm_yunseok(split):
    if split == 'train':
        file_path = 'data/train_fm_yunseok.tsv'
    elif split == 'test':
        file_path = 'data/test_fm_yunseok.tsv'
    else:
        raise ValueError('Invalid split')

    data = pd.read_csv(file_path, sep='\t', index_col=0)
    data['all_utterances'] = data['previous_utterance'].apply(lambda x: eval(x))

    label_columns = [
        'offensive', 'context_dependent',
        'offensiveness_disabled', 'context_offensiveness_disabled',
        'hate_speech_disabled', 'context_hate_speech_disabled'
    ]

    for col in label_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 1 if x == 'Y' else 0)

    # 날짜와 추가 컨텍스트 로드
    if 'date' in data.columns:
        data['date'] = data['date']
    else:
        data['date'] = ''

    if 'additional_context' in data.columns:
        data['additional_context'] = data['additional_context']
    else:
        data['additional_context'] = ''

    return data

def custom_loss(outputs, labels):
    main_label_indices = [0, 2, 4]      # offensive, offensiveness_disabled, hate_speech_disabled
    context_label_indices = [1, 3, 5]   # context_dependent, context_offensiveness_disabled, context_hate_speech_disabled
    loss = 0
    for main_idx, context_idx in zip(main_label_indices, context_label_indices):
        main_loss = F.binary_cross_entropy_with_logits(outputs[:, main_idx], labels[:, main_idx])
        loss += main_loss
        mask = labels[:, main_idx] == 1
        if mask.sum() > 0:
            context_loss = F.binary_cross_entropy_with_logits(outputs[mask, context_idx], labels[mask, context_idx])
            loss += context_loss
    return loss

def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        previous_utterance_representations = batch['previous_utterance_representations'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            previous_utterance_representations=previous_utterance_representations
        )

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (preds >= 0.5).astype(int)
        all_preds.append(preds)

        labels = labels.detach().cpu().numpy()
        all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    avg_loss = np.mean(losses)

    return {
        'loss': avg_loss,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            previous_utterance_representations = batch['previous_utterance_representations'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                previous_utterance_representations=previous_utterance_representations
            )

            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds >= 0.5).astype(int)
            all_preds.append(preds)

            labels = labels.cpu().numpy()
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    avg_loss = np.mean(losses)

    return {
        'loss': avg_loss,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_data = load_data(args.train_data, 'train')
    if args.test_data is None:
        test_data = load_data(args.train_data, 'test')
    else:
        test_data = load_data(args.test_data, 'test')

    # 레이블 컬럼 설정
    label_columns = [
        'offensive', 'context_dependent',
        'offensiveness_disabled', 'context_offensiveness_disabled',
        'hate_speech_disabled', 'context_hate_speech_disabled'
    ]

    # 트레인 데이터 전처리
    train_text = train_data['text'].tolist()
    train_labels = train_data[label_columns]
    train_all_utterances = train_data['all_utterances']
    train_previous_utterance = []
    train_date = train_data['date'].tolist()
    train_additional_context = train_data['additional_context'].tolist()

    if args.speaker_tags == 'A':
        speaker_tags = ['A: ', 'B: ']
    else:
        speaker_tags = ['']

    for i in range(len(train_all_utterances)):
        previous_utterance = train_all_utterances.iloc[i][:-1]
        for j in range(len(previous_utterance)):
            speaker_tag = speaker_tags[j % 2]
            previous_utterance[j] = speaker_tag + previous_utterance[j]
        train_text[i] = speaker_tags[0] + train_text[i]
        train_previous_utterance.append(previous_utterance)

    train_full_dialog = []
    for i in range(len(train_text)):
        dialog = ''
        if args.use_context_info:
            dialog += 'Date: ' + train_date[i] + ' '
            dialog += 'Context: ' + train_additional_context[i] + ' '
        for utterance in train_previous_utterance[i]:
            dialog += utterance + args.utterance_separator
        dialog += train_text[i]
        train_full_dialog.append(dialog)

    # 테스트 데이터 전처리
    test_text = test_data['text'].tolist()
    test_labels = test_data[label_columns]
    test_all_utterances = test_data['all_utterances']
    test_previous_utterance = []
    test_date = test_data['date'].tolist()
    test_additional_context = test_data['additional_context'].tolist()

    for i in range(len(test_all_utterances)):
        previous_utterance = test_all_utterances.iloc[i][:-1]
        for j in range(len(previous_utterance)):
            speaker_tag = speaker_tags[j % 2]
            previous_utterance[j] = speaker_tag + previous_utterance[j]
        test_text[i] = speaker_tags[0] + test_text[i]
        test_previous_utterance.append(previous_utterance)

    test_full_dialog = []
    for i in range(len(test_text)):
        dialog = ''
        if args.use_context_info:
            dialog += 'Date: ' + test_date[i] + ' '
            dialog += 'Context: ' + test_additional_context[i] + ' '
        for utterance in test_previous_utterance[i]:
            dialog += utterance + args.utterance_separator
        dialog += test_text[i]
        test_full_dialog.append(dialog)

    # Sentence-BERT 모델 로드
    context_encoder_name = args.context_encoder
    context_encoder_name_short = context_encoder_name.split('/')[-1]
    sentence_bert = SentenceTransformer(context_encoder_name, device=device)

    sentence_bert_dict_path = f'./models/sentence_bert_dict_{context_encoder_name_short}.pkl'
    if os.path.exists(sentence_bert_dict_path):
        with open(sentence_bert_dict_path, 'rb') as f:
            sentence_bert_dict = pickle.load(f)
    else:
        sentence_bert_dict = {}

    # 트레인 데이터의 이전 발화 임베딩 생성
    train_previous_utterance_representations = []
    for i in tqdm(range(len(train_previous_utterance)), desc="Encoding Train Utterances"):
        previous_utterance_representations = []
        for utterance in train_previous_utterance[i]:
            if utterance in sentence_bert_dict:
                embedding = sentence_bert_dict[utterance]
            else:
                embedding = sentence_bert.encode(utterance)
                sentence_bert_dict[utterance] = embedding
            previous_utterance_representations.append(torch.tensor(embedding))
        train_previous_utterance_representations.append(previous_utterance_representations)

    # 테스트 데이터의 이전 발화 임베딩 생성
    test_previous_utterance_representations = []
    for i in tqdm(range(len(test_previous_utterance)), desc="Encoding Test Utterances"):
        previous_utterance_representations = []
        for utterance in test_previous_utterance[i]:
            if utterance in sentence_bert_dict:
                embedding = sentence_bert_dict[utterance]
            else:
                embedding = sentence_bert.encode(utterance)
                sentence_bert_dict[utterance] = embedding
            previous_utterance_representations.append(torch.tensor(embedding))
        test_previous_utterance_representations.append(previous_utterance_representations)

    # 임베딩 저장
    with open(sentence_bert_dict_path, 'wb') as f:
        pickle.dump(sentence_bert_dict, f)

    # 토크나이저 로드
    main_encoder_name = args.main_encoder
    tokenizer = AutoTokenizer.from_pretrained(main_encoder_name)

    # 데이터셋 생성
    train_dataset = DialogDataset(
        full_dialog=train_full_dialog if not args.use_only_last else train_text,
        previous_utterance_representations=train_previous_utterance_representations,
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=512,
        max_utterance=10
    )

    test_dataset = DialogDataset(
        full_dialog=test_full_dialog if not args.use_only_last else test_text,
        previous_utterance_representations=test_previous_utterance_representations,
        labels=test_labels,
        tokenizer=tokenizer,
        max_len=512,
        max_utterance=10
    )

    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset),
        num_workers=0
    )

    # 모델 초기화
    bert = AutoModel.from_pretrained(main_encoder_name)
    num_labels = len(label_columns)
    model = CALIOPER(
        bert_model=bert,
        bert_hidden_size=bert.config.hidden_size,
        sentence_bert_hidden_size=768,
        dropout=0.1,
        num_labels=num_labels
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = args.epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    eval_results = pd.DataFrame(columns=['model', 'accuracy', 'f1_micro', 'f1_macro'])

    for repeat_count in range(args.repeat):
        print(f"Starting training iteration {repeat_count + 1}/{args.repeat}")

        model = CALIOPER(
            bert_model=bert,
            bert_hidden_size=bert.config.hidden_size,
            sentence_bert_hidden_size=768,
            dropout=0.1,
            num_labels=num_labels
        )
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_f1_macro = 0.0

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_metrics = train_epoch(
                model=model,
                dataloader=train_dataloader,
                loss_fn=custom_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )

            val_metrics = eval_model(
                model=model,
                dataloader=test_dataloader,
                loss_fn=custom_loss,
                device=device
            )

            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Macro F1: {train_metrics['f1_macro']:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"Validation Macro F1: {val_metrics['f1_macro']:.4f}")

            if val_metrics['f1_macro'] > best_f1_macro:
                best_f1_macro = val_metrics['f1_macro']
                model_filename = f'models/dialog_retrieval_model_best_{repeat_count + 1}.pt'
                torch.save(model.state_dict(), model_filename)
                print("Best model saved.")

        y_true = np.vstack(val_metrics['all_labels'])
        y_pred = np.vstack(val_metrics['all_preds'])

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Micro F1-score: {f1_micro:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")

        # 혐오 발언으로 분류된 댓글과 맥락 저장
        hate_speech_index = label_columns.index('hate_speech_disabled')
        hate_speech_comments = []
        for i in range(len(y_pred)):
            if y_pred[i][hate_speech_index] == 1:
                comment = test_data['text'].iloc[i]
                context = test_data['all_utterances'].iloc[i]
                hate_speech_comments.append({'comment': comment, 'context': context})

        hate_speech_df = pd.DataFrame(hate_speech_comments)
        hate_speech_df.to_csv('hate_speech_comments.csv', index=False)

        # 모델 저장
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_suffix = f"{args.main_encoder}_{args.context_encoder}_{timestamp}"
        model_filename_full = f'models/dialog_retrieval_model_{args.train_data}_{model_suffix}.pt'
        torch.save(model.state_dict(), model_filename_full)

        eval_results = eval_results.append({
            'model': model_filename_full,
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }, ignore_index=True)

        eval_results.to_csv(f'results/eval_results_{args.train_data}_{args.test_data}_{model_suffix}.csv', index=False)

        print(eval_results)
        print(eval_results.mean())

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_columns, zero_division=0))

if __name__ == '__main__':
    main()
