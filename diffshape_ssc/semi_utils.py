import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import T5Tokenizer, T5Model
from semi_backbone import FCN, Classifier
import torch.utils.data as data
import random
import torch.nn as nn


DEFAULT_T5_NAME = 't5-base'
MODEL_NAME = DEFAULT_T5_NAME

prompt_toolkit_series = ['This time series is ', 'This time series can be described as ',
                         'The key attributes of this time series are ', 'The nature of this time series is depicted by ',
                         'Here, the time series is defined by ',
                         'Describing this time series, we find ', 'Examining this time series reveals ',
                         'The features exhibited in this time series are ']
prompt_toolkit_end = ['.']


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_loss(args):
    if args.loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif args.loss == 'reconstruction':
        return nn.MSELoss()


def lan_shapelet_contrastive_loss(embd_batch, text_embd_batch, labels, device,
                              temperature=0.07, base_temperature=0.07):
    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, text_embd_batch.T),
        temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1

    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.sum(0) / (num_anchor + 1e-12)

    return loss


def get_all_text_labels(ucr_datasets_dict, dataset_name, num_labels, device, prompt_toolkit_series_i=None):
    text_labels = torch.empty(size=(num_labels, 768))
    for _label in range(num_labels):
        text_labels[_label] = get_ont_text_label(ucr_datasets_dict=ucr_datasets_dict, dataset_name=dataset_name,
                                                 num_label=_label, device=device,
                                                 prompt_toolkit_series_i=prompt_toolkit_series_i)

    return text_labels.to(device)


def get_ont_text_label(ucr_datasets_dict, dataset_name, num_label, device, prompt_toolkit_series_i=None):
    text_label = ucr_datasets_dict[dataset_name][str(num_label)]

    if prompt_toolkit_series_i is not None:
        text_label = prompt_toolkit_series[prompt_toolkit_series_i] + text_label + prompt_toolkit_end[0]

    # loading model and tokenizer
    # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    # model = T5Model.from_pretrained(MODEL_NAME)

    tokenizer = T5Tokenizer.from_pretrained('/dev_data/lz/t5-base')
    model = T5Model.from_pretrained('/dev_data/lz/t5-base')

    inputs = tokenizer(text_label, return_tensors='pt', padding=True)

    output = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
    pooled_sentence = output.last_hidden_state.detach()  # shape is [batch_size, seq_len, hidden_size]

    return torch.mean(pooled_sentence, dim=1)[0]


def get_each_sample_distance_shapelet(generator_shapelet, raw_shapelet, topk=1):
    if len(generator_shapelet.shape) < 3:
        emb1 = torch.unsqueeze(generator_shapelet, 1)  # n*1*d
    else:
        emb1 = generator_shapelet
    emb2 = torch.unsqueeze(raw_shapelet, 0)  # 1*n*d
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    topk, indices = torch.topk(w, topk)

    indices = torch.squeeze(indices)

    indices_raw_shapelets = raw_shapelet[indices]

    return topk.reshape(-1), indices_raw_shapelets


def get_similarity_shapelet(generator_shapelet):
    generator_shapelet = torch.squeeze(generator_shapelet)

    emb1 = torch.unsqueeze(generator_shapelet, 1)  # n*1*d
    emb2 = torch.unsqueeze(generator_shapelet, 0)  # 1*n*d

    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    return torch.norm(w)


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)

    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)

    if args.classifier == 'linear':
        classifier = Classifier(args.classifier_input, args.num_classes)

    return model, classifier


def load_data(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32)
    sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(sum_target))

    return sum_dataset, sum_target, num_classes


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def get_all_datasets(data, target):
    return k_fold(data, target)


def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []

    for raw_index, test_index in skf.split(data, target):
        raw_set = data[raw_index]
        raw_target = target[raw_index]

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets


def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_


def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set


def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


def evaluate_model_acc(val_loader, model, fcn_model, fcn_classifier):
    target_true = []
    target_pred = []

    num_val_samples = 0
    for data, target in val_loader:
        with torch.no_grad():
            predicted = model(torch.unsqueeze(data, 2))   ## torch.unsqueeze(x, 2)

            fcn_cls_emb = fcn_model(torch.squeeze(predicted, 2))
            val_pred = fcn_classifier(fcn_cls_emb)

            target_true.append(target.cpu().numpy())
            target_pred.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            num_val_samples = num_val_samples + len(target)

    target_true = np.concatenate(target_true)
    target_pred = np.concatenate(target_pred)

    return accuracy_score(target_true, target_pred)


def evaluate(val_loader, model, classifier, loss):

    target_true = []
    target_pred = []

    val_loss = 0
    sum_len = 0
    for data, target in val_loader:
        with torch.no_grad():
            val_pred = model(data)
            val_pred = classifier(val_pred)
            val_loss += loss(val_pred, target).item()
            target_true.append(target.cpu().numpy())
            target_pred.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            sum_len += len(target)

    return val_loss / sum_len, accuracy_score(target_true, target_pred)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, output_dim=32) -> None:
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x):
        return self.projection_head(x)


def sup_contrastive_loss(embd_batch, labels, device,
                         temperature=0.07, base_temperature=0.07):
    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, embd_batch.T),
        temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.sum(0) / (num_anchor + 1e-12)

    return loss


def get_pesudo_via_high_confidence_softlabels(y_label, pseudo_label_soft, mask_label, num_real_class, device, p_cutoff=0.95):
    all_end_label = torch.argmax(pseudo_label_soft, 1)
    pseudo_label_hard = torch.argmax(pseudo_label_soft, 1)

    class_counter = Counter(y_label[mask_label])
    for i in range(num_real_class):
        class_counter[i] = 0

    for i in range(len(mask_label)):
        if mask_label[i] is False: ## unlabeled data
            class_counter[pseudo_label_hard[i]] += 1
        else:
            all_end_label[i] = y_label[i]

    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        classwise_acc[i] = class_counter[i] / max(class_counter.values())

    pseudo_label = torch.softmax(pseudo_label_soft, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))

    end_mask_labeled = mask_label.copy()
    for i in range(len(end_mask_labeled)):
        if end_mask_labeled[i] is False:
            if cpl_mask[i]:
                end_mask_labeled[i] = True

    return end_mask_labeled, all_end_label
