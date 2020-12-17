import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
import os


class OconTrainDataset(Dataset):
    def __init__(self,
                 data,
                 cat_feat_name,
                 num_feat_name,
                 global_cfg,
                 include_last_event=True,
                 max_seq_len=500):
        super().__init__()
        self._ids = data['id'].to_numpy()
        self._times = data['time'].to_numpy()
        self._events = (data[cat_feat_name] + 1).to_numpy()
        self._time_deltas = data[num_feat_name].to_numpy()

        self.prediction_start = global_cfg.prediction_start
        self.prediction_end = global_cfg.prediction_end
        self.include_last_event = include_last_event
        self.padding = global_cfg.padding
        self.dobavka = global_cfg.dobavka
        self.drop_ratio = global_cfg.drop_ratio
        self.train_in_prediction_window = global_cfg.train_in_prediction_window

        self.max_seq_len = max_seq_len
        self._generate_sequences()

    def _generate_sequences(self):
        timestamps = []
        cat_feats = []
        num_feats = []
        returned = []
        cur_start, cur_end = 0, 1

        while cur_start < len(self._ids):
            if cur_end < len(self._ids) and \
                    self._ids[cur_start] == self._ids[cur_end] and \
                    cur_end - cur_start < self.max_seq_len and \
                    (self._times[cur_end] < self.prediction_start if not\
                        self.train_in_prediction_window else True):
                cur_end += 1
                continue
            else:
                # if cur_end belongs to id of the next ATM:
                if cur_end == len(self._ids) or \
                        self._ids[cur_start] != self._ids[cur_end] or \
                        (self._times[cur_end] > self.prediction_start if \
                            self.train_in_prediction_window else False):
                    returned.append(torch.BoolTensor([True] * (cur_end - cur_start - 1) + [False]))
                    if self.include_last_event:
                        ts = np.concatenate([self._times[cur_start:cur_end],
                                             np.ones(1) * (self.prediction_end + self.dobavka)])
                        timestamps.append(torch.FloatTensor(ts))
                        cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end]))
                        num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end]))
                    else:
                        if cur_start == cur_end - 1:
                            cur_start, cur_end = cur_end, cur_end + 1
                            continue
                        timestamps.append(torch.FloatTensor(self._times[cur_start:cur_end]))
                        cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end - 1]))
                        num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end - 1]))
                else:
                    if np.random.rand() < self.drop_ratio:
                        cur_start, cur_end = cur_end, cur_end + 1
                        continue
                    returned.append(torch.BoolTensor([True] * (cur_end - cur_start - 1)))
                    timestamps.append(torch.FloatTensor(self._times[cur_start:cur_end]))
                    cat_feats.append(torch.LongTensor(self._events[cur_start:cur_end - 1]))
                    num_feats.append(torch.FloatTensor(self._time_deltas[cur_start:cur_end - 1]))

                if not self.train_in_prediction_window and self._times[cur_end] > self.prediction_start:
                    while cur_end < len(self._ids) and \
                        self._ids[cur_end] == self._ids[cur_start]:
                        cur_end += 1

                cur_start, cur_end = cur_end, cur_end + 1

        self.timestamps = timestamps
        self.cat_feats = [x.unsqueeze(-1) for x in cat_feats]
        self.num_feats = [x.unsqueeze(-1) for x in num_feats]
        self.returned = returned

    def __getitem__(self, id):
        return self.timestamps[id], \
               self.cat_feats[id], \
               self.num_feats[id], \
               self.returned[id]

    def __len__(self):
        return len(self.timestamps)


def pad_collate_train(batch, padding):
    (timestamps, cat_feats, num_feats, return_mask) = zip(*batch)
    lens = np.array([len(seq) for seq in cat_feats])
    timestamps_padded = pad_sequence(timestamps, batch_first=True, padding_value=0)
    cat_feats_padded = pad_sequence(cat_feats, batch_first=True, padding_value=0)
    num_feats_padded = pad_sequence(num_feats, batch_first=True, padding_value=0)
    return_mask = pad_sequence(return_mask, batch_first=True, padding_value=0)
    non_pad_mask = cat_feats_padded.ne(padding).squeeze()

    return timestamps_padded, \
           cat_feats_padded, \
           num_feats_padded, \
           non_pad_mask, \
           return_mask, \
           lens


class OconTestDataset(Dataset):
    def __init__(self,
                 data,
                 cat_feat_name,
                 num_feat_name,
                 global_cfg,
                 max_seq_len=500):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.prediction_start = global_cfg.prediction_start
        self.prediction_end = global_cfg.prediction_end
        self.activity_start = global_cfg.activity_start

        has_event_in_activity = lambda x: ((x['time'] < self.prediction_start) &
                                           (x['time'] > self.activity_start)).any()
        valid_ids = data.groupby('id').filter(has_event_in_activity).id.unique()
        self._ids = data['id'][np.isin(data['id'], valid_ids)].to_numpy()
        self._times = data['time'][np.isin(data['id'], valid_ids)].to_numpy()
        self._events = (data[cat_feat_name] + 1)[np.isin(data['id'], valid_ids)].to_numpy()
        self._time_deltas = data[num_feat_name][np.isin(data['id'], valid_ids)].to_numpy()
        self._generate_sequences()

    def _generate_sequences(self):
        timestamps = []
        cat_feats = []
        num_feats = []
        targets = []
        cur_start, cur_end = 0, 1

        while cur_start < len(self._ids) - 1:
            if cur_end < len(self._ids) - 1 and \
                    self._ids[cur_start] == self._ids[cur_end] and \
                    self._times[cur_end] <= self.prediction_start:
                cur_end += 1
                continue
            else:
                start_id = max(cur_start, cur_end - self.max_seq_len)
                if self._times[cur_end] <= self.prediction_start or \
                        self._times[cur_end] > self.prediction_end:
                    timestamps.append(torch.FloatTensor(
                        self._times[start_id:cur_end]))
                    cat_feats.append(torch.LongTensor(
                        self._events[start_id:cur_end]))
                    num_feats.append(torch.FloatTensor(
                        self._time_deltas[start_id:cur_end]))
                    targets.append(torch.FloatTensor([-1]))
                else:
                    timestamps.append(torch.FloatTensor(
                        self._times[start_id:cur_end]))
                    cat_feats.append(torch.LongTensor(
                        self._events[start_id:cur_end]))
                    num_feats.append(torch.FloatTensor(
                        self._time_deltas[start_id:cur_end]))
                    targets.append(torch.FloatTensor([self._times[cur_end]]))

                while cur_end < len(self._ids) - 1 and \
                        self._ids[cur_start] == self._ids[cur_end]:
                    cur_start, cur_end = cur_end, cur_end + 1
                cur_start, cur_end = cur_end, cur_end + 1

        self.timestamps = timestamps
        self.cat_feats = [x.unsqueeze(-1) for x in cat_feats]
        self.num_feats = [x.unsqueeze(-1) for x in num_feats]
        self.targets = targets

    def __getitem__(self, id):
        return self.timestamps[id], \
               self.cat_feats[id], \
               self.num_feats[id], \
               self.targets[id]

    def __len__(self):
        return len(self.timestamps)


def pad_collate_test(batch):
    (timestamps, cat_feats, num_feats, targets) = zip(*batch)
    lens = np.array([len(seq) for seq in timestamps])
    timestamps_padded = pad_sequence(timestamps, batch_first=True, padding_value=0)
    cat_feats_padded = pad_sequence(cat_feats, batch_first=True, padding_value=0)
    num_feats_padded = pad_sequence(num_feats, batch_first=True, padding_value=0)
    targets = torch.FloatTensor(targets).squeeze()

    return timestamps_padded, \
           cat_feats_padded, \
           num_feats_padded, \
           targets, \
           lens


def get_ocon_train_val_loaders(model,
                               cat_feat_name,
                               num_feat_name,
                               global_cfg,
                               path=None,
                               batch_size=32,
                               max_seq_len=500,
                               train_ratio=0.7,
                               seed=42):
    if path is None:
        filename = 'data/OCON/train.csv'
    else:
        filename = path

    assert (model in ['rmtpp', 'rnnsm', 'grobformer'])

    activity_start = global_cfg.activity_start
    prediction_start = global_cfg.prediction_start
    prediction_end = global_cfg.prediction_end

    data = pd.read_csv(filename)
    ids = data.id.unique()
    train_ids, val_ids = train_test_split(ids, train_size=train_ratio, random_state=seed)
    train, val = data[np.isin(data.id, train_ids)], data[np.isin(data.id, val_ids)]
    train_ds = OconTrainDataset(train,
                                cat_feat_name,
                                num_feat_name,
                                global_cfg,
                                include_last_event=model == 'rnnsm' or model == 'grobformer',
                                max_seq_len=max_seq_len)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda x: pad_collate_train(x, global_cfg.padding),
                              drop_last=True)

    val_ds = OconTestDataset(val, cat_feat_name, num_feat_name, global_cfg, max_seq_len=max_seq_len)
    val_loader = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=pad_collate_test,
                            drop_last=False)

    return train_loader, val_loader


def get_ocon_test_loader(cat_feat_name,
                         num_feat_name,
                         global_cfg,
                         path=None,
                         batch_size=32,
                         max_seq_len=500):
    if path is None:
        filename = 'data/OCON/train.csv'
    else:
        filename = path

    activity_start = global_cfg.activity_start
    prediction_start = global_cfg.prediction_start
    prediction_end = global_cfg.prediction_end

    csv = pd.read_csv(filename)

    test_ds = OconTestDataset(ids, cat_feat_name, num_feat_name, global_cfg,
                              prediction_end)
    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=pad_collate_test,
                             drop_last=False)

    return test_loader
