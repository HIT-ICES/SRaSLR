import gensim
import nltk
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        # super(BertTextClassificationDataset, self).__init__()
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return torch.as_tensor(sequence).long(), torch.as_tensor(label).long()


# %%
class TextCNN(pl.LightningModule):
    def __init__(self, word_vectors, embed_size, feature_size, window_size, max_seq_len, output_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_vectors.vectors), freeze=False)

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embed_size,
                                    out_channels=feature_size,
                                    kernel_size=kernel),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=max_seq_len - kernel + 1))
            for kernel in window_size
        ])
        self.fc = nn.Linear(in_features=feature_size * len(window_size),
                            out_features=output_size)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, 128]

        # print('embed size 1',embed_x.size())  # 32*35*256
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        # embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*35
        # x: [batch, seq_len, emb_size]
        x = x.permute(0, 2, 1)  # [batch, emb_size, seq_len]
        outputs = [conv(x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        output = torch.cat(outputs, dim=1)

        output = output.squeeze(-1)  # [batch, feature_size * window]

        output = F.dropout(input=output, p=0.5)
        output = self.fc(output)
        return output  # [batch, feature_size * window]

    def training_step(self, train_batch, batch_idx):
        sequences, label = train_batch
        output = self(sequences)
        loss = F.cross_entropy(output, label, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, label = val_batch
        output = self(sequences)
        loss = F.cross_entropy(output, label)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        sequences, labels = test_batch
        output = self(sequences)
        output = output.to('cpu').numpy()
        target = labels.to('cpu').numpy()
        correct_top1, incorrect_top1 = 0, 0
        correct_top5, incorrect_top5 = 0, 0
        for o, t in zip(output, target):
            sorted_args = np.argsort(-o)
            if sorted_args[0] == t:
                correct_top1 += 1
            else:
                incorrect_top1 += 1
            if t in sorted_args[:5]:
                correct_top5 += 1
            else:
                incorrect_top5 += 1
        return {"correct_top1": correct_top1, "correct_top5": correct_top5, "incorrect_top1": incorrect_top1,
                "incorrect_top5": incorrect_top5}

    def test_epoch_end(self, outputs):
        correct_top1, incorrect_top1 = 0, 0
        correct_top5, incorrect_top5 = 0, 0
        for out in outputs:
            correct_top1 += out["correct_top1"]
            incorrect_top1 += out["incorrect_top1"]
            correct_top5 += out["correct_top5"]
            incorrect_top5 += out["incorrect_top5"]
        print({"acc_top1": correct_top1 / (correct_top1 + incorrect_top1),
               "acc_top5": correct_top5 / (correct_top5 + incorrect_top5)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def padding(sequence, ):
    # words_in_one_seq = nltk.word_tokenize(sequence.lower().strip())  # 用nltk分词
    # 截断或pad
    if len(sequence) > SEQ_MAX_LEN:
        # 大于最大长度，截断
        sampled_seq = sequence[:SEQ_MAX_LEN]
        return sampled_seq
    else:
        # 不足最大长度，在最后补m个pad字符串
        padding_len = SEQ_MAX_LEN - len(sequence)
        padded_seq = sequence + ['<pad>'] * padding_len
        return padded_seq


def get_word_id(raw_word):
    if raw_word in word_vectors:
        return word_vectors.vocab[raw_word].index
    else:
        return word_vectors.vocab['<unk>'].index


# %%
NUM_CLASS = 255
SEQ_MAX_LEN = 300

X = list(range(4099))
train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)
train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
    'data/pw_word_vectors.txt', binary=False)
text_all = np.array([line.strip().split(' =->= ')[1]
                     for line in
                     open('data/dscps_encoded.txt', 'r', encoding='utf8').readlines()])
tags_all = np.array(
    [int(line.strip()) for line in
     open('data/tags_id.txt', 'r', encoding='utf8').readlines()])

tokenized_text_all = [padding(nltk.word_tokenize(seq)) for seq in text_all]
text_ids_all = []
for tokenized_text in tokenized_text_all:
    text_ids = []
    for word in tokenized_text:
        text_ids.append(get_word_id(word))
    text_ids_all.append(text_ids)
text_ids_all = np.array(text_ids_all)

train_X = list(text_ids_all[train_ids])
train_Y = tags_all[train_ids]
train_dataset = CNNDataset(train_X, train_Y)

val_X = list(text_ids_all[val_ids])
val_Y = tags_all[val_ids]
val_dataset = CNNDataset(val_X, val_Y)

test_X = list(text_ids_all[test_ids])
test_Y = tags_all[test_ids]
test_dataset = CNNDataset(test_X, test_Y)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = TextCNN(word_vectors, 128, 768, [5, 6, 7, 8], SEQ_MAX_LEN, NUM_CLASS)
trainer = pl.Trainer(max_epochs=50, gpus=1, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
