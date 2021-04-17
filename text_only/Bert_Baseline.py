import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertConfig, BertModel


class BertTextClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, input_ids, tokentype_ids, attention_mask, targets, sample_weight=None, transform=None):
        self.input_ids = np.array(input_ids, dtype=np.int32)
        self.tokentype_ids = np.array(tokentype_ids, dtype=np.int32)
        self.attention_mask = np.array(attention_mask, dtype=np.int32)
        self.targets = np.array(targets, dtype=np.int32)
        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        else:
            self.sample_weight = np.ones(self.targets.shape)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_id, tokentype_id, attention_mask, target = \
            self.input_ids[index], self.tokentype_ids[index], self.attention_mask[index], self.targets[index]
        sample_weight = self.sample_weight[index]
        return torch.as_tensor(input_id).type(torch.LongTensor), torch.as_tensor(tokentype_id).type(torch.LongTensor), \
               torch.as_tensor(attention_mask).type(torch.LongTensor), \
               torch.as_tensor(target).type(torch.LongTensor), torch.as_tensor(sample_weight.astype('float'))


class BertClassifier(pl.LightningModule):
    def __init__(self, bert_checkpoint, classes, dense_dropout=0.5, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # Load model
        self.config = BertConfig.from_pretrained(bert_checkpoint)
        self.encoder = BertModel.from_pretrained(bert_checkpoint, config=self.config)
        self.decoder = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=self.config.hidden_size, out_features=classes),
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = output[0][:, 0]
        output = self.decoder(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-8)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, sample_weight = train_batch
        output = self.forward(input_id, tokentype_id, attention_mask)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss * sample_weight
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, sample_weight = val_batch
        output = self.forward(input_id, tokentype_id, attention_mask)
        loss = F.cross_entropy(output, target)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, sample_weight = val_batch
        output = self.forward(input_id, tokentype_id, attention_mask)
        output = output.to('cpu').numpy()
        target = target.to('cpu').numpy()
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


CLASSNUM = 255

X = list(range(4099))

train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)

train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)

text_all = np.array([line.strip().split(' =->= ')[1] for line in open('data/dscps_encoded.txt', 'r').readlines()])
tags_all = np.array([int(line.strip()) for line in open('data/tags_id.txt').readlines()])

# Bert prepare
bert_checkpoint = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)

MAX_LENGTH = 300
train_X = tokenizer(list(text_all[train_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
train_Y = tags_all[train_ids]
train_dataset = BertTextClassificationDataset(train_X['input_ids'], train_X['token_type_ids'],
                                              train_X['attention_mask'], train_Y)

val_X = tokenizer(list(text_all[val_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
val_Y = tags_all[val_ids]
val_dataset = BertTextClassificationDataset(val_X['input_ids'], val_X['token_type_ids'], val_X['attention_mask'], val_Y)

test_X = tokenizer(list(text_all[test_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
test_Y = tags_all[test_ids]
test_dataset = BertTextClassificationDataset(test_X['input_ids'], test_X['token_type_ids'], test_X['attention_mask'],
                                             test_Y)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

bert_model = BertClassifier(bert_checkpoint, CLASSNUM)

trainer = pl.Trainer(max_epochs=50, gpus=1, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(bert_model, train_dl, val_dl)
# trainer = pl.Trainer(max_epochs=50, gpus='0', callbacks=[EarlyStopping(monitor='val_loss')])

print(trainer.test(bert_model, test_dl))
