import gensim
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertConfig, BertModel


class SRaSLRMADataset(torch.utils.data.Dataset):

    def __init__(self, input_ids, tokentype_ids, attention_mask, targets, node_ids=None, transform=None):
        self.input_ids = np.array(input_ids, dtype=np.int32)
        self.tokentype_ids = np.array(tokentype_ids, dtype=np.int32)
        self.attention_mask = np.array(attention_mask, dtype=np.int32)
        self.targets = np.array(targets, dtype=np.int32)
        self.node_ids = np.array(node_ids, dtype=np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_id, tokentype_id, attention_mask, target = \
            self.input_ids[index], self.tokentype_ids[index], self.attention_mask[index], self.targets[index]
        node_id = self.node_ids[index]
        return torch.as_tensor(input_id).type(torch.LongTensor), torch.as_tensor(tokentype_id).type(torch.LongTensor), \
               torch.as_tensor(attention_mask).type(torch.LongTensor), \
               torch.as_tensor(target).type(torch.LongTensor), torch.as_tensor(node_id).type(torch.LongTensor)


class SRaSLRMA(pl.LightningModule):
    def __init__(self, bert_checkpoint, classes, node_embedding, node_embedding_size, dense_dropout=0.5, **kwargs):
        super().__init__()
        # Load model
        self.config = BertConfig.from_pretrained(bert_checkpoint)
        self.bert_encoder = BertModel.from_pretrained(bert_checkpoint, config=self.config)
        self.node_encoder = nn.Embedding.from_pretrained(torch.FloatTensor(node_embedding),
                                                         freeze=False)  # don't freeze
        self.decoder = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=self.config.hidden_size + node_embedding_size,
                      out_features=self.config.hidden_size + node_embedding_size),
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=self.config.hidden_size + node_embedding_size, out_features=classes),
        )

    def forward(self, input_ids, token_type_ids, attention_mask, node_ids):
        bert_output = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_output = bert_output[0][:, 0]
        node_output = self.node_encoder(node_ids)
        output = torch.cat((bert_output, node_output), 1)
        output = self.decoder(output)
        return output

    def configure_optimizers(self):
        bert_params = list(map(id, self.bert_encoder.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params, self.parameters())
        # 分层学习
        optimizer = torch.optim.Adam([
            {"params": self.bert_encoder.parameters(), "lr": 3e-5},
            {"params": base_params, "lr": 0.001},
        ])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, node_id = train_batch
        output = self.forward(input_id, tokentype_id, attention_mask, node_id)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, node_id = val_batch
        output = self.forward(input_id, tokentype_id, attention_mask, node_id)
        loss = F.cross_entropy(output, target)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        input_id, tokentype_id, attention_mask, target, node_id = val_batch
        output = self.forward(input_id, tokentype_id, attention_mask, node_id)
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

## load graph model
PATH = 'data/embed/ma_embedding_100.txt'
node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)

X = list(range(4099))

train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)

train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)

train_nodes = [node_embeddings.vocab[str(n)].index for n in train_ids]
val_nodes = [node_embeddings.vocab[str(n)].index for n in val_ids]
test_nodes = [node_embeddings.vocab[str(n)].index for n in test_ids]

text_all = np.array([line.strip().split(' =->= ')[1] for line in open('data/dscps_encoded.txt', 'r').readlines()])
tags_all = np.array([int(line.strip()) for line in open('data/tags_id.txt').readlines()])

bert_checkpoint = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)

MAX_LENGTH = 300
train_X = tokenizer(list(text_all[train_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
train_Y = tags_all[train_ids]
train_dataset = SRaSLRMADataset(train_X['input_ids'], train_X['token_type_ids'], train_X['attention_mask'], train_Y,
                                train_nodes)

val_X = tokenizer(list(text_all[val_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
val_Y = tags_all[val_ids]
val_dataset = SRaSLRMADataset(val_X['input_ids'], val_X['token_type_ids'], val_X['attention_mask'], val_Y, val_nodes)

test_X = tokenizer(list(text_all[test_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
test_Y = tags_all[test_ids]
test_dataset = SRaSLRMADataset(test_X['input_ids'], test_X['token_type_ids'], test_X['attention_mask'], test_Y,
                               test_nodes)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

b_ma_model = SRaSLRMA(bert_checkpoint, CLASSNUM, node_embeddings.vectors,
                      node_embeddings.vector_size)
trainer = pl.Trainer(max_epochs=50, gpus=1, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(b_ma_model, train_dl, val_dl)
trainer.test(b_ma_model, test_dl)
