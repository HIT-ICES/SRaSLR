import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
import gensim


class MADataset(torch.utils.data.Dataset):

    def __init__(self, targets, node_ids=None, transform=None):
        self.node_ids = np.array(node_ids, dtype=np.int32)
        self.targets = np.array(targets, dtype=np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        node_id = self.node_ids[index]
        target = self.targets[index]
        return torch.as_tensor(target).type(torch.LongTensor), torch.as_tensor(node_id).type(torch.LongTensor)


class MAGraphClassifier(pl.LightningModule):
    def __init__(self, classes, node_embedding, node_embedding_size, dense_dropout=0.5, **kwargs):
        super().__init__()
        # Load model
        self.node_encoder = nn.Embedding.from_pretrained(torch.FloatTensor(node_embedding),
                                                         freeze=False)  # don't freeze
        self.decoder = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=node_embedding_size, out_features=node_embedding_size),
            nn.Dropout(dense_dropout),
            nn.Linear(node_embedding_size, out_features=classes),
        )

    def forward(self, node_ids):
        output = self.node_encoder(node_ids)
        output = self.decoder(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-8)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        target, node_id = train_batch
        output = self.forward(node_id)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        target, node_id = val_batch
        output = self.forward(node_id)
        loss = F.cross_entropy(output, target)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        target, node_id = val_batch
        output = self.forward(node_id)
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

PATH = 'data/embed/ma_embedding_100.txt'
node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)

X = list(range(4099))

train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)

train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)

train_nodes = [node_embeddings.vocab[str(n)].index for n in train_ids]
val_nodes = [node_embeddings.vocab[str(n)].index for n in val_ids]
test_nodes = [node_embeddings.vocab[str(n)].index for n in test_ids]

tags_all = np.array([int(line.strip()) for line in open('data/tags_id.txt').readlines()])

train_Y = tags_all[train_ids]
train_dataset = MADataset(train_Y, train_nodes)

val_Y = tags_all[val_ids]
val_dataset = MADataset(val_Y, val_nodes)

test_Y = tags_all[test_ids]
test_dataset = MADataset(test_Y, test_nodes)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

ma_model = MAGraphClassifier(CLASSNUM, node_embeddings.vectors, node_embeddings.vector_size)
trainer = pl.Trainer(max_epochs=50, gpus='0', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(ma_model, train_dl, val_dl)
trainer.test(ma_model, test_dl)
