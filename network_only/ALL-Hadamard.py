import gensim
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


class ALLHadamardDataset(torch.utils.data.Dataset):

    def __init__(self, targets, node_ids1, node_ids2, transform=None):
        self.targets = np.array(targets, dtype=np.int32)
        self.node_ids1 = np.array(node_ids1, dtype=np.int32)
        self.node_ids2 = np.array(node_ids2, dtype=np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        node_id = self.node_ids1[index]
        node_id2 = self.node_ids2[index]
        return torch.as_tensor(target).type(torch.LongTensor), torch.as_tensor(node_id).type(
            torch.LongTensor), torch.as_tensor(node_id2).type(torch.LongTensor)


class ALLHadamardClassifier(pl.LightningModule):
    def __init__(self, classes, node_embedding, node_embedding2, node_embedding_size, dense_dropout=0.3, **kwargs):
        super().__init__()
        # Load model
        self.node_encoder = nn.Embedding.from_pretrained(torch.FloatTensor(node_embedding),
                                                         freeze=False)  # don't freeze
        self.node_encoder2 = nn.Embedding.from_pretrained(torch.FloatTensor(node_embedding2), freeze=False)
        self.decoder = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=node_embedding_size, out_features=node_embedding_size),
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=node_embedding_size, out_features=classes),
        )

    def forward(self, node_ids1, node_ids2):
        node_output = self.node_encoder(node_ids1)
        node_output2 = self.node_encoder2(node_ids2)
        output = node_output * node_output2
        output = self.decoder(output)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-8)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        target, node_id1, node_id2 = train_batch
        output = self.forward(node_id1, node_id2)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = loss.mean()
        self.log('train loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        target, node_id, node_id2 = val_batch
        output = self.forward(node_id, node_id2)
        loss = F.cross_entropy(output, target)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        target, node_id, node_id2 = val_batch
        output = self.forward(node_id, node_id2)
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

# load graph model
PATH = 'data/embed/ma_embedding_100.txt'
node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)

PATH2 = 'data/embed/mm_aa_freq_100.txt'
node_embeddings2 = gensim.models.KeyedVectors.load_word2vec_format(PATH2, binary=False)

X = list(range(4099))

train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)

train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)

train_nodes = [node_embeddings.vocab[str(n)].index for n in train_ids]
val_nodes = [node_embeddings.vocab[str(n)].index for n in val_ids]
test_nodes = [node_embeddings.vocab[str(n)].index for n in test_ids]

train_nodes2 = [node_embeddings2.vocab[str(n)].index for n in train_ids]
val_nodes2 = [node_embeddings2.vocab[str(n)].index for n in val_ids]
test_nodes2 = [node_embeddings2.vocab[str(n)].index for n in test_ids]

tags_all = np.array([int(line.strip()) for line in open('data/tags_id.txt').readlines()])

MAX_LENGTH = 300
train_Y = tags_all[train_ids]
train_dataset = ALLHadamardDataset(train_Y, train_nodes, train_nodes2)

val_Y = tags_all[val_ids]
val_dataset = ALLHadamardDataset(val_Y, val_nodes, val_nodes2)

test_Y = tags_all[test_ids]
test_dataset = ALLHadamardDataset(test_Y, test_nodes, test_nodes2)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)

model = ALLHadamardClassifier(CLASSNUM, node_embeddings.vectors, node_embeddings2.vectors,
                              node_embeddings.vector_size)
trainer = pl.Trainer(max_epochs=50, gpus='0', callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
