# README

## Introduction
This repo is the source code for paper _SRaSLR: A Novel Social Relation Aware Service Label Recommendation Model_.

The code is oranized as follows:

```
root
├─data              edges of service social network, service descrptions, and labels
│  └─embed          node embeddings produced by Node2Vec
│          
├─network_only      service social network-only methods
│      
├─sraslr            SRaSLR methods
│      
└─text_only         text-only methods
```

## Data

All data are listed in `data/` folder.

`*.edgelist`: The edge list of network *. Each line represents an edge. The first and second columns are node ids. The third column(if exists) represents the weights.

`dscps_encoded.txt`: Service id and its description.

`titles_encoded.txt`: Service id and its title.

`tags.txt`: The line number(starts from 0) and the content represent a service id and its primary category.

`tags_id.txt`: The line number(starts from 0) and the content represent a service's id and its primary category's id.



## Usage
### Requirements
The following packages are required:

```
torch==1.7.1
pytorch_lightning==1.2.1
numpy==1.19.2
gensim==3.8.3
nltk==3.5
scikit_learn==0.24.1
transformers==4.5.1
```

Pre-trained word vectors are also needed for LSTM and TextCNN models. You can download the word vectors used in the paper from here: [pw-word-vectors.txt](https://drive.google.com/file/d/1USrkYApyVXVqf3RsHgFR5kB9t0wPZj4u/view?usp=sharing).

### Train models
- Clone this project.
- Download the pre-trained word vectors if you need(Only LSTM and TextCNN use the vectors). Put the file into `data/`.
- Go into the root of repo and install the required package listed in `requirements.txt` by:
```commandline
pip install -r requirement.txt
```
- Use `python` command to train and test the model. For example:
```commandline
python sraslr/SRaSLR-ALL-Cat.py
```

