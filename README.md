# MVA Altegrad challenge 2021

## Link prediction

Team members :
* Gabriel Baker : gabriel.baker@telecom-paris.fr
* Yujin Cho : yujin.cho@ens-paris-saclay.fr
* Fabien Merceron : fabien.merceron@ens-paris-saclay.fr

This work intends to study different approaches of leveraging text and graph structure information to predict missing links in a scientific collaboration network.
The problem was proposed in the ALTEGRAD course and is the objective of a Kaggle challenge.

## Dataset
It contains different sources of data (abstract, authors and edge connections)

## Runing the script

```Python
!python main.py --abstract_emb_type sbert
```

Please install following libraries
```Python
!pip install -qU sentence-transformers
!pip install node2vec
!pip install --upgrade gensim
```

## Experiments / Results
In order to leverage the most of our three different sources of data (abstracts, authors and edges connections), we tested different strategies of feature extraction for each source and different combinations of the generated features.

The table  summarizes the results of our best input combinations, but not of the best models, as there were small variations that produced better models than the first 3 lines (the last line was indeed our best model). An example of this are the models of the second and third rows (SBERT \& GAT / SBERT \& Doc2Vec), which performed better when training with the whole dataset, but we decided to present the results of all models with the division of training and validation set to have a fairer comparison.

| Abstract Embedding | Node Embedding | Manual features | Train | Val | Test |
| ------------- | ------------- | ------------- | --------- | --------- |--------- |
| Doc2Vec | Node2Vec | No  | 0.078 | 0.052 | 0.225 |
| SBERT | GAT | No  | 0.087 | 0.082 | 0.143 |
| SBERT | Node2Vec | No  | 0.061 | 0.052 | 0.146 |
| SBERT | Node2Vec | Yes  | 0.059 | 0.048 | 0.104 |
