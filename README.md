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
To facilitate the run of our model we created a colab notebook available [here](https://colab.research.google.com/drive/1SapsWlFHveQJoVZ9UJ6B1MmEh0X-3hQL?usp=sharing).

But you can also locally train the model as explained below.

### Prepare the environment
We recommend the download of the following folders with the precomputed features and the dataset.

- ./data [Google drive link](https://drive.google.com/drive/folders/1Li1ycoCGqvFARk8992R5UTL2D56cn6pg?usp=sharing)
- ./saved\_feats [Google drive link](https://drive.google.com/drive/folders/1bqoZ9bxdFn7iLexoQ_Em69EvdzC4KdD0?usp=sharing)

Please install the following not so common libraries

```Python
!pip install -qU sentence-transformers
!pip install -q node2vec
!pip install -q --upgrade gensim
!pip install -q dgl
```

### Training
Use this command to reproduce our best model

```
python main.py --use_manual_features
```

### Evaluate
To create the submission file run the following

```
python eval.py --use_manual_features
```

## Experiments / Results
In order to leverage the most of our three different sources of data (abstracts, authors and edges connections), we tested different strategies of feature extraction for each source and different combinations of the generated features.

The table  summarizes the results of our best input combinations, but not of the best models, as there were small variations that produced better models than the first 3 lines (the last line was indeed our best model). An example of this are the models of the second and third rows (SBERT \& GAT / SBERT \& Doc2Vec), which performed better when training with the whole dataset, but we decided to present the results of all models with the division of training and validation set to have a fairer comparison.

We reached top 5 in kaggle challenge private leaderboard with 0.09725

| Abstract Embedding | Node Embedding | Manual features | Train | Val | Test |
| ------------- | ------------- | ------------- | --------- | --------- |--------- |
| Doc2Vec | Node2Vec | No  | 0.078 | 0.052 | 0.225 |
| SBERT | GAT | No  | 0.087 | 0.082 | 0.143 |
| SBERT | Node2Vec | No  | 0.061 | 0.052 | 0.146 |
| SBERT | Node2Vec | Yes  | 0.059 | 0.048 | 0.104 |


Result table : Negative - log likelihood obtained after training the network and choosing the model with the smallest validation loss. Test score is calculated with approximately 50% of the test dataset
