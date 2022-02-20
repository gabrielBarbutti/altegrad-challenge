# MVA Altegrad challenge 2021

Team members :
* Gabriel Baker : gabriel.baker@telecom-paris.fr
* Yujin Cho : yujin.cho@ens-paris-saclay.fr
* Fabien Merceron : fabien.merceron@ens-paris-saclay.fr

This work intends to study different approaches of leveraging text and graph structure information to predict missing links in a scientific collaboration network.
The problem was proposed in the ALTEGRAD course and is the objective of a Kaggle challenge.

## Experiments


| Abstract Embedding | Node Embedding | Manual features | Train | Val | Test |
| ------------- | ------------- | ------------- | --------- | --------- |--------- |
| Doc2Vec | Node2Vec | No  | 0.078 | 0.052 | 0.225 |
| SBERT | GAT | No  | 0.087 | 0.082 | 0.143 |
| SBERT | Node2Vec | No  | 0.061 | 0.052 | 0.146 |
| SBERT | Node2Vec | Yes  | TBD | TBD | TBD |
