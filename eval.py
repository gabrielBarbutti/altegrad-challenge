import csv
import pickle
import argparse
import networkx as nx
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MLP
from data import MyDataset


parser = argparse.ArgumentParser(description='ALTEGRAD challenge test file')

# Data arguments
parser.add_argument('--base_data_dir', type=str, default='./data',
                    help='Path to the data folder')

# Precomputed features
parser.add_argument('--base_feats_dir', type=str, default='./saved_feats',
                    help='Path to the pre computed features')
parser.add_argument('--abst_emb_file', type=str, default='abstracts_embeds_bert.pkl',
                    help='File name of the abstracts embeddings')
parser.add_argument('--node_emb_file', type=str, default='nodes_embeds_node2vec.pkl',
                    help='File name of the trained Node2Vec model')
parser.add_argument('--coauthors_file', type=str, default='co-authors_dict.pkl',
                    help='File name of the co-authors dictionary')

# Model choices
parser.add_argument('--hidden_size', type=int, default=100,
                    help='Hidden size of the main classification model')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout used in the main classification model')

parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size')
parser.add_argument('--model_path', type=str, default='./model_weights/model.pt',
                    help='Directory to save model weights')
parser.add_argument('--use_manual_features', action='store_true',
                    help='Flag to use manual features')
parser.add_argument('--subm_csv_path', type=str, default='submission.csv',
                    help='Path location to save csv submission file')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the graph from an edge list file
edgelist_path = Path(args.base_data_dir).joinpath('edgelist.txt')
G = nx.read_edgelist(edgelist_path, delimiter=',',
                     create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Created graph')
print('Number of nodes:', n)
print('Number of edges:', m)

# Load abstract embeddings if it exist, otherwise generate them
abstract_embed_path = Path(args.base_feats_dir).joinpath(args.abst_emb_file)
if abstract_embed_path.is_file():
    print('Loading abstract embeddings')
    f = open(abstract_embed_path, "rb")
    abstracts_embeds = pickle.load(f)
    f.close()
else :
    raise ValueError('Abstracts embedding needs to already be computed')
abstract_feat_size = abstracts_embeds.shape[1]

# Load node embeddings if it exist, otherwise generate them
node_embed_path = Path(args.base_feats_dir).joinpath(args.node_emb_file)
if node_embed_path.is_file():
    print('Loading node embeddings')
    f = open(node_embed_path, "rb")
    nodes_embeds = pickle.load(f)
    f.close()
else :
    raise ValueError('Nodes embedding needs to already be computed')
node_feat_size = nodes_embeds.shape[1]

# Create authors dict
authors_dict = dict()
authors_path = Path(args.base_data_dir).joinpath('authors.txt')
with open(authors_path, 'r', encoding="utf8") as f:
    for line in f:
        node, authors = line.split('|--|')
        authors = authors.split(',')
        authors[-1] = authors[-1][:-1] #removing the \n on last name
        authors_dict[int(node)] = authors

# Load co-authors dict
f = open(Path(args.base_feats_dir).joinpath(args.coauthors_file), "rb")
coauthors_dict = pickle.load(f)
f.close()

# Construct test dataset
node_pairs = list()
test_path = Path(args.base_data_dir).joinpath('test.txt')
with open(test_path, 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))
node_pairs = np.array(node_pairs)

test_set = MyDataset(G, node_pairs, abstracts_embeds, nodes_embeds,
                     authors_dict, coauthors_dict, args.use_manual_features)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = MLP(abstract_feat_size, node_feat_size, args.hidden_size, args.dropout).to(device)
model.load_state_dict(torch.load(args.model_path))

y_pred = []
model.eval()
with torch.no_grad():
    for idx, (x_abstracts, x_nodes, y) in enumerate(test_loader):
        x_abstracts = x_abstracts.to(device)
        x_nodes = x_nodes.to(device)
        y = y.to(device)
        y_pred += (F.softmax(model(x_abstracts, x_nodes), dim=1)[:, 1]).detach().cpu().tolist()



# Write predictions to a file
predictions = zip(range(len(y_pred)), y_pred)
with open(args.subm_csv_path, "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row)

print('Saved submission file', args.subm_csv_path)
