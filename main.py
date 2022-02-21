import argparse

import networkx as nx
import csv
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import pickle

from data import MyDataset
from train import train
from model import MLP
from SBERT import generate_abst_emb_sbert
from Node2Vec import generate_node_emb_node2vec
from Doc2Vec import generate_abst_emb_doc2vec
from GAT import generate_node_emb_gat

parser = argparse.ArgumentParser(description='ALTEGRAD challenge main train file')

# Data arguments
parser.add_argument('--base_data_dir', type=str, default='./data',
                    help='Path to the train data folder')

# Precomputed features
parser.add_argument('--base_feats_dir', type=str, default='./saved_feats',
                    help='Path to the pre computed features')
parser.add_argument('--abst_emb_file', type=str, default='abstracts_embeds_bert.pkl',
                    help='File name of the abstracts embeddings')
parser.add_argument('--node_emb_file', type=str, default='nodes_embeds_node2vec.pkl',
                    help='File name of the trained Node2Vec model')
parser.add_argument('--coauthors_file', type=str, default='co-authors_dict.pkl',
                    help='File name of the co-authors dictionary')

# Train arguments
parser.add_argument('--train_percent', type=float, default=0.8,
                    help='Percentage of data to be used in the training set')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--decay_stp_sz', type=int, default=10,
                    help='Learning decay step size')
parser.add_argument('--decay_gamma', type=float, default=0.7,
                    help='Learning decay gamma')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--model_base_dir', type=str, default='./model_weights/',
                    help='Directory to save model weights')

# Model choices
parser.add_argument('--hidden_size', type=int, default=100,
                    help='Hidden size of the main classification model')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout used in the main classification model')

# Choose embedding techinique
parser.add_argument('--abstract_emb_type', type=str, default='sbert', choices=['sbert', 'doc2vec'],
                    help='Choose between abstract embedding types [sbert, doc2vec]')
parser.add_argument('--doc2vec_dim', type=int, default=64,
                    help='Dimension for doc2vec embedding')
parser.add_argument('--node2vec_dim', type=int, default=64,
                    help='Dimension for node2vec embedding')
parser.add_argument('--node_emb_type', type=str, default='node2vec', choices=['node2vec', 'gat'],
                    help='Choose between node embedding types [node2vec, gat]')
parser.add_argument('--use_manual_features', action='store_true',
                    help='Flag to use manual features')



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

# Create abstracts dict
abstracts_path = Path(args.base_data_dir).joinpath('abstracts.txt')
abstracts = dict()
with open(abstracts_path, 'r', encoding="utf8") as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract[:-1] #Removes \n

# Create authors dict
authors_dict = dict()
authors_path = Path(args.base_data_dir).joinpath('authors.txt')
with open(authors_path, 'r', encoding="utf8") as f:
    for line in f:
        node, authors = line.split('|--|')
        authors = authors.split(',')
        authors[-1] = authors[-1][:-1] #removing the \n on last name
        authors_dict[int(node)] = authors

# Load abstract embeddings if it exist, otherwise generate them
abstract_embed_path = Path(args.base_feats_dir).joinpath(args.abst_emb_file)
if abstract_embed_path.is_file():
    print('Loading abstract embeddings')
    f = open(abstract_embed_path, "rb")
    abstracts_embeds = pickle.load(f)
    f.close()
else :
    if args.abstract_emb_type == 'sbert':
        abstracts_embeds = generate_abst_emb_sbert(abstracts, abstract_embed_path)
    elif args.abstract_emb_type == 'doc2vec':
        abstracts_embeds = generate_abst_emb_doc2vec(abstracts, abstract_embed_path,
                                                     abstracts_path, n, args.doc2vec_dim)
    else:
        raise ValueError('Embedding type not supported for the abstracts')
abstract_feat_size = abstracts_embeds.shape[1]

# Load node embeddings if it exist, otherwise generate them
node_embed_path = Path(args.base_feats_dir).joinpath(args.node_emb_file)
if node_embed_path.is_file():
    print('Loading node embeddings')
    f = open(node_embed_path, "rb")
    nodes_embeds = pickle.load(f)
    f.close()
else :
    if args.node_emb_type == 'node2vec':
        nodes_embeds = generate_node_emb_node2vec(G, node_embed_path, n, args.node2vec_dim)
    elif args.node_emb_type == 'gat':
        nodes_embeds = generate_node_emb_gat(edgelist_path, abstracts_embeds, node_embed_path, device)
    else:
        raise ValueError('Embedding type not supported for the nodes')
node_feat_size = nodes_embeds.shape[1]

# Add dimensions in the model for the manual features
if args.use_manual_features:
    print('Using manual features')
    node_feat_size += 3

# Generate positive and negative edges list
node_pairs = np.zeros((2*m, 2))
node_pairs[:m] = np.array(list(G.edges()))

for i in range(m):
    nodes = np.random.randint(0, n, size=(2,))
    # Make sure there isn't an edge between the nodes
    while G.has_edge(nodes[0], nodes[1]):
        nodes = np.random.randint(0, n, size=(2,))
    node_pairs[m+i] = nodes

# Load co-authors dict
f = open(Path(args.base_feats_dir).joinpath(args.coauthors_file), "rb")
coauthors_dict = pickle.load(f)
f.close()

# Create training dataset
dataset = MyDataset(G, node_pairs, abstracts_embeds, nodes_embeds, authors_dict,
                    coauthors_dict, args.use_manual_features)

batch_size = args.batch_size
train_set, val_set = torch.utils.data.random_split(dataset, [int(2*m*args.train_percent), 2*m - int(2*m*args.train_percent)])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

n_epochs = args.n_epochs

model = MLP(abstract_feat_size, node_feat_size, args.hidden_size, args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.NLLLoss()
scheduler = StepLR(optimizer, step_size=args.decay_stp_sz, gamma=args.decay_gamma)

train_losses, val_losses = train(model, device, train_loader, val_loader, optimizer,
                                 criterion, n_epochs, scheduler, args.model_base_dir)
