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

from data import MyDataset
from train import train
from model import MLP
from sbert import generate_abst_emb_sbert
from node2vec import generate_node_emb_node2vec

parser = argparse.ArgumentParser(description='ALTEGRAD challenge main train file')

# Data arguments
parser.add_argument('--base_data_dir', type=str, default='./data',
                    help='Path to the train data folder')

# Precomputed features
parser.add_argument('--base_feats_dir', type=str, default='./saved_feats',
                    help='Path to the pre computed features')
parser.add_argument('--abst_emb_file', type=str, default='abstracts_embeds_bert.pkl',
                    help='File name of the abstracts embeddings')
parser.add_argument('--node_emb_file', type=str, default='saved_model_embed_dim_64',
                    help='File name of the trained Node2Vec model')

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

# Choose embedding techinique
parser.add_argument('--abstract_emb_type', type=str, default='sbert', choices=['sbert', 'doc2vec'],
                    help='Choose between abstract embedding types [sbert, doc2vec]')
parser.add_argument('--doc2vec_dim', type=int, default=64,
                    help='Dimension for doc2vec embedding')
parser.add_argument('--node_emb_type', type=str, default='node2vec', choices=['node2vec', 'gat'],
                    help='Choose between node embedding types [node2vec, gat]')
parser.add_argument('--use_manual_features', action='store_true',
                    help='Flag to use manual features')



args = parser.parse_args()

# Create the graph from an edge list file
G = nx.read_edgelist(f'{args.base_data_dir}/edgelist.txt', delimiter=',',
                     create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Created graph')
print('Number of nodes:', n)
print('Number of edges:', m)

# Create abstracts dict
abstracts = dict()
with open(f'{args.base_data_dir}/abstracts.txt', 'r', encoding="utf8") as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract[:-1] #Removes \n

# Create authors dict
authors_dict = dict()
with open(f'{args.base_data_dir}/authors.txt', 'r', encoding="utf8") as f:
    for line in f:
        node, authors = line.split('|--|')
        authors = authors.split(',')
        authors[-1] = authors[-1][:-1] #removing the \n on last name
        authors_dict[int(node)] = authors


# Load abstract embeddings if it exist, otherwise generate them
abstract_embed_path = Path(args.base_feats_dir+args.abst_emb_file)
if abstract_embed_path.is_file():
    print('Loading abstract embeddings')
    embed_file = open(abstract_embed_path, "rb")
    abstracts_embeds = pickle.load(embed_file)
    embed_file.close()
else :
    if args.abstract_emb_type == 'sbert':
        abstracts_embeds = generate_abst_emb_sbert(abstracts, abstract_embed_path)
        abstract_feat_size = 768
    elif args.abstract_emb_type == 'doc2vec':
        abstracts_embeds = generate_abst_emb_doc2vec(abstracts, abstract_embed_path, args.doc2vec_dim)
        abstract_feat_size = args.doc2vec_dim
    else:
        raise ValueError('Embedding type not supported for the abstracts')

# Load node embeddings if it exist, otherwise generate them
node_embed_path = Path(args.base_feats_dir+args.node_emb_file)
if node_embed_path.is_file():
    print('Loading node embeddings')
    embed_file = open(node_embed_path, "rb")
    nodes_embeds = pickle.load(embed_file)
    embed_file.close()
else :
    if args.node_emb_type == 'node2vec':
        nodes_embeds = generate_node_emb_node2vec(G, node_embed_path, n, dim_emb)
        node_feat_size = args.node2vec_dim
    elif args.node_emb_type == 'gat':
        nodes_embeds = generate_node_emb_gat(CHANGE, node_embed_path, args.doc2vec_dim)
        node_feat_size = 512
    else:
        raise ValueError('Embedding type not supported for the nodes')

# Add dimensions in the model for the manual features
if args.use_manual_features:
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

# Create training dataset
dataset = MyDataset(G, node_pairs, abstracts_embeds, nodes_embeds, args.use_manual_features)

batch_size = args.batch_size
train_set, val_set = torch.utils.data.random_split(dataset, [int(2*m*args.train_percent), 2*m - int(2*m*args.train_percent)])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = args.n_epochs

model = MLP(abstract_feat_size, node_feat_size, args.hidden_size, args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.NLLLoss()
scheduler = StepLR(optimizer, step_size=args.decay_stp_sz, gamma=args.decay_gamma)

train_losses, val_losses = train(model, device, train_loader, val_loader, optimizer, criterion, n_epochs, scheduler)
