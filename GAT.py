import torch
import pickle
import numpy as np
from torch import nn
import torch.nn.functional as F

import dgl
from dgl import function as fn
from dgl.nn.pytorch import GATConv


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

def construct_negative_graph(graph, k, device):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,)).to(device)
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

class GraphModelGAT(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size1,hidden_size2, num_head1, num_head2, nonlinearity):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_size, hidden_size1, num_head1,activation=nonlinearity))
        for i in range(n_layers - 2):
            self.layers.append(GATConv(hidden_size1*num_head1, hidden_size1, num_head1, activation=nonlinearity))
        self.layers.append(GATConv(hidden_size1 * num_head1, hidden_size2, num_head2, activation=nonlinearity))
        self.pred = DotProductPredictor()

    def forward(self, g, neg_g, x, return_emb=False):
        bs = x.shape[0]
        for i, layer in enumerate(self.layers):
            x = layer(g, x.reshape(bs, -1))
        #x = x.reshape(bs, -1)
        #print(outputs.shape)
        if return_emb:
            return x
        return self.pred(g, x), self.pred(neg_g, x)

def generate_node_emb_gat(edgelist_path, abstracts_embeds, node_embed_path, device):
    edge_list = np.genfromtxt(edgelist_path, delimiter=',', dtype=int)

    # Edges are directional in DGL; Make them bi-directional.
    u = torch.from_numpy(np.concatenate([edge_list[:, 0], edge_list[:, 1]])).to(device)
    v = torch.from_numpy(np.concatenate([edge_list[:, 1], edge_list[:, 0]])).to(device)
    # Construct a DGLGraph
    G = dgl.graph((u, v))
    
    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - pos_score + neg_score).clamp(min=0).mean()
    
    node_features = G.ndata['feat']
    n_features = node_features.shape[1]
    k = 1
    model = GraphModelGAT(n_layers=3, input_size=abstracts_embeds.shape[1], hidden_size1=256,
                          hidden_size2=512, num_head1= 4, num_head2 = 1, nonlinearity=F.elu).to(device)
    
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        negative_graph = construct_negative_graph(G, k, device)
        pos_score, neg_score = model(G, negative_graph, node_features.float())
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

    embeddings = model(G, negative_graph, node_features.float(), return_emb=True)

    f = open( node_embed_path, "wb")
    pickle.dump(embeddings[:, 0], f)
    f.close()

    return embeddings
