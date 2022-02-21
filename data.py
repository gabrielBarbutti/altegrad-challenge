import networkx as nx
import numpy as np
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from manual_features import common_authors_publication,get_jaccard_index,get_adamic_adar_index,get_degree

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, G, node_pairs, abstracts_embeds, nodes_embeds,
                 authors_dict, use_manual_features):
        self.abstracts_embeds = abstracts_embeds
        self.nodes_embeds = nodes_embeds
        self.authors_dict = authors_dict
        self.use_manual_features = use_manual_features
        self.node_pairs = node_pairs
        m = len(node_pairs)//2
        self.G = G


        if use_manual_features:
            self.degree = get_degree(self.G)
            self.pagerank = pagerank(self.G)
            self.degree_centrality = nx.algorithms.centrality.degree_centrality(self.G)
            self.jaccard = get_jaccard_index(self.G)
            self.adamic_adar = get_adamic_adar_index(self.G)

        self.y = np.zeros((m*2,))
        self.y[:m] = 1


    def __getitem__(self, index):
        n1, n2 = self.node_pairs[index]

        x_abstracts = np.concatenate((self.abstracts_embeds[int(n1)],
                                      self.abstracts_embeds[int(n2)]))

        x_nodes = np.concatenate((self.nodes_embeds[int(n1)],
                                  self.nodes_embeds[int(n2)]))

        if self.use_manual_features:
            common_auth,common_pub = common_authors_publication(int(n1),int(n2), self.authors_dict)
            authors_emb = np.concatenate((common_auth,common_pub),axis=None)
            x_sum_degree = self.degree[n1] + self.degree[n2]
            x_diff_degree = abs(self.degree[n1] - self.degree[n2])
            if (n1, n2) in self.jaccard:
                x_jaccard = self.jaccard[(n1, n2)]
            else :
                x_jaccard = 0

            if (n1, n2) in self.adamic_adar :
                x_adamic = self.adamic_adar[(n1, n2)]
            else :
                x_adamic = 0

            node_features =  np.concatenate((x_sum_degree, x_diff_degree, x_jaccard, x_adamic), axis=None)
            x_nodes = np.concatenate((x_nodes,node_features),axis=None) #added
            x_nodes = np.concatenate((x_nodes,authors_emb),axis=None) #added

        x_abstracts = torch.from_numpy(x_abstracts).float()
        x_nodes = torch.from_numpy(x_nodes).float()

        return x_abstracts, x_nodes, torch.tensor(self.y[index]).long()

    def __len__(self):
        return len(self.node_pairs)

