class MLP(nn.Module):
    def __init__(self, abstract_emb_size, node_emb_size, hidden_size, dropout):
        super(MLP, self).__init__()
        self.abstracts_layers = nn.Sequential(nn.Linear(abstract_emb_size*2, hidden_size),
                                              nn.ReLU(),
                                              nn.BatchNorm1d(num_features=hidden_size),
                                              nn.Dropout(p=dropout))
        self.nodes_layers = nn.Sequential(nn.Linear(node_emb_size*2, hidden_size),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(num_features=hidden_size),
                                          nn.Dropout(p=dropout))
        self.final_layers = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(num_features=2*hidden_size),
                                          nn.Dropout(p=dropout),
                                          
                                          nn.Linear(2*hidden_size, 2),
                                          nn.LogSoftmax(dim=1))
        
    def forward(self, x_abstracts, x_nodes):
        x_abstracts = self.abstracts_layers(x_abstracts)
        x_nodes = self.nodes_layers(x_nodes)
        x = torch.cat((x_abstracts, x_nodes), axis=1)
        x = self.final_layers(x)
        return x
