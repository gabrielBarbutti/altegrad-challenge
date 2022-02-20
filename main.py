import argparse

import networkx as nx


parser = argparse.ArgumentParser(description='ALTEGRAD challenge main train file')

# Data arguments
parser.add_argument('--base_data_dir', type=str, default='./data',
                    help='Path to the train data folder')

# Precomputed features
parser.add_argument('--base_feats_dir', type=str, default='./saved_feats',
                    help='Path to the pre computed features')
parser.add_argument('--abst_emb_file', type=str, default='abstracts_embeds_bert.pkl',
                    help='File name of the abstracts embeddings')
parser.add_argument('--node2vec_file', type=str, default='saved_model_embed_dim_64',
                    help='File name of the trained Node2Vec model')

# Train arguments
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--model_base_dir', type=str, default='./model_weights/',
                    help='Directory to save model weights')


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
abstract_embed_path = Path(args.base_feats_dir+args.abstract_embed_path)
if abstract_embed_path.is_file():
    print('Loading abstract embeddings')
    embed_file = open(abstract_embed_path, "rb")
    abstracts_embeds = pickle.load(embed_file)
    embed_file.close()
else :
    embed_file = generate_abst_emb(abstracts)

# Load pretrained Node2Vec model
node2vec_model = Word2Vec.load("saved_model_embed_dim_64")

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
dataset = MyDataset(G, node_pairs, abstracts_embeds, node2vec_model.wv)

batch_size = 512
train_set, test_set = torch.utils.data.random_split(dataset, [int(2*m*0.8), 2*m - int(2*m*0.8)])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

###CONTINUE WORINKING FROM HERE
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 20
model = MLP(768, 67).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

train_losses_aux, test_losses_aux = train(model, device, train_loader, test_loader, optimizer, criterion, n_epochs, scheduler)
