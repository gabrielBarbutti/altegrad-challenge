from gensim.models import Word2Vec
from node2vec import Node2Vec
import numpy as np
import pickle

def generate_node_emb_node2vec(G, node_embed_path, n, dim_emb):
    node2vec_embed = np.zeros((n,dim_emb))

    node2vec = Node2Vec(G, dimensions=dim_emb, walk_length=30, num_walks=5, workers=1)
    # Embed nodes
    node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)

    for i in range(n):
        node2vec_embed[i,:] = node2vec_model.wv[str(i)]

    embed_file = open(node_embed_path, "wb")
    pickle.dump(node2vec_embed, embed_file)
    embed_file.close()

    return node2vec_embed
