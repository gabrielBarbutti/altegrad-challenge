from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pickle

def generate_embedding(model,document):
    return model.infer_vector(document.split())

def generate_node_emb_doc2vec(abstracts, abstract_embed_path, abstracts_path, n, dim_emb):
    print('Building abstract embeddings')
    documents = [TaggedDocument(abstracts[key], [i]) for i, key in enumerate(abstracts)]
    model = Doc2Vec(documents, vector_size=dim_emb, min_count=2)
    print('Doc2Vec model trained')

    abstracts_embeds = np.zeros((n,dim_emb))
    with open(abstracts_path, 'r', encoding="utf8") as f:
        for line in f:
            node, abstract = line.split('|--|')
            abstracts_embeds[int(node),:] = generate_embedding(model,abstract)


    f = open(abstract_embed_path, "wb")
    pickle.dump(abstracts_embeds, f)
    f.close()

    return abstracts_embeds
