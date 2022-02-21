from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def generate_abst_emb_sbert(abstracts, abstract_embed_path):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    print('Tokenizing data')
    documents = []
    documents_len = []
    for paragraph in tqdm(list(abstracts.values())):
        sentences = tokenizer.tokenize(paragraph)
        documents_len.append(len(sentences))
        documents += sentences

    print('Building abstract embeddings. This step can take a long time')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    raw_embeds = model.encode(documents, batch_size=256, show_progress_bar=True)

    abstracts_embeds = np.zeros((n, 768))
    total_l = 0
    for idx, l in enumerate(documents_len):
        if l > 0:
            abstracts_embeds[idx] = raw_embeds[total_l:(total_l+l)].mean(axis=0)
            total_l += l

    embed_file = open(abstract_embed_path, "wb")
    pickle.dump(abstracts_embeds, embed_file)
    embed_file.close()

    return abstracts_embeds
