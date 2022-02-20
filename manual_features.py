def common_authors_publication(node_0, node_1):
    '''
    give back the number document the common authors released
    '''
    #WHERE IS THIS FILE CREATED
    co_aut_dict_file = open("co-authors_dict.pkl", "rb")
    dict_coauthors = pickle.load(co_aut_dict_file)
    co_aut_dict_file.close()

    list_authors_0 = authors_dict[int(node_0)]
    list_authors_1 = authors_dict[int(node_1)]
    common = list(set(list_authors_0).intersection(list_authors_1))
    nb_copubli = 0
    for author0 in list_authors_0 :
      for author1 in list_authors_1 :
        if author0!= author1:
            if author1 in dict_coauthors[author0]:
              nb_copubli += dict_coauthors[author0][author1]

    return len(common), nb_copubli

def check_autocitation(node_0, node_1):
    '''
    Check if at least 1 author is in common in a pair of articles

    Returns a binary variable =0 if none author in common and =1 if any
    '''
    list_authors_0 = authors_dict[int(node_0)]
    list_authors_1 = authors_dict[int(node_1)]
    if not list(set(list_authors_0).intersection(list_authors_1)):
        autocitation = 0
    else : 
        autocitation = 1
    return autocitation      
      
def overlap_authors(node_0, node_1):
    '''
    Check if at least 1 author is in common in a pair of articles
    
    Returns a binary variable =0 if none author in common and =1 if any
    '''
    list_authors_0 = authors_dict[int(node_0)]
    list_authors_1 = authors_dict[int(node_1)]
    overlap = list(set(list_authors_0).intersection(list_authors_1))

    return len(overlap)

def get_degree(G):
    dict_degrees = {}
    for node in list(G.nodes()):
        dict_degrees[node] = G.degree(node)
    return dict_degrees

def get_adamic_adar_index(G):
    dict_adamic_adar_index = {}
    for edge in list(G.edges()):
        n1, n2 = edge[0], edge[1]
        _, _, adar = list(nx.adamic_adar_index(G, [(n1, n2)]))[0]
        dict_adamic_adar_index[(n1,n2)] = adar
    return dict_adamic_adar_index

def get_jaccard_index(G):
    dict_jaccard = {}
    for edge in list(G.edges()):
        n1, n2 = edge[0], edge[1]
        _, _, jacc = list(nx.jaccard_coefficient(G, [(n1, n2)]))[0]
        dict_jaccard[(n1,n2)] = jacc
    return dict_jaccard

def scaled_dict(dictionnary):
    mean = np.mean(list(dictionnary.values()))
    std = np.std(list(dictionnary.values()))
    for key, value in dictionnary.items():
        dictionnary[key] = (value-mean)/std
    return dictionnary
