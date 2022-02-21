node_pairs = list()
with open('./test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        node_pairs.append((int(t[0]), int(t[1])))
node_pairs = np.array(node_pairs)

test_set = MyDataset(G, node_pairs, abstracts_embeds, node2vec_model.wv)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


y_pred = []
model.eval()
with torch.no_grad():
    for idx, (x_abstracts, x_nodes, y) in enumerate(test_loader):
        x_abstracts = x_abstracts.to(device)
        x_nodes = x_nodes.to(device)
        y = y.to(device)
        y_pred += (F.softmax(model(x_abstracts, x_nodes), dim=1)[:, 1]).detach().cpu().tolist()



# Write predictions to a file
predictions = zip(range(len(y_pred)), y_pred)
with open("submission_bert_node2vec_mlp_0_7_dropout.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row)
