def train(model, device, train_loader, test_loader, optimizer, criterion,
          n_epochs, scheduler, save_model, model_path):
    train_losses = []
    test_losses = []
    F_best_test_loss = 10
    for epoch in range(n_epochs):
        with tqdm(total=len(train_loader), unit_scale=True, postfix={'train loss':0.0, 'test loss':0.0},
                        desc="Epoch : %i/%i" % (epoch, n_epochs-1), ncols=100) as pbar:
            model.train()
            total_loss = 0
            test_loss = 0
            for idx, (x_abstracts, x_nodes, y) in enumerate(train_loader):
                x_abstracts = x_abstracts.to(device)
                x_nodes = x_nodes.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x_abstracts, x_nodes)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_loss = total_loss/(idx+1)
                pbar.set_postfix({'train loss':train_loss, 'test loss':test_loss})
                pbar.update(1)
            
            
            if test_loader is not None:
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for (x_abstracts, x_nodes, y) in test_loader:
                        x_abstracts = x_abstracts.to(device)
                        x_nodes = x_nodes.to(device)
                        y = y.to(device)
                        y_pred = model(x_abstracts, x_nodes)
                        test_loss += criterion(y_pred, y)

                test_loss /= len(test_loader)
                test_loss = test_loss.item()
                pbar.set_postfix({'train loss':train_loss, 'test loss':test_loss})
                test_losses.append(test_loss)
                if test_loss<=F_best_test_loss and save_model:
                    F_best_test_loss = test_loss
                    torch.save(model.state_dict(), model_path)
                    print("Model saved")
                
                model.train()
                
             
            train_losses.append(train_loss)
            scheduler.step()
            
    return train_losses, test_losses
