def train_classification(config: object, best_loss = np.inf, warm_up: int = 0) -> tuple[object, str]:

    '''
    Training loop for classification.

    Args:
        config (object): Input parameters for training. 
        best_loss (int): If specified, only models with a better loss will be saved. Default: 0.
        warmup_up (int): Takes the number og warmup up epochs specified. Default: 0 (No warm up).

    Returns:
        tuple (object, str): A tuple with the trained model, and str for txt report.
    '''
    #Loading model
    model = config.model
    model_name = config.model_name
    device = config.device
    path_models = config.path_models

    # Data
    X_train_freq, X_val_freq, X_test_freq = config.data[0],config.data[1],config.data[2]
    Y_train, Y_val, Y_test = config.y_data[0], config.y_data[1], config.y_data[2]
    
    # Creating the dataloader
    train_dataloader = get_dataloader(batch_size = config.batch_size, shuffle = config.shuffle, data_class = config.data_class[0])
    val_dataloader = get_dataloader(batch_size = 100, shuffle = config.shuffle, data_class = config.data_class[1])
    test_dataloader = get_dataloader(batch_size = 100, shuffle = config.shuffle, data_class = config.data_class[2])

    optimizer = config.optimizer()
    scheduler = config.lr_scheduler(optimizer)
    n_pos, n_neg = torch.sum(Y_train), torch.sum(Y_train==0)
    criterion = config.criterion(weights=n_neg/n_pos)

    patience_count = 1 # initialize patience counter
    history = pd.DataFrame(columns=['train', 'val', 'test'])# used for training, validation, and test loss
    
    start_time = time.time() # Used to time training run

    if warm_up:
        print(f"Warm up with {warm_up} steps.")
        optimizer_warmup = torch.optim.Adam(model.parameters(), lr=config.lr*1e-8)
        for epoch in tqdm(range(1, warm_up + 1)):
            model = model.train()

            for batch_in,labels in train_dataloader:
                labels = labels.to(device).float()
                optimizer_warmup.zero_grad()
                seq_pred = model(batch_in)
                loss = criterion(seq_pred, labels)
                loss.backward()
                optimizer_warmup.step()

    for epoch in tqdm(range(1, config.n_epochs + 1)):
        
        model.train()

        # lists for losses
        train_losses = []
        val_losses = []
        test_losses = []

        for batch_in,labels in train_dataloader:
            labels = labels.to(device).float()
            optimizer.zero_grad()
            seq_pred = model(batch_in)
            loss = criterion(seq_pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():

            # validation steps
            for batch_in,labels in val_dataloader:
                labels = labels.to(device).float()
                seq_pred = model(batch_in)
                loss = criterion(seq_pred, labels)
                val_losses.append(loss.item())

            # test steps
            for batch_in,labels in test_dataloader:
                labels = labels.to(device).float()
                seq_pred = model(batch_in)
                loss = criterion(seq_pred, labels)
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        test_loss = np.mean(test_losses)

        # Saving loss for each epoch in dictionary
        history.loc[len(history)] = [train_loss, val_loss, test_loss]

        # Print for each epoch
        print(f'\nEpoch {epoch}: Train loss: {train_loss:.4f} {" "*6} Val loss: {val_loss:.4f} {" "*6} Test loss: {test_loss:.4f} {" "*6}')

        if config.lr_scheduler_name == 'ReduceLROnPlateau':
            scheduler.step(val_loss) #automatic lr 
        elif scheduler is not None:
            scheduler.step() #automatic lr 

        if val_loss < (1-config.delta)*best_loss: #Only saved if validation loss is improved by delta
            print(f'\nValidation loss decreased ({best_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
            torch.save(model.state_dict(),path_models+model_name)
            best_loss = val_loss
            patience_count = 1
        else:
            patience_count+=1 # If model is not improved patience + 1


        if patience_count>config.patience:
            break

    # Calculate training time
    end_time = time.time()
    total_time = end_time-start_time
    m, s = divmod(total_time, 60)

    model.load_state_dict(torch.load(path_models+model_name))

    #Saving training information in txt report
    txt_output = config.txt_output
    txt_output += '\n\nClassification Run 1\n'
    txt_output += 'Loss: \n'
    txt_output += 'Train: ' + str(f"{history['train'].iloc[-1]:.4f}") + '\n'
    txt_output += 'Val: ' + str(f"{history['val'].iloc[-1]:.4f}") + '\n'
    txt_output += 'Test: ' + str(f"{history['test'].iloc[-1]:.4f}") + '\n'
    txt_output += 'Val loss save: ' + str(f"{best_loss:.4f}") + '\n'
    txt_output += 'Epochs: ' + str(epoch) + '\n'
    txt_output += 'Time: ' + f'{m:02.0f}:{s:02.0f}' + '\n'
    epochs = range(1, epoch+1)

    #saving losses to pickle
    history.to_pickle(f'{config.folder_runs}{model_name}_r{config.run}.pkl')

    #Plotting loss
    plt.figure()
    plt.plot(epochs, history['train'], 'b', label='Training Loss')
    plt.plot(epochs, history['val'], 'g', label='Validation Loss')
    plt.plot(epochs, history['test'], 'r', label='Testing Loss')
    plt.title('Loss for ' + model_name)
    plt.legend()

    plt.savefig(f'{config.folder_runs}{model_name}_r{config.run}.png')
    #plt.show()

    return model, txt_output
