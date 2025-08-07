import torch

def train_GNN_regression_model(epochs, model, optimizer, criterion, data, early_stopper, scheduler=None, checkpoint_path=None, device=None):
    # Move the data and model to the device
    data = data.to(device)
    model = model.to(device)

    train_losses = []
    val_losses = []

    # Handle compiled models
    is_compiled = hasattr(model, '_orig_mod')
    raw_model = model._orig_mod if is_compiled else model

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        train_loss = criterion(output[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(data)
            val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

        print(f'Epoch: {epoch+1}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss.item())
            else:
                scheduler.step()

        # Early stopping
        early_stopper(val_loss.item(), raw_model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Reload best model checkpoint
    raw_model.load_state_dict(torch.load(checkpoint_path))
    return train_losses, val_losses

def train_GNN_model(epochs, model, optimizer, criterion, data, early_stopper, scheduler=None, checkpoint_path=None, device=None):
    # Move the data and model to the device
    data = data.to(device)
    model = model.to(device)

    train_losses = []
    val_losses = []

    # Handle compiled models
    is_compiled = hasattr(model, '_orig_mod')
    raw_model = model._orig_mod if is_compiled else model

    for epoch in range(epochs):
        # Training with oversampling
        model.train()
        optimizer.zero_grad()
        
        # Create oversampled training data for this epoch
        train_indices = torch.where(data.train_mask)[0]
        train_labels = data.y[train_indices]
        
        # Count samples per class in training set
        class_counts = torch.bincount(train_labels)
        max_count = class_counts.max()
        
        # Oversample minority classes to match majority class
        oversampled_indices = []
        for class_idx in range(len(class_counts)):
            class_mask = train_labels == class_idx
            class_indices = train_indices[class_mask]
            
            if len(class_indices) == 0:
                continue
                
            # Repeat indices to reach max_count
            n_repeats = max_count // len(class_indices)
            remainder = max_count % len(class_indices)
            
            repeated = class_indices.repeat(n_repeats)
            if remainder > 0:
                perm_indices = torch.randperm(len(class_indices))[:remainder]
                repeated = torch.cat([repeated, class_indices[perm_indices]])
            oversampled_indices.append(repeated)
        
        oversampled_train_indices = torch.cat(oversampled_indices)
        
        # Forward pass with oversampled data
        output = model(data)
        train_loss = criterion(output[oversampled_train_indices], data.y[oversampled_train_indices])
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(data)
            val_loss = criterion(output[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

        print(f'Epoch: {epoch+1}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss.item())
            else:
                scheduler.step()

        # Early stopping
        early_stopper(val_loss.item(), raw_model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Reload best model checkpoint
    raw_model.load_state_dict(torch.load(checkpoint_path))
    return train_losses, val_losses
