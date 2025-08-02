from torch.optim.lr_scheduler import SequentialLR, LinearLR
import torch

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
        # Training
        model.train()
        optimizer.zero_grad()

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
