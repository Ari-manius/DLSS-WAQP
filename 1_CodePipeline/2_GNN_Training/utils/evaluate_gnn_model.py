# Function to evaluate the model on the validation or test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import numpy as np

# Function to evaluate the model on the validation or test set
def evaluate_gnn_model(data, model, mask_type='val', device=None):

    # Put model and data on the correct device
    model = model.to(device)
    data = data.to(device)

    model.eval()  # Set the model to evaluation mode

    # Determine the appropriate mask
    if mask_type == 'val':
        mask = data.val_mask
    elif mask_type == 'test':
        mask = data.test_mask
    else:
        raise ValueError("mask_type must be 'val' or 'test'")

    # Get predictions with a single forward pass
    with torch.no_grad():
        outputs = model(data)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # If the model returns a tuple, select only the logits

        # Check for NaN values in outputs before prediction
        masked_outputs = outputs[mask]
        if torch.isnan(masked_outputs).any():
            print(f"Warning: {torch.isnan(masked_outputs).sum().item()} NaN values found in model outputs")
            # Replace NaN with very negative values to avoid affecting argmax
            masked_outputs = torch.where(torch.isnan(masked_outputs), torch.tensor(-1e9, device=masked_outputs.device), masked_outputs)
        
        _, predicted = torch.max(masked_outputs, 1)  # Get predicted classes for masked nodes

        # Convert to numpy for sklearn metrics
        y_preds = predicted.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        
        # Final check for NaN in predictions
        if np.isnan(y_preds).any():
            print(f"Error: {np.isnan(y_preds).sum()} NaN values still present in predictions")
            # Remove samples with NaN predictions
            valid_mask = ~np.isnan(y_preds)
            y_preds = y_preds[valid_mask]
            y_true = y_true[valid_mask]
            print(f"Removed {(~valid_mask).sum()} samples with NaN predictions")

    # Calculate accuracy and metrics
    accuracy = accuracy_score(y_true, y_preds)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_true, y_preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_preds))

    results = classification_report(y_true, y_preds, digits=3, output_dict=True)
    matrix = confusion_matrix(y_true, y_preds)

    return results, matrix

def evaluate_gnn_regression(data, model, mask_type='val', device='cpu'):
    model = model.to(device)
    data = data.to(device)
    model.eval()

    if mask_type == 'val':
        mask = data.val_mask
    elif mask_type == 'test':
        mask = data.test_mask
    else:
        raise ValueError("mask_type must be 'val' or 'test'")

    with torch.no_grad():
        outputs = model(data)[mask].cpu().numpy().flatten()
        targets = data.y[mask].cpu().numpy().flatten()

    mse = mean_squared_error(targets, outputs)
    mae = mean_absolute_error(targets, outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, outputs)

    # MAPE & MPE (Avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((targets - outputs) / (targets + epsilon))) * 100
    mpe = np.mean((targets - outputs) / (targets + epsilon)) * 100

    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print("test")
    print(f"MAPE:  {mape:.2f}%")
    print(f"MPE:   {mpe:.2f}%")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MPE": mpe
    }

