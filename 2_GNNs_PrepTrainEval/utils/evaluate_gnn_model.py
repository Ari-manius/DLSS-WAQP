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

        _, predicted = torch.max(outputs[mask], 1)  # Get predicted classes for masked nodes

        # Convert to numpy for sklearn metrics
        y_preds = predicted.cpu().numpy()
        y_true = data.y[mask].cpu().numpy()

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

