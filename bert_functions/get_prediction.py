import torch
import numpy as np

def pred(data_loader, model, device, threshold=0.5):
    # Set the model to evaluation mode to disable dropout and batch normalization layers.
    model.eval()
    all_preds = []  # List to store all prediction arrays
    all_input_ids = []  # List to store all input ID arrays

    # Disable gradient computation to speed up the process and reduce memory usage
    with torch.no_grad():
        # Iterate through each batch in the data loader
        for batch_idx, data in enumerate(data_loader):
            # Extract input IDs, attention masks, and token type IDs from the batch and move them to the specified device
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            # Pass the data through the model to get the outputs
            outputs = model(ids, mask, token_type_ids)
            # Apply sigmoid function to convert logits to probabilities
            probs = torch.sigmoid(outputs)
            # Apply threshold to convert probabilities to binary predictions (0 or 1)
            preds = (probs >= threshold).float()

            # Convert predictions and input IDs to numpy arrays and append to respective lists
            all_preds.append(preds.cpu().numpy())
            all_input_ids.append(ids.cpu().numpy())

    # Stack all prediction arrays and all input ID arrays to form single numpy arrays
    all_preds = np.vstack(all_preds)
    all_input_ids = np.vstack(all_input_ids)

    # Define function to update productivity based on the last two values of the prediction array
    def update_productivity(lst):
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()  # Convert numpy array to list if necessary
        last_two = lst[-2:]  # Extract the last two elements to determine productivity
        lst = lst[:-2]  # Remove the last two elements from the list
        # Determine productivity based on the last two values and append to the list
        if last_two == [0, 0]:
            lst.append(3)
        elif last_two == [1, 0]:
            lst.append(1)
        elif last_two == [0, 1]:
            lst.append(2)
        return lst

    # Define function to update valence based on the third-to-last and second-to-last values of the prediction array
    def update_valence(lst):
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()  # Convert numpy array to list if necessary
        last_two = lst[-3:-1]  # Extract the third-to-last and second-to-last elements to determine valence
        lst.pop(-2)  # Remove the third-to-last element
        lst.pop(-2)  # Remove the second-to-last element
        # Determine valence based on extracted values and append to the list
        if last_two == [0, 0]:
            lst.append(3)
        elif last_two == [1, 0]:
            lst.append(1)
        elif last_two == [0, 1]:
            lst.append(2)
        else:
            lst.append(4)
        return lst

    # Process each prediction array to update productivity and valence
    new_pred = []
    for x in all_preds:
        pred = update_productivity(x)  # Update productivity
        pred = update_valence(pred)  # Update valence
        new_pred.append(pred)  # Append the updated prediction to the list

    # Return the all input IDs and the new predictions with updated productivity and valence
    return all_input_ids, new_pred
