import pandas as pd

def eval(y_pred, y_true):
    """
    Evaluate prediction accuracy by comparing predicted labels against true labels at various threshold levels.

    Parameters:
        y_pred (pd.DataFrame): DataFrame containing predicted labels.
        y_true (pd.DataFrame): DataFrame containing true labels.

    Returns:
        pd.DataFrame: A DataFrame summarizing the number of accurate predictions at various thresholds.
    """
    # Define thresholds from 0.0 to 2.0, with steps of 0.1
    bench = [i / 10 for i in range(21)]

    good = []  # List to hold the count of good predictions per threshold
    for j in range(len(bench)):
        t = 0  # Temporary variable to count number of predictions meeting the current threshold
        for i in range(y_pred.shape[0]):  # Iterate over each row
            pred_row = y_pred.iloc[i]
            true_row = y_true.iloc[i]
            
            # Get the predicted labels
            pred_labels = [col for col in pred_row.index[:-2] if pred_row[col] == 1]
            # Get the true labels
            true_labels = [col for col in true_row.index[:-2] if true_row[col] == 1]
            
            c = 0  # Count of correctly predicted labels
            for y in pred_labels:
                if y in true_labels:
                    c += 1
            
            # Calculate the similarity metric as the average of precision and recall
            if len(pred_labels) != 0:
                test = c / len(pred_labels) + c / len(true_labels)
                # Assign counts to bins based on the threshold
                if j != len(bench) - 1:
                    if test >= bench[j] and test < bench[j+1]:
                        t += 1
                else:
                    if test >= bench[j]:
                        t += 1
        
        good.append(t)

    # Create a DataFrame with thresholds as indices and counts of good predictions
    good = pd.DataFrame(good, index=bench, columns=['Count'])
    return good
