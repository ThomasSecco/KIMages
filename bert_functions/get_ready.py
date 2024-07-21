import ast
import pandas as pd

def ready(df):
    """
    Prepare a DataFrame for analysis by transforming and creating new columns based on existing data.

    Parameters:
        df (pd.DataFrame): The input DataFrame with various columns, including 'Valenz' and 'Produktivität'.

    Returns:
        tuple: A tuple containing the modified DataFrame and a DataFrame of true labels (y_true).
    """

    # Define a function to categorize 'Valenz' into 'Positive'
    def create_positive(value):
        # Return 1 for values that indicate a positive sentiment (either 1 or 4)
        if value == 1 or value == 4:
            return 1
        else:
            return 0

    # Define a function to categorize 'Valenz' into 'Negative'
    def create_negative(value):
        # Return 1 for values that indicate a negative sentiment (either 2 or 4)
        if value == 2 or value == 4:
            return 1
        else:
            return 0

    # Apply the functions above to create 'Positive' and 'Negative' columns based on 'Valenz'
    df['Positive'] = df['Valenz'].apply(create_positive)
    df['Negative'] = df['Valenz'].apply(create_negative)

    # Define a function to categorize 'Produktivität' into 'Productive'
    def create_productive(value):
        # Return 1 for productive value (1)
        if value == 1:
            return 1
        else:
            return 0

    # Define a function to categorize 'Produktivität' into 'Unproductive'
    def create_non_productive(value):
        # Return 1 for unproductive value (2)
        if value == 2:
            return 1
        else:
            return 0

    # Apply the functions to create 'Productive' and 'Unproductive' columns based on 'Produktivität'
    df['Productive'] = df['Produktivität'].apply(create_productive)
    df['Unproductive'] = df['Produktivität'].apply(create_non_productive)

    # Define a function to convert string representations of lists back into actual lists
    def convert_to_list(value):
        # Attempt to evaluate the string as a list, handle exceptions if evaluation fails
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    # Apply the conversion function to the 'Description' column
    df['Description'] = df['Description'].apply(convert_to_list)

    # Fill missing values in specific columns with empty strings or zeros
    for col in df.columns:
        if col in ['Photo scene', 'Photo title', 'Note', 'Description']:
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)

    # Remove columns not used in the predictive model and keep y_true labels
    y_true = df.drop(columns=['Bildbez', 'Photo scene', 'Photo title', 'Note', 'Description'])

    # Concatenate text fields into a single 'combined' column for further text analysis
    df['combined'] = df['Photo scene'] + '. ' + df['Photo title'] + '. ' + df['
