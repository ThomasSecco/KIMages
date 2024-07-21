from models import description
import pickle

def desc(df, path):
    """
    Process a DataFrame to add a description to each row based on the image associated with it.
    The description process uses an image processing model to evaluate and describe images.

    Parameters:
        df (DataFrame): A pandas DataFrame that includes a column 'Bildbez' with image filenames.
        path : Path to the folder containing images.
    Returns:
        DataFrame: The original DataFrame augmented with a new 'Description' column.
    """
    desc = []

    # Attempt to load progress from a pickle file to resume from the last checkpoint.
    try:
        with open('desc_progress.pkl', 'rb') as f:
            desc = pickle.load(f)
    except FileNotFoundError:
        # If no progress file exists, start with an empty list.
        pass

    # Start processing from where the last progress was saved.
    i = len(desc)  # Index to track the number of processed entries.
    for x in df['Bildbez'][len(desc):]:  # Only process unprocessed entries.
        path = path+ f'/{x}.jpg'  # Construct the full path to the image.
        try:
            # Use the description module to process the image and get results.
            res = description.results(path)
            # Compare results from different models to categorize image content.
            c = description.compare_lists(res[1], res[2], res[3])
            # Score the descriptions based on how well they match across models.
            f = description.score(c[0], c[1], c[2], res[0])
            if f[1] > 50:  # If the score is above 50%, use detailed description.
                desc.append(f[0])
            else:
                # If the score is low, append an empty description.
                desc.append([''])
        except FileNotFoundError:
            # Handle cases where the image file is not found.
            desc.append([''])

        # Save progress to a pickle file after each image is processed.
        with open('desc_progress.pkl', 'wb') as f:
            pickle.dump(desc, f)

        i += 1  # Increment processed count.

        # Print progress as a percentage of total rows to process.
        print(f'Progress: {100 * i / df.shape[0]:.2f}%')

    # Add the compiled descriptions as a new column in the DataFrame.
    df['Description'] = desc
    return df
