import spacy
import pandas as pd
import deepl
from pandas import DataFrame
from typing import List

# Load a medium-sized English language model from spaCy
nlp = spacy.load("en_core_web_md")

def rename(df: DataFrame) -> DataFrame:
    """
    Renames specified columns in a dataframe according to a predefined mapping and creates new binary columns
    based on existing data.

    Parameters:
        df (DataFrame): Dataframe with specific columns to be renamed and processed.

    Returns:
        DataFrame: The modified dataframe with renamed columns and new binary columns.
    """
    translations = {
    'Fotomotiv': 'Photo scene',
    'Fototitel': 'Photo title',
    'Anmerkung': 'Note',
    'Funkt_Verä_1_1': 'Functional Change',
    'Opt_Verä_1_2': 'Appearance Change',
    'Krankh_1_3': 'Illness',
    'Sterben_1_4': 'Dying or Death',
    'natProz_1_5': 'Aging as natural process',
    'Mobili_2_1': 'Mobility',
    'BauInfra_2_2': 'Infrastructure',
    'Hilfsm_2_3': 'Aids in everyday life',
    'Weiterent_3_1': 'Education and mental stimulation',
    'Toleranz_4_3': 'Openness or Wisdom',
    'Einsamk_5_1': 'Loneliness',
    'Feste_5_2': 'Festivities and traditions',
    'Familie_5_3': 'Family and Friends',
    'Liebe_5_4': 'Love ',
    'Haustier_5_5': 'Pet',
    'Wohlst_6_1': 'Prosperity and old-age poverty',
    'Naturerleb_6_2': 'Experience and enjoyment of Nature',
    'Aktivsein_6_3': 'Being active and healthy behavior',
    'Kultur_6_4': 'Cultural experience and enjoyment',
    'Müßig_6_5': 'Pleasure and recreation',
    'Zeit_4_4': 'Experience of time'}
    # Binary column creation based on conditions
    df['Control'] = df.apply(lambda row: 0 if row['KontrSE_4_1'] == 0 and row['Selbstk_4_2'] == 0 else 1, axis=1)
    df['Craft and technology'] = df.apply(lambda row: 0 if row['Handwerk_3_2'] == 0 and row['Technik_3_3'] == 0 else 1, axis=1)
    df.drop(['Handwerk_3_2', 'Technik_3_3', 'KontrSE_4_1', 'Selbstk_4_2'], axis=1, inplace=True)
    df = df.rename(columns=translations)
    return df




def trad(df: DataFrame, auth_key : str) -> DataFrame:
    """
    Translates the user input columns in a dataframe from German to English using DeepL.

    Parameters:
        df (DataFrame): Dataframe with columns containing text to be translated.
        auth_key : DeepL API authentication key

    Returns:
        DataFrame: Dataframe with translated text.
    """
    translator = deepl.Translator(auth_key)
    x = [df['Photo scene'], df['Photo title'], df['Note']]
    df = df.drop(columns=['Photo scene', 'Photo title', 'Note'])

    a = [translator.translate_text(text, target_lang="EN-GB").text for text in x[0] if text]
    b = [translator.translate_text(text, target_lang="EN-GB").text if text else '' for text in x[1]]
    c = [translator.translate_text(text, target_lang="EN-GB").text if text else '' for text in x[2]]

    df['Photo scene'] = a
    df['Photo title'] = b
    df['Note'] = c

    return df



def final(path: str, auth_key : str) -> DataFrame:
    """
    Processes an Excel file to make it ready to be used for the following parts.

    Parameters:
        path (str): Path to the Excel file.
        auth_key : DeepL API authentication key
        
    Returns:
        DataFrame: Processed dataframe ready for further analysis.
    """
    df1 = pd.read_excel(path)
    df = df1.drop(columns=['FotoNr', 'Bild_vorh', 'TN', 'Geschlecht', 'Gruppe', 'Hauptoberkategorie',
                           'Hauptsubkategorie', 'Anz_Kodierungen', 'Control',
                           'Phys_Verä_1', 'Wohnum_2', 'Kompetenz_3', 'Resilienz_4', 'Einbindung_5', 'ZeitWohlst_6', 'Erinnerung_7'])
    df = df.fillna('')
    df = rename(df)
    df = trad(df,auth_key)
    return df