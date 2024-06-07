import spacy
import pandas as pd
import deepl
import spacy
import re

nlp = spacy.load("en_core_web_md")

def rename(df):
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
    'Zeit_4_4': 'Experience of time',}
    df['Control'] = df.apply(lambda row: 0 if row['KontrSE_4_1'] == 0 and row['Selbstk_4_2'] == 0 else 1, axis=1)
    df['Craft and technology'] = df.apply(lambda row: 0 if row['Handwerk_3_2'] == 0 and row['Technik_3_3'] == 0 else 1, axis=1)
    df.drop(['Handwerk_3_2', 'Technik_3_3','KontrSE_4_1','Selbstk_4_2'], axis=1, inplace=True)
    df = df.rename(columns=translations)
    return df



def embedding(x):
    return nlp(x).vector



def trad(df):
    auth_key = "fc4d8fc2-4e49-44a3-acab-af4cd8013bb1:fx"  
    translator = deepl.Translator(auth_key)

    x=[df['Photo scene'],df['Photo title'],df['Note']]
    df=df.drop(columns=['Photo scene','Photo title','Note'])

    a=[]
    for i in x[0]:
        translation = translator.translate_text(i, target_lang="EN-GB")
        a.append(translation.text)


    b=[]
    for i in x[1]:
        if i!='':
            translation = translator.translate_text(i, target_lang="EN-GB")
            b.append(translation.text)
        else:
            b.append(i)

    c=[]
    for i in x[2]:
        if i!='':
            translation = translator.translate_text(i, target_lang="EN-GB")
            c.append(translation.text)
        else:
            c.append(i)


    df['Photo scene']=a
    df['Photo title']=b
    df['Note']=c

    return df


def tovector(df):    
    
    x=[df['Photo scene'],df['Photo title'],df['Note']]
    df=df.drop(columns=['Photo scene','Photo title','Note'])


    scene=list(map(embedding,x[0]))
    title=list(map(embedding,x[1]))
    note=list(map(embedding,x[2]))

    


    df['Photo scene']=scene
    df['Photo title']=title
    df['Note']=note

    return df




def final(path):
    df1=pd.read_excel(path)
    df=df1.drop(columns=['FotoNr','Bild_vorh','TN','Geschlecht','Gruppe','Hauptoberkategorie',
                         'Hauptsubkategorie','Valenz','Produktivität','Anz_Kodierungen','Control',
                         'Phys_Verä_1','Wohnum_2','Kompetenz_3','Resilienz_4','Einbindung_5','ZeitWohlst_6','Erinnerung_7'])
    df=df.fillna('')
    df=rename(df)
    df=trad(df)
    return df