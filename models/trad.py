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
    'Produktivität': 'Productivity',
    'Phys_Verä_1': 'Physical Change',
    'Wohnum_2': 'Living Conditions',
    'Kompetenz_3': 'Competence',
    'Resilienz_4': 'Resilience',
    'Einbindung_5': 'Integration',
    'ZeitWohlst_6': 'Time Wellbeing',
    'Erinnerung_7': 'Memory',
    'Funkt_Verä_1_1': 'Functional Change',
    'Opt_Verä_1_2': 'Optical Change',
    'Krankh_1_3': 'Sickness',
    'Sterben_1_4': 'Dying',
    'natProz_1_5': 'Natural Process',
    'Mobili_2_1': 'Mobility',
    'BauInfra_2_2': 'Building Infrastructure',
    'Hilfsm_2_3': 'Assistive Devices',
    'Weiterent_3_1': 'Development',
    'Handwerk_3_2': 'Craftsmanship',
    'Technik_3_3': 'Technology',
    'KontrSE_4_1': 'Self-Contradiction',
    'Selbstk_4_2': 'Self Control',
    'Toleranz_4_3': 'Tolerance',
    'Einsamk_5_1': 'Loneliness',
    'Feste_5_2': 'Celebrations',
    'Familie_5_3': 'Family',
    'Liebe_5_4': 'Love',
    'Haustier_5_5': 'Pet',
    'Wohlst_6_1': 'Wealth',
    'Naturerleb_6_2': 'Nature Experience',
    'Aktivsein_6_3': 'Activity',
    'Kultur_6_4': 'Culture',
    'Müßig_6_5': 'Idle',
    'Zeit_4_4': 'Time',
    'Anz_Kodierungen': 'Num Encodings',
    'Control': 'Control'}
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
    df=df1.drop(columns=['FotoNr','Bild_vorh','TN','Geschlecht','Gruppe','Hauptoberkategorie','Hauptsubkategorie','Valenz','Anz_Kodierungen','Control'])
    df=df.fillna('')
    df=rename(df)
    df=trad(df)
    return df