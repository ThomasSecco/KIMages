from vect import simi
import pandas as pd

def benchmark(df,epsilon=0.8): #epsilon representing the quantile 
    df1=df.drop(columns=['Bildbez','Productivity','Photo scene','Photo title','Note','class','Description'])
    x=df[['Photo scene','Photo title','Note','Description']]
    x=x.fillna('')
    df1=df1.fillna(0)

    benchmarks={}
    for col in df1.columns:
        max_sims=[]
        for i in range(df.shape[0]):
            max_sim=0
            if df1[col][i]==1:
                for j in range(4):
                    if type(x[x.columns[j]][i])==str:
                        for word in x[x.columns[j]][i].split():
                            s=simi(col,word)
                            if s>max_sim:
                                max_sim=s
                    else:
                        for string in x[x.columns[j]][i]:
                            for word in string.split():
                                s=simi(col,word)
                                if s>max_sim:
                                    max_sim=s
                max_sims.append(max_sim)
        ser=pd.Series(max_sims)    
        benchmarks[f'{col}']=ser.quantile(epsilon)
    return benchmarks


def prediction_one(x,benchmarks,tolerance=0.8):
    y_pred=[]
    columns=list(benchmarks.keys())
    for col in columns:
        max_sim=0
        for j in range (4):
            if type(x[j])==str:
                for y in x[j].split():
                    s=simi(col,y)
                    if s>max_sim:
                        max_sim=s
            else:
                for string in x[j]:
                    for word in string.split():
                        s=simi(col,word)
                        if s>max_sim:
                            max_sim=s
        if max_sim>min(benchmarks[col],tolerance):
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    return y_pred


def prediction_list(x,benchmarks,tolerance=0.8):
    x=x.fillna('')
    y_pred=[]
    for i in range(len(x)):
        y_pred.append(prediction_one(x.iloc[i],benchmarks,tolerance))
    return pd.DataFrame(y_pred,columns=list(benchmarks.keys()))