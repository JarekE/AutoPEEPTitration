# Load the data
import pandas as pd

def load():
    df = pd.read_csv('C:/Users/inesg/Documents/Uni/Master/Seminar/breath_data/190114_peep_tit_1.csv')
    #print(df)
    #print(df[df.columns[::16]])
    return(df[df.columns[::16]])
