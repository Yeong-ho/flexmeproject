import pandas as pd
'''
data_dict = {
    'Name' : ['John', 'Sabre','Kim','Sato','Lee', 'Smith','David'],
    'Country' : ['USA', 'France', 'Korea', 'Japan', 'Korea', 'USA', 'USA',],
    'Age' : [31,33,28,40,36,55,48],
    'Job' : ['Student','Lawyer','Developer','Chef', 'Professor','CEO','Banker']        
            }

df = pd.DataFrame(data_dict)
df.to_csv('./test.csv',index=True)
'''

df = pd.read_csv('./test.csv')

print(df)