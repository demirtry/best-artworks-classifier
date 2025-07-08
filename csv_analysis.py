import pandas as pd


dataset = pd.read_csv('data/artists.csv', sep=',')
print(dataset['genre'].unique())