# Import Libraries 
import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import matplotlib.pyplot as plt
# Ensures graphs to be displayed in ipynb
%matplotlib inline   

# read data into dataframe
titanic_df = pd.read_csv('../input/train.csv',header=0)  # Always use header=0 to read header of csv files

orig_df = pd.DataFrame.copy(titanic_df)
print(titanic_df.head())
orig_df.head()
orig_df.info()

#dropping irrelevant column for building model
titanic_df.drop(['Name','Sex','Ticket'],1, inplace= True)
titanic_df.convert_objects(convert_numeric = True)
titanic_df.fillna(0, inplace= True)

print (titanic_df.head())

def handle_non_numerical_data(titanic_df):
    # handling non-numerical data: must convert.
	columns = titanic_df.columns.values
	
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
        
		#print(column,df[column].dtype)	
		if titanic_df[column].dtype !=np.int64 and titanic_df[column].dtype !=np.float64:
			column_contents = titanic_df[column].values.tolist()
            #finding just the uniques
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
					text_digit_vals[unique] = x
					x+=1
			# now we map the new "id" vlaue
            # to replace the string. 
			titanic_df[column] = list(map(convert_to_int, titanic_df[column]))
			
	return titanic_df
	
titanic_df = handle_non_numerical_data(titanic_df)
#print (titanic_df.head())


X = np.array(titanic_df.drop(['Survived'],1). astype(float))
X = preprocessing.scale(X)
y = np.array(titanic_df['Survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# add a new column to our original dataframe:
orig_df['cluster_group'] = np.nan

#iterate through the labels and populate the labels to the empty column:
for i in range(len(X)):
    orig_df['cluster_group'].iloc[i] = labels[i]



#check the survival rates for each of the groups we happen to find:
n_clusters_ =len(np.unique(labels))
survival_rates = {}


for i in range(n_clusters_):
    temp_df = orig_df[(orig_df['cluster_group']==float(i))]
    #print(temp_df.head())
    survival_cluster = temp_df[(temp_df['Survived'] ==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    #print(i,survival_rate)
    survival_rates[i] = survival_rate
    
print (survival_rates)