""" 
-----------------TOP rated 1000 IMDB Movies Dataset------------------
    This script analyzes the Top 1000 rated movies from the IMDB dataset.
    It performs data loading, cleaning, and visualization tasks to provide insights into movie ratings, 
    genres, and release years.
source: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

Starting Dataset is 16 columns and 1000 rows.

The most important columns are:
- 'Series_Title': The title of the movie.
- 'Released_Year': The release year of the movie.
- 'Certificate': The certification rating of the movie (e.g., PG-13, R).
- 'Runtime': The duration of the movie in minutes.
- 'Genre': The genre(s) of the movie.
- 'IMDB_Rating': The IMDB rating of the movie.     ////OUTPUT////
- 'Meta_score': The Metascore rating of the movie.
- 'Director': The director of the movie.
- 'Gross': The gross earnings of the movie.

"""

# Import necessary libraries
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('IMDB_Top_1000.csv')


""" ================Summary Statistics================= """

print("____________Summary Statistics:___________")
print(df.describe().round(2))  # Display summary statistics rounded to 2 decimal places

average_IMDB_rating = df['IMDB_Rating'].mean()      #  /!\  can be used as output
print(f'Average IMDB Rating: {average_IMDB_rating:.2f}/10 ')    #calculating and displaying average IMDB rating

average_metascore = df['Meta_score'].mean()
print(f'Average Metascore: {average_metascore:.2f}/100 ') #same for metascore



df['Runtime'] = df['Runtime'].str.replace('min', '').str.strip()
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce').astype('Int64') #removing 'min' and converting to numeric type
average_runtime= df['Runtime'].mean()                                         #to calculate average runtime
print(f'Average Runtime: {average_runtime:.2f} minutes ')

average_number_of_votes = df['No_of_Votes'].mean()
print(f'Average Number of Votes: {average_number_of_votes:.2f} votes ') #same for number of votes


df['Gross'] = pd.to_numeric(df['Gross'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce') #cleaning 'Gross' column from punctuations 
average_gross = df['Gross'].mean()                                                                              #to keep only numeric values
print(f'Average Gross: {average_gross:,.2f} $')



"""==================OUTLIERS DETECTION=================="""
print("____________Outliers Detection:___________")
# Calculated values using an IQR method, we will not remove outliers this is the maximum values allowed before being considered an outlier
print("IMDB_Rating  outliers if > 8.70 \nNo_of_Votes  outliers if  > 851113.75 \nRuntime  outliers if > 178.75 \nGross  outliers if > 290000000.00")
'''
IMDB_Rating  outliers if > 8.7
No_of_Votes  outliers if  > 851113.75
Runtime  outliers if > 178
Gross  outliers if > 290000000.00
'''


"""==================Missing Values================== """
print("____________Missing Values in each column:___________")
missing_values=df.isnull().sum() #count missing values in each column
print(missing_values)
print("\n")

""" ---------------Data Cleaning: Handling Missing Values--------------- """


mean_metascore = df['Meta_score'].mean()                  #1. For 'Meta_score', we can fill missing values with the average Metascore.
df['Meta_score'] = df['Meta_score'].fillna(mean_metascore)

mean_gross = df['Gross'].mean()                            #2. For 'Gross', we can fill missing values with the average Gross.
df['Gross'] = df['Gross'].fillna(mean_gross)




# Still missing values in 'Certificate'and the 'Poster_Link'and 'Overview' we will drop those rows
df = df.drop('Certificate', axis=1)  
df = df.drop('Poster_Link', axis=1)
df= df.drop('Overview', axis=1)
# Also drop 'Star1', 'Star2', 'Star3', 'Star4' columns as they are not crucial for our analysis
df = df.drop('Star1', axis=1)
df = df.drop('Star2', axis=1)
df = df.drop('Star3', axis=1)
df = df.drop('Star4', axis=1)

"""---------------------------------------------------------------------"""


print("Missing values after cleaning:")
missing_values_after_cleaning=df.isnull().sum()
print(missing_values_after_cleaning)


'''===========================Duplicate values ==========================''' 
print("____________Duplicate Values:___________")
duplicate_values = df.duplicated().sum()
print(f'Number of duplicate rows: {duplicate_values}')
print("We dont have duplicate rows in our dataset.\n")


print("We have multiple genres in some rows, we will keep only the first genre for simplicity.\n") 
print(" Drama: 1\n ,Crime: 2\n ,Action: 3\n ,Biography: 4\n ,Adventure: 5\n ,Comedy: 6\n ,Thriller: 7\n ,Mystery: 8\n ,Western: 9\n ,Fantasy: 10\n ,Animation: 11\n ,Horror: 12\n ,Family: 13\n ,Film-Noir: 14\n ")

"""---------------Data cleaning : Encoding Categorical Variables--------------- """

df["Genre"] = df["Genre"].str.split(", ").str[0]    # We will only keep the first genre if multiple genres are listed
print(df["Genre"].unique())                         # Display unique genres after cleaning      


df["Genre"] = (df["Genre"].map({                    #converting all genres to unique integers
    "Drama": 1,
    "Crime": 2,
    "Action": 3,
    "Biography": 4,
    "Adventure": 5,
    "Comedy": 6,
    "Thriller": 7,
    "Mystery": 8,
    "Western": 9,
    "Fantasy": 10,
    "Animation": 11,
    "Horror": 12,
    "Family": 13,
    "Film-Noir": 14,
}))



unique_titles = df['Series_Title'].unique()                                     #replace Series_Title with unique integers  
title_mapping = {name: idx + 1 for idx, name in enumerate(unique_titles)} 
df['Series_Title'] = df['Series_Title'].map(title_mapping)
print(df)

''' 
//////////////////////////IF NEEDED///////////////////////////////////
"replace Director names with unique integers"
unique_directors = df['Director'].unique()
director_mapping = {name: idx + 1 for idx, name in enumerate(unique_directors)}
df['Director'] = df['Director'].map(director_mapping)
print(df)
//////////////////////////IF NEEDED///////////////////////////////////
'''
df = df.drop('Director', axis=1) #for now we drop Director column as not crucial for our analysis
"""---------------------------------------------------------------------"""


print("\n")
print("________________Data Visualization:________________")
"""----------------Data Visualization to see distributions and outliers--------------- """

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

axes[0].hist(df['IMDB_Rating'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of IMDB Ratings')
axes[0].set_xlabel('IMDB Rating')
axes[0].set_ylabel('Number of Movies')

axes[1].hist(df['No_of_Votes'], bins=20, color='salmon', edgecolor='black')
axes[1].set_title('Distribution of Number of Votes')
axes[1].set_xlabel('Number of Votes')
axes[1].set_ylabel('Number of Movies')

axes[2].hist(df['Runtime'].dropna(), bins=20, color='lightgreen', edgecolor='black')
axes[2].set_title('Distribution of Movie Runtimes')
axes[2].set_xlabel('Runtime (minutes)')
axes[2].set_ylabel('Number of Movies')

axes[3].hist(df['Gross'].dropna(), bins=20, color='orange', edgecolor='black')
axes[3].set_title('Distribution of Gross Earnings')
axes[3].set_xlabel('Gross Earnings ($)')
axes[3].set_ylabel('Number of Movies')

plt.tight_layout()
plt.show()


# Boxplots to visualize outliers, outliers will be ootside the rectangle "
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].boxplot(df['IMDB_Rating'].dropna())
axes[0].set_title('Boxplot of IMDB Ratings')
axes[0].set_ylabel('IMDB Rating')

axes[1].boxplot(df['Runtime'].dropna())
axes[1].set_title('Boxplot of Runtime')
axes[1].set_ylabel('Runtime (minutes)')

axes[2].boxplot(df['No_of_Votes'].dropna())
axes[2].set_title('Boxplot of Votes')
axes[2].set_ylabel('Votes Count')


plt.tight_layout()
plt.show()


#           Scatter plots to visualize potential correlations 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].scatter(df['Runtime'], df['IMDB_Rating'], alpha=0.7, s=30, color='tab:blue', edgecolor='k', linewidth=0.3)
axes[0].set_title("Runtime vs IMDB Rating")
axes[0].set_xlabel("Runtime (minutes)")
axes[0].set_ylabel("IMDB Rating")
axes[0].grid(True)

axes[1].scatter(df['No_of_Votes'], df['IMDB_Rating'], alpha=0.7, s=30, color='tab:orange', edgecolor='k', linewidth=0.3)
axes[1].set_title("Number of Votes vs IMDB Rating")
axes[1].set_xlabel("Number of Votes")
axes[1].grid(True)


plt.tight_layout()
plt.show()


# MOvies per Genre
df['Genre'].value_counts().plot(kind='bar', figsize=(10,5))
plt.title("Number of Movies per Genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.show()


#Correlation analysis using heatmap 

sns.heatmap(df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation analysis using heatmap")
plt.show()



"""---------------Data Scaling---------------"""

'SCalinfg the data using MinMax Scaler'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']] = scaler.fit_transform(df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']])
print("\n")
print("____________Scaled Data:___________")
print(df)   
original_scaler = scaler
"""---------------------------------------------------------------------"""









"""---------------Outliers Decision---------------"""

#       We have some outliers in our dataset but we will keep them 
#       as they might represent important variations in movie ratings and earnings.

'''---------------------------------------------------------------------'''





"""---------------------------------------------------------------------"""

"""==================PHASE 1 : UNSUPERVISED LEARNING - KMEANS CLUSTERING=================="""


""" Elbow method to find the optimal number of clusters for K-Means Clustering """

from sklearn.cluster import KMeans
k_range = range(1, 10)

features = ['IMDB_Rating','Meta_score','Runtime','Gross','No_of_Votes','Genre']  # Features to be used for clustering

error1 = []
for K in k_range:
    km =  KMeans(n_clusters=K,random_state=42)
    km.fit(df[features])
    error1.append(km.inertia_)



plt.plot(k_range, error1, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Error / Inertia')
plt.title('Elbow Method to find optimal K')
plt.show()   



# From the elbow plot, we choose K=4 as the optimal number of clusters

km = KMeans(n_clusters=4, random_state=42)
y_predicted = km.fit_predict(df[features])
# print(y_predicted)

df['cluster'] = y_predicted

# Visualizing the clusters based on IMDB_Rating and Gross (2D plot for simplicity ignore if clustres not really accurate)
plt.figure(figsize=(14, 8))
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]
df4 = df[df['cluster'] == 3]

plt.scatter(df1['IMDB_Rating'], df1['Gross'], color='green')
plt.scatter(df2['IMDB_Rating'], df2['Gross'], color='red')
plt.scatter(df3['IMDB_Rating'], df3['Gross'], color='blue')
plt.scatter(df4['IMDB_Rating'], df4['Gross'], color='orange')

plt.show()


"""==================== PHASE 2 — SUPERVISED LEARNING ===================="""

#importing necessary libraries for KNN Classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns



# Preparing data for KNN Classification
X = df[features]  
y = df['cluster'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""---------------KNN Classification---------------"""



accuracies = []         #Finding the best value of k for KNN 
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(8, 5))                  #Plotting accuracy vs k to see the best k
plt.plot(range(1, 21), accuracies, marker='o', color='blue')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k')
plt.grid(True)
plt.show()


print("\nAccuracies for each k:")
print(accuracies)

best_k = accuracies.index(max(accuracies)) + 1          # +1 because index starts at 0
print(f"\nThe best value of k is {best_k}")
print(best_k)
knn_classification_best_k_value = KNeighborsClassifier(n_neighbors=best_k)  #Using the best k value found
knn_classification_best_k_value.fit(X_train, y_train)                       #Fitting the model
y_pred_answers = knn_classification_best_k_value.predict(X_test)        #Predicting the test set

accuracy = accuracy_score(y_test, y_pred_answers)           #Calculating accuracy etc...
print(f"Accuracy:  {accuracy:.4f}  of predictions are correct")


precision = precision_score(y_test, y_pred_answers, average='weighted', zero_division=0)
print(f"Precision: {precision:.4f}  of the films predicted in a cluster actually belong there")

recall = recall_score(y_test, y_pred_answers, average='weighted')
print(f"Recall:    {recall:.4f}  of the films in a cluster are correctly identified")

f1 = f1_score(y_test, y_pred_answers, average='weighted')
print(f"F1-Score:  {f1:.4f}  mean of Accuracy and Recall")




"""
Confusion Matrix Interpretation:

                Predicted
                0    1    2    3
            ---------------------
Actual    0|    [45]  2    1    0     ← 45 films du cluster 0 bien classés
          1|    1  [38]   2    1     ← 38 films du cluster 1 bien classés
          2|    0    3  [52]   1     ← 52 films du cluster 2 bien classés  
          3|    1    0    2  [51]    ← 51 films du cluster 3 bien classés
          


"""

print("\n Confusion Matrix:") #confusion matrix to visualize the performance of the classification
cm = confusion_matrix(y_test, y_pred_answers)
print(cm)




from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier



"""---------------Random Forest Classification---------------"""
my_model = RandomForestClassifier(n_estimators=100, random_state=42) #Using Random Forest Classifier as another supervised learning method
my_model.fit(X_train, y_train)
y_pred_rf = my_model.predict(X_test)        #
accuracy_rf = accuracy_score(y_test, y_pred_rf)     #
print(f"\nRandom Forest Classifier Accuracy: {accuracy_rf:.4f}")
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # 5-Fold Cross-Validation
cv_scores = cross_val_score(my_model, X, y)
print(f"Random Forest Classifier Cross-Validation Scores: {cv_scores}")
print(f"Random Forest Classifier Average CV Score: {cv_scores.mean():.4f}")

# Detailed metrics for Random Forest Classifier

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')




# Comparing KNN and Random Forest Classifier

print("\n Random Forest  vs KNN :")


print("\n")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"KNN Accuracy: {accuracy:.4f}")
print("\n")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"KNN Precision: {precision:.4f}")
print("\n")

print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"KNN Recall: {recall:.4f}")
print("\n")
print(f"Random Forest F1-Score: {f1_rf:.4f}")
print(f"KNN F1-Score: {f1:.4f}")

print(df)






""" 
-----------------TOP rated 1000 IMDB Movies Dataset------------------
This script analyzes the Top 1000 rated movies from the IMDB dataset.
It performs data loading, cleaning, and visualization tasks to provide insights into movie ratings, 
genres, and release years.
source: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

Starting Dataset is 16 columns and 1000 rows.

The most important columns are:
- 'Series_Title': The title of the movie.
- 'Released_Year': The release year of the movie.
- 'Certificate': The certification rating of the movie (e.g., PG-13, R).
- 'Runtime': The duration of the movie in minutes.
- 'Genre': The genre(s) of the movie.
- 'IMDB_Rating': The IMDB rating of the movie.     ////OUTPUT////
- 'Meta_score': The Metascore rating of the movie.
- 'Director': The director of the movie.
- 'Gross': The gross earnings of the movie.

"""

# Import necessary libraries
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('IMDB_Top_1000.csv')


""" ================Summary Statistics================= """

print("____________Summary Statistics:___________")
print(df.describe().round(2))  # Display summary statistics rounded to 2 decimal places

average_IMDB_rating = df['IMDB_Rating'].mean()      #  /!\  can be used as output
print(f'Average IMDB Rating: {average_IMDB_rating:.2f}/10 ')    #calculating and displaying average IMDB rating

average_metascore = df['Meta_score'].mean()
print(f'Average Metascore: {average_metascore:.2f}/100 ') #same for metascore



df['Runtime'] = df['Runtime'].str.replace('min', '').str.strip()
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce').astype('Int64') #removing 'min' and converting to numeric type
average_runtime= df['Runtime'].mean()                                         #to calculate average runtime
print(f'Average Runtime: {average_runtime:.2f} minutes ')

average_number_of_votes = df['No_of_Votes'].mean()
print(f'Average Number of Votes: {average_number_of_votes:.2f} votes ') #same for number of votes


df['Gross'] = pd.to_numeric(df['Gross'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce') #cleaning 'Gross' column from punctuations 
average_gross = df['Gross'].mean()                                                                              #to keep only numeric values
print(f'Average Gross: {average_gross:,.2f} $')



"""==================OUTLIERS DETECTION=================="""
print("____________Outliers Detection:___________")
# Calculated values using an IQR method, we will not remove outliers this is the maximum values allowed before being considered an outlier
print("IMDB_Rating  outliers if > 8.70 \nNo_of_Votes  outliers if  > 851113.75 \nRuntime  outliers if > 178.75 \nGross  outliers if > 290000000.00")
'''
IMDB_Rating  outliers if > 8.7
No_of_Votes  outliers if  > 851113.75
Runtime  outliers if > 178
Gross  outliers if > 290000000.00
'''


"""==================Missing Values================== """
print("____________Missing Values in each column:___________")
missing_values=df.isnull().sum() #count missing values in each column
print(missing_values)
print("\n")

""" ---------------Data Cleaning: Handling Missing Values--------------- """


mean_metascore = df['Meta_score'].mean()                  #1. For 'Meta_score', we can fill missing values with the average Metascore.
df['Meta_score'] = df['Meta_score'].fillna(mean_metascore)

mean_gross = df['Gross'].mean()                            #2. For 'Gross', we can fill missing values with the average Gross.
df['Gross'] = df['Gross'].fillna(mean_gross)




# Still missing values in 'Certificate'and the 'Poster_Link'and 'Overview' we will drop those rows
df = df.drop('Certificate', axis=1)  
df = df.drop('Poster_Link', axis=1)
df= df.drop('Overview', axis=1)
# Also drop 'Star1', 'Star2', 'Star3', 'Star4' columns as they are not crucial for our analysis
df = df.drop('Star1', axis=1)
df = df.drop('Star2', axis=1)
df = df.drop('Star3', axis=1)
df = df.drop('Star4', axis=1)

"""---------------------------------------------------------------------"""


print("Missing values after cleaning:")
missing_values_after_cleaning=df.isnull().sum()
print(missing_values_after_cleaning)


'''===========================Duplicate values ==========================''' 
print("____________Duplicate Values:___________")
duplicate_values = df.duplicated().sum()
print(f'Number of duplicate rows: {duplicate_values}')
print("We dont have duplicate rows in our dataset.\n")


print("We have multiple genres in some rows, we will keep only the first genre for simplicity.\n") 
print(" Drama: 1\n ,Crime: 2\n ,Action: 3\n ,Biography: 4\n ,Adventure: 5\n ,Comedy: 6\n ,Thriller: 7\n ,Mystery: 8\n ,Western: 9\n ,Fantasy: 10\n ,Animation: 11\n ,Horror: 12\n ,Family: 13\n ,Film-Noir: 14\n ")

"""---------------Data cleaning : Encoding Categorical Variables--------------- """

df["Genre"] = df["Genre"].str.split(", ").str[0]    # We will only keep the first genre if multiple genres are listed
print(df["Genre"].unique())                         # Display unique genres after cleaning      


df["Genre"] = (df["Genre"].map({                    #converting all genres to unique integers
    "Drama": 1,
    "Crime": 2,
    "Action": 3,
    "Biography": 4,
    "Adventure": 5,
    "Comedy": 6,
    "Thriller": 7,
    "Mystery": 8,
    "Western": 9,
    "Fantasy": 10,
    "Animation": 11,
    "Horror": 12,
    "Family": 13,
    "Film-Noir": 14,
}))



unique_titles = df['Series_Title'].unique()                                     #replace Series_Title with unique integers  
title_mapping = {name: idx + 1 for idx, name in enumerate(unique_titles)} 
df['Series_Title'] = df['Series_Title'].map(title_mapping)
print(df)

''' 
//////////////////////////IF NEEDED///////////////////////////////////
"replace Director names with unique integers"
unique_directors = df['Director'].unique()
director_mapping = {name: idx + 1 for idx, name in enumerate(unique_directors)}
df['Director'] = df['Director'].map(director_mapping)
print(df)
//////////////////////////IF NEEDED///////////////////////////////////
'''
df = df.drop('Director', axis=1) #for now we drop Director column as not crucial for our analysis
"""---------------------------------------------------------------------"""


print("\n")
print("________________Data Visualization:________________")
"""----------------Data Visualization to see distributions and outliers--------------- """

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

axes[0].hist(df['IMDB_Rating'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of IMDB Ratings')
axes[0].set_xlabel('IMDB Rating')
axes[0].set_ylabel('Number of Movies')

axes[1].hist(df['No_of_Votes'], bins=20, color='salmon', edgecolor='black')
axes[1].set_title('Distribution of Number of Votes')
axes[1].set_xlabel('Number of Votes')
axes[1].set_ylabel('Number of Movies')

axes[2].hist(df['Runtime'].dropna(), bins=20, color='lightgreen', edgecolor='black')
axes[2].set_title('Distribution of Movie Runtimes')
axes[2].set_xlabel('Runtime (minutes)')
axes[2].set_ylabel('Number of Movies')

axes[3].hist(df['Gross'].dropna(), bins=20, color='orange', edgecolor='black')
axes[3].set_title('Distribution of Gross Earnings')
axes[3].set_xlabel('Gross Earnings ($)')
axes[3].set_ylabel('Number of Movies')

plt.tight_layout()



# Boxplots to visualize outliers, outliers will be ootside the rectangle "
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].boxplot(df['IMDB_Rating'].dropna())
axes[0].set_title('Boxplot of IMDB Ratings')
axes[0].set_ylabel('IMDB Rating')

axes[1].boxplot(df['Runtime'].dropna())
axes[1].set_title('Boxplot of Runtime')
axes[1].set_ylabel('Runtime (minutes)')

axes[2].boxplot(df['No_of_Votes'].dropna())
axes[2].set_title('Boxplot of Votes')
axes[2].set_ylabel('Votes Count')


plt.tight_layout()



#           Scatter plots to visualize potential correlations 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].scatter(df['Runtime'], df['IMDB_Rating'], alpha=0.7, s=30, color='tab:blue', edgecolor='k', linewidth=0.3)
axes[0].set_title("Runtime vs IMDB Rating")
axes[0].set_xlabel("Runtime (minutes)")
axes[0].set_ylabel("IMDB Rating")
axes[0].grid(True)

axes[1].scatter(df['No_of_Votes'], df['IMDB_Rating'], alpha=0.7, s=30, color='tab:orange', edgecolor='k', linewidth=0.3)
axes[1].set_title("Number of Votes vs IMDB Rating")
axes[1].set_xlabel("Number of Votes")
axes[1].grid(True)


plt.tight_layout()



# MOvies per Genre
df['Genre'].value_counts().plot(kind='bar', figsize=(10,5))
plt.title("Number of Movies per Genre")
plt.xlabel("Genre")
plt.ylabel("Count")



#Correlation analysis using heatmap 

sns.heatmap(df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation analysis using heatmap")




"""---------------Data Scaling---------------"""

'SCalinfg the data using MinMax Scaler'
from sklearn.preprocessing import StandardScaler
df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']] = StandardScaler().fit_transform(df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']])
print("\n")
print("____________Scaled Data:___________")
print(df)   
"""---------------------------------------------------------------------"""









"""---------------Outliers Decision---------------"""

#       We have some outliers in our dataset but we will keep them 
#       as they might represent important variations in movie ratings and earnings.

'''---------------------------------------------------------------------'''





"""---------------------------------------------------------------------"""

"""==================PHASE 1 : UNSUPERVISED LEARNING - KMEANS CLUSTERING=================="""


""" Elbow method to find the optimal number of clusters for K-Means Clustering """

from sklearn.cluster import KMeans
k_range = range(1, 10)

features = ['IMDB_Rating','Meta_score','Runtime','Gross','No_of_Votes','Genre']  # Features to be used for clustering

error1 = []
for K in k_range:
    km =  KMeans(n_clusters=K,random_state=42)
    km.fit(df[features])
    error1.append(km.inertia_)



plt.plot(k_range, error1, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Error / Inertia')
plt.title('Elbow Method to find optimal K')
  



# From the elbow plot, we choose K=4 as the optimal number of clusters

km = KMeans(n_clusters=4, random_state=42)
y_predicted = km.fit_predict(df[features])
# print(y_predicted)

df['cluster'] = y_predicted

# Visualizing the clusters based on IMDB_Rating and Gross (2D plot for simplicity)
plt.figure(figsize=(14, 8))
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]
df4 = df[df['cluster'] == 3]

plt.scatter(df1['IMDB_Rating'], df1['Gross'], color='green')
plt.scatter(df2['IMDB_Rating'], df2['Gross'], color='red')
plt.scatter(df3['IMDB_Rating'], df3['Gross'], color='blue')
plt.scatter(df4['IMDB_Rating'], df4['Gross'], color='orange')




"""==================== PHASE 2 — SUPERVISED LEARNING ===================="""

#importing necessary libraries for KNN Classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns



# Preparing data for KNN Classification
X = df[features]  
y = df['cluster'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""---------------KNN Classification---------------"""



accuracies = []         #Finding the best value of k for KNN 
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(8, 5))                  #Plotting accuracy vs k to see the best k
plt.plot(range(1, 21), accuracies, marker='o', color='blue')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k')
plt.grid(True)



print("\nAccuracies for each k:")
print(accuracies)

best_k = accuracies.index(max(accuracies)) + 1          # +1 because index starts at 0
print(f"\nThe best value of k is {best_k}")
print(best_k)
knn_classification_best_k_value = KNeighborsClassifier(n_neighbors=best_k)  #Using the best k value found
knn_classification_best_k_value.fit(X_train, y_train)                       #Fitting the model
y_pred_answers = knn_classification_best_k_value.predict(X_test)        #Predicting the test set

accuracy = accuracy_score(y_test, y_pred_answers)           #Calculating accuracy etc...
print(f"Accuracy:  {accuracy:.4f}  of predictions are correct")


precision = precision_score(y_test, y_pred_answers, average='weighted', zero_division=0)
print(f"Precision: {precision:.4f}  of the films predicted in a cluster actually belong there")

recall = recall_score(y_test, y_pred_answers, average='weighted')
print(f"Recall:    {recall:.4f}  of the films in a cluster are correctly identified")

f1 = f1_score(y_test, y_pred_answers, average='weighted')
print(f"F1-Score:  {f1:.4f}  mean of Accuracy and Recall")




"""
Confusion Matrix Interpretation:

                Predicted
                0    1    2    3
            ---------------------
Actual    0|    [45]  2    1    0     ← 45 films du cluster 0 bien classés
          1|    1  [38]   2    1     ← 38 films du cluster 1 bien classés
          2|    0    3  [52]   1     ← 52 films du cluster 2 bien classés  
          3|    1    0    2  [51]    ← 51 films du cluster 3 bien classés
          


"""

print("\n Confusion Matrix:") #confusion matrix to visualize the performance of the classification
cm = confusion_matrix(y_test, y_pred_answers)
print(cm)




from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier



"""---------------Random Forest Classification---------------"""
my_model = RandomForestClassifier(n_estimators=100, random_state=42) #Using Random Forest Classifier as another supervised learning method
my_model.fit(X_train, y_train)
y_pred_rf = my_model.predict(X_test)        #
accuracy_rf = accuracy_score(y_test, y_pred_rf)     #
print(f"\nRandom Forest Classifier Accuracy: {accuracy_rf:.4f}")
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # 5-Fold Cross-Validation
cv_scores = cross_val_score(my_model, X, y)
print(f"Random Forest Classifier Cross-Validation Scores: {cv_scores}")
print(f"Random Forest Classifier Average CV Score: {cv_scores.mean():.4f}")

# Detailed metrics for Random Forest Classifier

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')




# Comparing KNN and Random Forest Classifier

print("\n Random Forest  vs KNN :")


print("\n")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"KNN Accuracy: {accuracy:.4f}")
print("\n")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"KNN Precision: {precision:.4f}")
print("\n")

print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"KNN Recall: {recall:.4f}")
print("\n")
print(f"Random Forest F1-Score: {f1_rf:.4f}")
print(f"KNN F1-Score: {f1:.4f}")

print(df)


"""==================== PHASE 3 — PREDICT THREE NEW DATA POINTS ===================="""


new_movies = pd.DataFrame({
    'IMDB_Rating': [3, 5, 8.7],         
    'Meta_score': [90, 70, 50],            
    'Runtime': [150, 120, 95],           
    'Gross': [400_000_000, 100_000_000, 20_000_000],  
    'No_of_Votes': [1_500_000, 400_000, 100_000],     
    'Genre': [1, 3, 6]  # Drama, Action, Comedy                   
})


scaler = StandardScaler()
scaler.fit(df[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']])

new_movies_scaled = new_movies.copy()   #Scaling with the same scaler as original data
new_movies_scaled[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']] = original_scaler.transform(
    new_movies[['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime', 'Gross']]
)


predicted_clusters = knn_classification_best_k_value.predict(new_movies_scaled[features])



print('Input values:')
print(new_movies)
for i in range(len(new_movies)):
    
    current_cluster = predicted_clusters[i]
    print("Predicted cluster for Movie N°",i+1,"is cluster n°",current_cluster)


    
    
    
    
