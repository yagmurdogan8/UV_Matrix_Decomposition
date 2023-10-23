import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from math import sqrt

# data locations
data_location = "./ml-1m/"
movies_location = f"{data_location}movies.dat"
ratings_location = f"{data_location}ratings.dat"
users_location = f"{data_location}users.dat"

# create dataframes for ratings, users, movies
column_ratings = ['UserID', 'MovieID', 'Rating', 'Zip-Timestamp']
fields_ratings = ['UserID', 'MovieID', 'Rating']
df_ratings = pd.read_table(ratings_location, engine="python", sep="::", names=column_ratings, usecols=fields_ratings,
                           encoding="ISO-8859-1")
print(df_ratings.head())

column_users = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
fields_users = ['UserID']
df_users = pd.read_table(users_location, engine="python", sep="::", names=column_users, usecols=fields_users,
                         encoding="ISO-8859-1")
print(df_users.head())

column_movies = ['MovieID', 'Title', 'Genres']
fields_movies = ['MovieID']
df_movies = pd.read_table(movies_location, engine="python", sep="::", names=column_movies, usecols=fields_movies,
                          encoding="ISO-8859-1")
print(df_movies.head())

# merged dataframe containing [user_id, movie_id, rating]
df = df_ratings.merge(df_users, on='UserID')
df = df.merge(df_movies, on="MovieID")

# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import KFold
# import pandas as pd  # data locations
# from math import sqrt
#
# movies_location = "./ml-1m/movies.dat"
# ratings_location = "./ml-1m/ratings.dat"
# users_location = "./ml-1m/users.dat"
#
# column_ratings = ['UserID', 'MovieID', 'Rating', 'Zip-Timestamp']
# fields_ratings = ['UserID', 'MovieID', 'Rating']
#
# df_ratings = pd.read_table(ratings_location, engine="python", sep="::", names=column_ratings, usecols=fields_ratings,
#                            encoding="ISO-8859-1")
# print(df_ratings.head())
#
# column_users = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
# fields_users = ['UserID']
# df_users = pd.read_table(users_location, engine="python", sep="::", names=column_users, usecols=fields_users,
#                          encoding="ISO-8859-1")
# print(df_users.head())
#
# column_movies = ['MovieID', 'Title', 'Genres']
# fields_movies = ['MovieID']
# df_movies = pd.read_table(movies_location, engine="python", sep="::", names=column_movies, usecols=fields_movies,
#                           encoding="ISO-8859-1")
# print(df_movies.head())
#
# df = df_ratings.merge(df_users, on='UserID')
# df = df.merge(df_movies, on='MovieID')
#
# # X = df[["UserID","MovieID"]]
# # y = df[['Rating']]
#
# fold = KFold(n_splits=5, shuffle=True, random_state=86)  # my birthday :)
#
# rmse_scores = []
# mae_scores = []
#
# for train_index, test_index in fold.split(df_ratings):
#     train_data = df.iloc[train_index]
#     test_data = df.iloc[test_index]
#
#     # Create a user-movie ratings matrix
#     user_movie_matrix = pd.pivot_table(train_data, values='Rating', index='UserID', columns='MovieID').fillna(0)
#
#     # Perform SVD on the training data
#     U, S, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)
#
#     # Choose the number of singular values/components
#     k = 5
#
#     # Construct U_k, S_k, and Vt_k
#     U_k = U[:, :k]
#     S_k = np.diag(S[:k])
#     Vt_k = Vt[:k, :]
#
#     # Make predictions using U_k, S_k, and Vt_k
#     prediction = np.dot(np.dot(U_k, S_k), Vt_k)
#
#     # Calculate RMSE and MAE for the test set
#     user_indices = test_data['UserID'].values - 1
#     movie_indices = test_data['MovieID'].values - 1
#     ratings = test_data['Rating'].values
#
#     predicted_ratings = prediction[user_indices, movie_indices]
#
#     rmse = sqrt(mean_squared_error(ratings, predicted_ratings))
#     mae = mean_absolute_error(ratings, predicted_ratings)
#
#     rmse_scores.append(rmse)
#     mae_scores.append(mae)
#
# # Calculate the average RMSE and MAE
# avg_rmse = np.mean(rmse_scores)
# avg_mae = np.mean(mae_scores)
#
# print("Average RMSE:", avg_rmse)
# print("Average MAE:", avg_mae)

# ratings_data = pd.read_csv('./ml-1m//ratings.dat', sep='::',
#                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='ISO-8859-1')
#
# movies_data = pd.read_csv('./ml-1m//movies.dat', sep='::', names=['MovieID', 'MovieName', 'Genre'],
#                           engine='python', encoding='ISO-8859-1')
#
# # No of users and movies
# num_users = ratings_data['UserID'].nunique()
# num_movies = movies_data['MovieID'].nunique()
#
# ratings_matrix = np.zeros((num_users, num_movies))
#
# for row in ratings_data.itertuples():
#     ratings_matrix[row.UserID - 1, row.MovieID - 1] = row.Rating
#
# # Number of folds for cross-validation
# num_folds = 5
# kf = KFold(n_splits=num_folds)
#
# rmse_scores = []
# mae_scores = []
#
# for train_indices, test_indices in kf.split(ratings_data):
#     train_data = ratings_data.iloc[train_indices]
#     test_data = ratings_data.iloc[test_indices]
#
#     # Create a user-movie ratings matrix
#     user_movie_matrix = np.zeros((num_users, num_movies))
#     for row in train_data.itertuples():
#         user_movie_matrix[row.UserID - 1, row.MovieID - 1] = row.Rating
#
#     # Perform SVD on the training data
#     U, S, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)
#
#     # Choose the number of singular values/components
#     k = 20
#
#     # Construct U_k, S_k, and Vt_k
#     U_k = U[:, :k]
#     S_k = np.diag(S[:k])
#     Vt_k = Vt[:k, :]
#
#     # Make predictions using U_k, S_k, and Vt_k
#     prediction = np.dot(np.dot(U_k, S_k), Vt_k)
#
#     # Calculate RMSE and MAE for the test set
#     user_indices = test_data['UserID'].values - 1
#     movie_indices = test_data['MovieID'].values - 1
#     ratings = test_data['Rating'].values
#
#     predicted_ratings = prediction[user_indices, movie_indices]
#
#     rmse = sqrt(mean_squared_error(ratings, predicted_ratings))
#     mae = mean_absolute_error(ratings, predicted_ratings)
#
#     rmse_scores.append(rmse)
#     mae_scores.append(mae)
#
# # Calculate the average RMSE and MAE over all folds
# avg_rmse = np.mean(rmse_scores)
# avg_mae = np.mean(mae_scores)
#
# print("Average RMSE:", avg_rmse)
# print("Average MAE:", avg_mae)

# Load the MovieLens 1M dataset

ratings_data = pd.read_csv('./ml-1m/ratings.dat', sep='::',
                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'],engine='python', encoding='ISO-8859-1')
movies_data = pd.read_csv('./ml-1m/movies.dat', sep='::', names=['MovieID', 'Title', 'Genre'],
                          engine='python', encoding='ISO-8859-1')

movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movies_data['MovieID'])}

fold = KFold(n_splits=5, shuffle=True, random_state=86)  # my birthday :)

num_users = ratings_data['UserID'].nunique()
num_movies = len(movies_data)

ratings_matrix = np.zeros((num_users, num_movies))

for row in ratings_data.itertuples():
    movie_index = movie_id_to_index.get(row.MovieID)
    if movie_index:
        ratings_matrix[row.UserID - 1, movie_index] = row.Rating

rmse_scores = []
mae_scores = []

for train_indices, test_indices in fold.split(ratings_data):
    train_data = ratings_data.iloc[train_indices]
    test_data = ratings_data.iloc[test_indices]

    user_movie_matrix = np.zeros((num_users, num_movies))
    for row in train_data.itertuples():
        movie_index = movie_id_to_index.get(row.MovieID)
        if movie_index is not None:
            user_movie_matrix[row.UserID - 1, movie_index] = row.Rating

    U, S, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)

    k = 500

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    prediction = np.dot(np.dot(U_k, S_k), Vt_k)

    user_indices = test_data['UserID'].values - 1
    movie_indices = [movie_id_to_index.get(movie_id) for movie_id in test_data['MovieID']]
    ratings = test_data['Rating'].values

    predicted_ratings = prediction[user_indices, movie_indices]

    rmse = sqrt(mean_squared_error(ratings, predicted_ratings))
    mae = mean_absolute_error(ratings, predicted_ratings)

    rmse_scores.append(rmse)
    mae_scores.append(mae)

avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)

print("Average RMSE:", avg_rmse)
print("Average MAE:", avg_mae)
