import numpy as np
from sklearn.model_selection import KFold
import pandas as pd  # data locations

data_location = "./ml-1m/"
movies_location = f"{data_location}movies.dat"
ratings_location = f"{data_location}ratings.dat"
users_location = f"{data_location}users.dat"  # create dataframes

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

df = df_ratings.merge(df_users, on='UserID')
df = df.merge(df_movies, on='MovieID')

# X = df[["UserID","MovieID"]]
# y = df[['Rating']]  # split data in 5

fold = KFold(n_splits=5, shuffle=True, random_state=86)  # my birthday :)

rmse_scores = []
mae_scores = []

for train_index, test_index in fold.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    # Create a user-movie ratings matrix
    user_movie_matrix = pd.pivot_table(train_data, index='UserID', columns='MovieID', values='Rating').fillna(0)

    # Perform SVD on the training data
    U, S, Vh = np.linalg.svd(user_movie_matrix, full_matrices=True)

    # Choose the number of singular values/components
    k = 10

    # Perform matrix reconstruction to get predictions
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vh_k = Vh[:k, :]
    prediction = np.dot(np.dot(U_k, S_k), Vh_k)

    # Extract test data for evaluation
    test_user_ids = test_data['UserID']
    test_movie_ids = test_data['MovieID']
    test_ratings = test_data['Rating']

    # Calculate RMSE and MAE for each test data point
    rmse = 0.0
    mae = 0.0
    num_test_samples = len(test_user_ids)

    for i in range(num_test_samples):
        user_id = test_user_ids.iloc[i]
        movie_id = test_movie_ids.iloc[i]
        rating = test_ratings.iloc[i]

        # Check if the user and movie IDs are within the bounds
        if user_id in user_movie_matrix.index and movie_id in user_movie_matrix.columns:
            predicted_rating = prediction[user_id, movie_id]
            rmse += (rating - predicted_rating) ** 2
            mae += abs(rating - predicted_rating)

    rmse = np.sqrt(rmse / num_test_samples)
    mae = mae / num_test_samples

    rmse_scores.append(rmse)
    mae_scores.append(mae)

# Calculate the average RMSE and MAE
avg_rmse = np.mean(rmse_scores)
avg_mae = np.mean(mae_scores)

print("Average RMSE:", avg_rmse)
print("Average MAE:", avg_mae)