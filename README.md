#Music Recommendation System
Data Set 1 

[Uploading Screenshot 2023-06-12 at 3.09.40 PM.png…]()


Data Set 2

![Screenshot 2023-03-28 at 1.28.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1fcf1576-2214-4226-9d9a-18298c54c0a9/Screenshot_2023-03-28_at_1.28.04_PM.png)

https://zenodo.org/record/831189

Data Set 3 

https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset

*We are able to recommend top 10 similar songs to user based on the input. The recommendation is based on similarity of numerical features of the songs. We have calculated the cosine distance and identified the songs with highest similarity.*

![Screenshot 2023-03-28 at 2.04.07 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/58b0df7e-e7cc-471e-a33a-21e97d5dd4ea/Screenshot_2023-03-28_at_2.04.07_PM.png)

![Screenshot 2023-03-28 at 2.04.23 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/39b34fb0-1f1f-43ee-8405-40c8dcd4b938/Screenshot_2023-03-28_at_2.04.23_PM.png)

![Screenshot 2023-03-28 at 2.04.54 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b8e6a11a-f32e-494c-a7d0-c91072edd3d2/Screenshot_2023-03-28_at_2.04.54_PM.png)

![Screenshot 2023-03-28 at 2.11.52 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e76aaa98-c077-4537-89e6-7014bd4204a4/Screenshot_2023-03-28_at_2.11.52_PM.png)

 years on the x-axis

 the sound feature values on the y-axis. 

Each sound feature is  represented by a different colored line on the plot, with the title indicating which sound features are being plotted.

The sound features included in the plot are **`acousticness`**, **`danceability`**, **`energy`**, **`instrumentalness`**, **`liveness`**, and **`valence`**.

![Screenshot 2023-03-28 at 2.12.54 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/08607e2d-924c-4868-97f9-8a07047d5fe8/Screenshot_2023-03-28_at_2.12.54_PM.png)

The input data is a Pandas DataFrame called **`year_data`**
 which should have a column called 'year' representing the decade and a column called 'loudness' representing the loudness feature value for each decade.

![Screenshot 2023-03-28 at 2.14.02 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/87e3d475-3435-47eb-8582-56bc078f72c3/Screenshot_2023-03-28_at_2.14.02_PM.png)

horizontal bar chart using the Plotly Express library.

the code selects the top 10 most popular genres using the **`nlargest()`**
 function from the Pandas library.

The x-axis represents the genre names,

 the y-axis represents the values of four different sound features - valence, energy, danceability, and acousticness.

The **`barmode='group'`** parameter is used to group the bars for each sound feature.

The chart shows the trend of the four sound features for the top 10 genres.

![Screenshot 2023-03-28 at 2.22.48 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9294a228-2eb2-40a7-8f63-b7a807d1dcef/Screenshot_2023-03-28_at_2.22.48_PM.png)

This code creates a comma-separated string of all the genres 

present in the 'genres' column of the 'genre_data' DataFrame.

![Screenshot 2023-03-28 at 2.25.51 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6d7e1b0e-fd91-462f-b58d-789686fdb48a/Screenshot_2023-03-28_at_2.25.51_PM.png)

This code extracts a list of unique artist names from the 'artists' column

of the 'artist_data' dataframe,

 then converts that list into a comma-separated string.

The list of artist names is extracted using the 'tolist()' method of the 'artists' colum

![Screenshot 2023-03-28 at 2.27.25 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f4d42c73-e668-4b1a-8e9f-a42913291853/Screenshot_2023-03-28_at_2.27.25_PM.png)

**`top10_popular_artists`**
 is a DataFrame that contains the top 10 most popular artists

 based on their average popularity score across all their songs.

**`top10_most_song_produced_artists`**
 is a DataFrame that contains the top 10 artists who have produced the most number of songs in the dataset, based on the count of songs they have produced.

# C**lustering**

![Screenshot 2023-03-28 at 2.30.18 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b309701-3ab5-44ca-b973-4f488671008d/Screenshot_2023-03-28_at_2.30.18_PM.png)

This code is using t-SNE (t-Distributed Stochastic Neighbor Embedding), which is an unsupervised machine learning algorithm for dimensionality reduction and visualization of high-dimensional data

In this case, it is being used to visualize the clusters of genres based on their audio features.

Code applies t-SNE to reduce the dimensionality of the data down to two dimensions. The resulting two-dimensional coordinates are stored in a numpy array called **`genre_embedding`**.

Next, the code creates a pandas dataframe called **`projection`** that includes the x and y coordinates from the **`genre_embedding`** array, as well as the genre names and cluster assignments from the **`genre_data`** dataframe.

![Screenshot 2023-03-28 at 2.33.29 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56b792c9-eb7d-4559-a735-8b9c44689867/Screenshot_2023-03-28_at_2.33.29_PM.png)

dimensionality reduction using Principal Component Analysis (PCA) to visualize the clusters of songs

PCA is an unsupervised learning algorithm that takes high dimensional data and converts it into low dimensional data while preserving as much variance as possible.

it creates a pandas DataFrame 'projection' with the x and y coordinates of each song, as well as its cluster label and title

Finally, it uses the 'px.scatter' function from Plotly Express to create a scatter plot of the song embeddings

to find the song name from spotify api; 

![Screenshot 2023-03-28 at 2.35.36 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9c37f4d-72d2-43f4-8649-de26fbee8141/Screenshot_2023-03-28_at_2.35.36_PM.png)

takes two arguments, **`name`**and **`year`**

creates an empty dictionary called **`song_data`**

If a track is found, the function extracts  **`explicit`**
 status, **`duration_ms`**
, and **`popularity`**
 from the Spotify results. It also retrieves audio features for the track using **`sp.audio_features()`**
 and adds them to **`song_data`**
.

![Screenshot 2023-03-28 at 2.38.05 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff6995ce-83d2-4f42-b5b8-ffd06cc6aade/Screenshot_2023-03-28_at_2.38.05_PM.png)

"get_song_data" that takes two arguments: "song" and "spotify_data”

The "song" argument is a dictionary that contains the name and year of a song.

The "spotify_data" argument is a pandas DataFrame that contains details of songs.

The function tries to fetch the song details from the "spotify_data" DataFrame using the "name" and "year" of the song. If the details are available, the function returns the information.

Finally, the function returns the song details fetched from the local dataset or Spotify API. If no information is found, it returns None.

![Screenshot 2023-03-28 at 2.40.47 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8a7ae056-e186-42fb-abcf-d9873097d961/Screenshot_2023-03-28_at_2.40.47_PM.png)

The code defines a function named **`get_mean_vector`** that takes two arguments - **`song_list`** which is a list of dictionaries representing songs and **`spotify_data`** which is a pandas DataFrame containing song data. The function first initializes an empty list **`song_vectors`**. Then, it iterates through each song in **`song_list`** and calls another function **`get_song_data`** to fetch the song information either from the local dataset or from the Spotify dataset. If the song data is not available in either, it prints a warning message and moves to the next song. Otherwise, it selects only the numerical features of the song data and appends it to **`song_vectors`**.

Finally, the function converts the **`song_vectors`** list into a numpy array **`song_matrix`** where each row represents a song and each column represents a numerical feature. The **`numpy.mean`** function is then used to calculate the mean of each numerical feature across all songs, resulting in a 1D array representing the mean vector. This mean vector can be used to compare with other songs or clusters of songs to see how similar or dissimilar they are in terms of their numerical features.

![Screenshot 2023-03-28 at 2.41.44 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5b3ced77-535b-4972-a782-cc337f250ce2/Screenshot_2023-03-28_at_2.41.44_PM.png)

This flattened dictionary is useful for creating a pandas DataFrame, where each key in the dictionary becomes a column and the list of values for each key becomes a column of data.

# **Slide 7 here**

![Screenshot 2023-03-28 at 2.43.41 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/079cb34b-d3c0-490e-9d1b-886d2670bb5a/Screenshot_2023-03-28_at_2.43.41_PM.png)

This code is a function named **`recommend_songs`** which takes in a list of songs and recommends similar songs from a Spotify dataset. Here is a step-by-step explanation of how the function works:

1. **`metadata_cols`** is a list of columns that contain metadata about songs, including the name, year, and artists.
2. The function first uses **`flatten_dict_list`** to flatten the input song list into a dictionary with keys for the song name, year, and artists.
3. It then calls the **`get_mean_vector`** function to get the mean vector of the numerical features of the input songs. This mean vector is the "center" of the input songs.
4. The function then uses a **`scaler`** to scale the numerical features of both the input songs and the entire Spotify dataset.
5. It then calculates the cosine distances between the scaled mean vector and all the songs in the scaled dataset using the **`cdist`** function from the **`scipy.spatial.distance`** module. This produces a list of distances between the input mean vector and each song in the dataset.
6. The function then sorts the list of distances and gets the top **`n_songs`** songs with the lowest distances (i.e., the most similar songs).
7. The **`rec_songs`** dataframe is created by filtering the Spotify dataset to only include songs that are not in the input song list, and then selecting the metadata columns specified in **`metadata_cols`**.
8. The **`to_dict()`** method is called on the **`rec_songs`** dataframe to convert the recommended songs into a list of dictionaries with metadata about each song, which is then returned by the function.

![Screenshot 2023-03-28 at 2.47.47 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/79898a65-609e-437b-9c30-77ce297465f0/Screenshot_2023-03-28_at_2.47.47_PM.png)

The **`recommend_songs`**
 function takes a list of songs (**`song_list`**
) and the Spotify dataset (**`spotify_data`**
) as input and returns a list of recommended songs based on the input.

# Output

![Screenshot 2023-03-28 at 2.49.35 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d8d4d3f2-2401-47fb-9049-3e46ce15fbcb/Screenshot_2023-03-28_at_2.49.35_PM.png)

The function **`recommend_songs()`** takes in two arguments: **`song_list`** and **`spotify_data`**.

In this case, the **`song_list`** argument is a list of two dictionaries containing song names and years: **`[{'name': 'I Will Follow', 'year':2010}, {'name': 'Come As You Are', 'year':1991}]`**.

The **`spotify_data`** argument is a Pandas DataFrame that contains data about various songs, including their names, years, and numerical features like tempo, loudness, and danceability.

The function then proceeds to flatten the **`song_list`** input and get the mean vector of numerical features of the input using the **`get_mean_vector()`** function. It then scales both the input mean vector and the numerical features in the **`spotify_data`** using a **`StandardScaler()`** transformer object from a pipeline stored in **`song_cluster_pipeline`**.

Next, the function calculates the cosine distance between the scaled mean input vector and each row in the scaled **`spotify_data`** DataFrame using the **`cdist()`** function. It then sorts the distances in ascending order, takes the top **`n_songs`** songs with the smallest distances, and returns a list of the top **`n_songs`** recommended songs with their metadata columns (**`name`**, **`year`**, and **`artists`**) as a list of dictionaries.

In this case, the function will return a list of 10 recommended songs that are most similar to the mean input vector of the two songs specified in **`song_list`**: songs that are likely to be similar in terms of their musical features, tempo, loudness, etc. The exact songs that are recommended will depend on the contents of the **`spotify_data`** DataFrame.

![Screenshot 2023-03-28 at 2.50.34 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d2c2605-925c-40c4-9921-7e8e8e2103be/Screenshot_2023-03-28_at_2.50.34_PM.png)
