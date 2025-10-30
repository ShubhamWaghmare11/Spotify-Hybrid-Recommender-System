import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



track_ids_save_path = "data/track_ids.npy"
filtered_data_save_path = "data/collab_filtered_data.csv"
interaction_matrix_save_path = "data/interaction_matrix.npz"

songs_data_path = "data/cleaned_data.csv"
user_listening_history_data_path = "data/User Listening History.csv"


def filter_songs_data(songs_data, track_ids, save_df_path):

    filtered_data = songs_data[songs_data['track_id'].isin(track_ids)]


    filtered_data.sort_values(by='track_id', inplace=True)

    filtered_data.reset_index(drop=True, inplace=True)


    filtered_data.to_csv(save_df_path, index=False)
    
    return filtered_data

def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    save_npz(file_path, matrix)


def create_interaction_matrix(history_data, track_ids_save_path, save_matrix_path):
    df = history_data.copy()

    # Ensure proper dtypes
    df['playcount'] = df['playcount'].astype('float64')

    # Convert to categorical
    df = df.categorize(columns=['user_id', 'track_id'])

    # Save mappings
    user_mapping = df['user_id'].cat.categories
    track_mapping = df['track_id'].cat.categories

    np.save(track_ids_save_path, track_mapping.values, allow_pickle=True)

    # Map categorical codes to integers (not categories)
    df = df.assign(
        user_idx=df['user_id'].cat.codes,
        track_idx=df['track_id'].cat.codes
    )

    # Compute the interaction matrix
    interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    interaction_matrix = interaction_matrix.compute()

    # Build sparse matrix
    row_indices = interaction_matrix['track_idx'].astype(int)
    col_indices = interaction_matrix['user_idx'].astype(int)
    values = interaction_matrix['playcount']

    n_tracks = row_indices.max() + 1
    n_users = col_indices.max() + 1

    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    save_sparse_matrix(interaction_matrix, save_matrix_path)


def collaborative_recommendation(song_name, artist_name, track_ids, songs_data, interaction_matrix, k=5):
    
    song_name, artist_name = song_name.lower(), artist_name.lower()

    song_row = songs_data.loc[(songs_data["name"].str.lower() == song_name) & (songs_data["artist"].str.lower() == artist_name)]

    if len(song_row) == 0:
        return pd.DataFrame()

    input_track_id = song_row['track_id'].values.item()

    ind = np.where(track_ids == input_track_id)[0].item()

    input_array = interaction_matrix[ind]

    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    recommendation_track_ids = track_ids[recommendation_indices]

    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]

    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),"score":top_scores})

    top_k_songs = songs_data.loc[songs_data['track_id'].isin(recommendation_track_ids)].merge(scores_df, on='track_id').sort_values(by='score',ascending=False).drop(columns=['track_id','score']).reset_index(drop=True)

    return top_k_songs


def main():
    # load the history data
    user_data = dd.read_csv(user_listening_history_data_path)
    
    user_data = user_data[~user_data['track_id'].isin(['TRCEFVZ128F4283203','TRDTUTO128F422F138'])]

    # get the unique track ids
    unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()
    
    # filter the songs data
    songs_data = pd.read_csv(songs_data_path)
    filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    
    # create the interaction matrix
    create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)


if __name__ == "__main__":
    main()