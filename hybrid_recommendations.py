import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    """
    Hybrid recommender that combines content-based and collaborative filtering results
    using weighted normalized similarity scores, aligned by track_id.
    """

    def __init__(self,
                 number_of_recommendations: int,
                 weight_content_based: float):

        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1.0 - weight_content_based


    # -------------------------------------------------------------------------
    # Similarity Calculations
    # -------------------------------------------------------------------------

    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        song_row = songs_data.loc[(songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)]

        song_index = song_row.index[0]

        input_vector = transformed_matrix[song_index]

        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        return content_similarity_scores

    def __calculate_collaborative_similarities(self, song_name, artist_name, songs_data, track_ids, interaction_matrix):
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        input_track_id = song_row['track_id'].values.item()
        ind = np.where(track_ids == input_track_id)[0].item()

        input_array = interaction_matrix[ind]

        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return collaborative_similarity_scores

    def __normalize_similarities(self, similarity_scores):
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    
    def __weighted_combination(self, content_based_scores, collaborative_based_scores):
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_based_scores)
        return weighted_scores
    
    def give_recommendation(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        # Calculate content-based similarities
        content_similarity_scores = self.__calculate_content_based_similarities(song_name, artist_name, songs_data, transformed_matrix)
        normalized_content_scores = self.__normalize_similarities(content_similarity_scores)

        # Calculate collaborative filtering similarities
        collaborative_similarity_scores = self.__calculate_collaborative_similarities(song_name, artist_name, songs_data, track_ids, interaction_matrix)
        normalized_collaborative_scores = self.__normalize_similarities(collaborative_similarity_scores)

        # Combine scores
        hybrid_scores = self.__weighted_combination(normalized_content_scores, normalized_collaborative_scores)

        # Get top-N recommendations
        recommendation_indices = np.argsort(hybrid_scores.ravel())[-self.number_of_recommendations-1:][::-1]
        
        recommendation_track_ids = track_ids[recommendation_indices]
        top_scores = np.sort(hybrid_scores.ravel())[-self.number_of_recommendations-1:][::-1]

        scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),"score": top_scores})

        top_k_songs = (
                    songs_data
                    .loc[songs_data['track_id'].isin(recommendation_track_ids)]
                    .merge(scores_df, on='track_id')
                    .sort_values(by='score', ascending=False)
                    .drop(columns=['track_id','score'])
                    .reset_index(drop=True)
        )

        return top_k_songs
        
        


        
        