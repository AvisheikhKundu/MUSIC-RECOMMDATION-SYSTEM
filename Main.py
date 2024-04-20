# Import necessary libraries
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
def load_dataset(file_path):
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0, 1))
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

# Preprocess the data
def preprocess_data(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    return trainset, testset

# Train the recommendation model
def train_model(trainset):
    model = SVD()
    model.fit(trainset)
    return model

# Evaluate the model
def evaluate_model(model, testset):
    predictions = model.test(testset)
    accuracy = rmse(predictions)
    print("RMSE:", accuracy)
    return accuracy

# Generate collaborative filtering recommendations
def collaborative_filtering_recommendations(model, user_id, data):
    recommendations = []
    for item_id in data.raw_item_ids():
        est_rating = model.predict(user_id, item_id).est
        recommendations.append((item_id, est_rating))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# Generate content-based recommendations
def content_based_recommendations(df, user_preferences):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['song_features'])
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    song_indices = pd.Series(df.index, index=df['song_title'])
    
    idx = song_indices[user_preferences['song_title']]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    song_indices = [i[0] for i in sim_scores]
    
    return df[['song_title', 'artist_name']].iloc[song_indices]

# Generate hybrid recommendations (combining collaborative and content-based)
def hybrid_recommendations(collab_rec, content_rec):
    # Combine recommendations from both methods
    hybrid_rec = {}
    for item, rating in collab_rec:
        hybrid_rec[item] = rating
    
    for index, row in content_rec.iterrows():
        if row['song_title'] not in hybrid_rec:
            hybrid_rec[row['song_title']] = 0.5  # Assign a default rating if not present
    
    # Sort recommendations by rating
    hybrid_rec = sorted(hybrid_rec.items(), key=lambda x: x[1], reverse=True)
    
    return hybrid_rec[:10]

# Main function
def main():
    # Load data
    file_path = 'your_dataset.csv'
    data = load_dataset(file_path)
    
    # Preprocess data
    trainset, testset = preprocess_data(data)
    
    # Train model
    model = train_model(trainset)
    
    # Evaluate model
    evaluate_model(model, testset)
    
    # Generate recommendations
    user_id = 'user123'
    collaborative_recommendations = collaborative_filtering_recommendations(model, user_id, data)
    
    df = pd.read_csv(file_path)
    user_preferences = {'song_title': 'example_song_title', 'artist_name': 'example_artist_name'}
    content_based_recommendations = content_based_recommendations(df, user_preferences)
    
    hybrid_recommendations = hybrid_recommendations(collaborative_recommendations, content_based_recommendations)
    
    print("Collaborative Filtering Recommendations:")
    print(collaborative_recommendations[:10])
    
    print("\nContent-Based Recommendations:")
    print(content_based_recommendations)
    
    print("\nHybrid Recommendations:")
    print(hybrid_recommendations)

if __name__ == "__main__":
    main()
