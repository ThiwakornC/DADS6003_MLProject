import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def find_similar_movies(movie_id, n_similar=5):
    if movie_id >= item_embeddings.shape[0]:
        raise ValueError(f"movie_id {movie_id} is out of range. Max valid id is {item_embeddings.shape[0]-1}")
    
    movie_embedding = item_embeddings[movie_id]
    similarities = cosine_similarity([movie_embedding], item_embeddings)
    similar_indices = similarities[0].argsort()[::-1][1:n_similar+1]
    similar_scores = similarities[0][similar_indices]
    
    return list(zip(similar_indices, similar_scores))

def recommend_for_user(user_id, top_n=5):
    if user_id >= user_embeddings.shape[0]:
        raise ValueError(f"user_id {user_id} is out of range. Max valid id is {user_embeddings.shape[0]-1}")
    
    user_emb = user_embeddings[user_id]
    scores = np.dot(user_emb, item_embeddings.T)
    top_movies = np.argsort(scores)[::-1][:top_n]
    return top_movies

def cluster_users(n_clusters=5):
    # จัดกลุ่มผู้ใช้ด้วย KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(user_embeddings)
    
    # สรุปข้อมูลของแต่ละกลุ่ม
    for i in range(n_clusters):
        cluster_size = np.sum(clusters == i)
        print(f"\nCluster {i}:")
        print(f"Number of users: {cluster_size}")
        
        # หาตัวแทนของกลุ่ม (centroid)
        centroid = kmeans.cluster_centers_[i]
        
        # หาหนังที่น่าจะเป็นที่นิยมในกลุ่มนี้
        scores = np.dot(centroid, item_embeddings.T)
        top_movies = np.argsort(scores)[::-1][:5]
        print("Top 5 recommended movies for this cluster:")
        for rank, movie_id in enumerate(top_movies, 1):
            print(f"{rank}. Movie ID: {movie_id}")
    
    return clusters

def get_recommendations(user_id=None, movie_id=None, n_recommendations=5):
    if user_id is not None:
        print(f"\nGetting recommendations for user {user_id}:")
        recommendations = recommend_for_user(user_id, n_recommendations)
        for i, movie_id in enumerate(recommendations, 1):
            print(f"{i}. Movie ID: {movie_id}")
            
    if movie_id is not None:
        print(f"\nFinding similar movies to movie {movie_id}:")
        similar_movies = find_similar_movies(movie_id, n_recommendations)
        for i, (similar_id, score) in enumerate(similar_movies, 1):
            print(f"{i}. Movie ID: {similar_id}, Similarity Score: {score:.3f}")

# Load embeddings
with h5py.File(r'C:\Users\Naphat\Desktop\Docker_ML\Pretrain\ml-1m_NeuMF_32_[64,32]_1736442202.h5', 'r') as f:
    try:
        user_embeddings = np.array(f['mf_embedding_user/mf_embedding_user_W'])
        item_embeddings = np.array(f['mf_embedding_item/mf_embedding_item_W'])
        
        print("=== Model Information ===")
        print(f"Number of users: {user_embeddings.shape[0]}")
        print(f"Number of items: {item_embeddings.shape[0]}")
        print(f"Embedding dimension: {item_embeddings.shape[1]}")
        print("========================\n")

    except Exception as e:
        print("Error:", e)

def main():
    while True:
        print("\n=== Movie Recommendation System ===")
        print("1. Get user recommendations")
        print("2. Find similar movies")
        print("3. Cluster users")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            try:
                user_id = int(input("Enter user ID: "))
                n_rec = int(input("Number of recommendations (default 5): ") or "5")
                get_recommendations(user_id=user_id, n_recommendations=n_rec)
            except ValueError:
                print("Please enter valid numbers!")
                
        elif choice == '2':
            try:
                movie_id = int(input("Enter movie ID: "))
                n_rec = int(input("Number of similar movies (default 5): ") or "5")
                get_recommendations(movie_id=movie_id, n_recommendations=n_rec)
            except ValueError:
                print("Please enter valid numbers!")
                
        elif choice == '3':
            try:
                n_clusters = int(input("Enter number of clusters (default 5): ") or "5")
                clusters = cluster_users(n_clusters)
                print(f"\nClustering complete! Users have been divided into {n_clusters} groups.")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == '4':
            print("Thank you for using the recommendation system!")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()