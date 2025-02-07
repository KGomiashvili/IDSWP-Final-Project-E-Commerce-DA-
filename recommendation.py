import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendations(df, product_id, top_n=5):
    """
    Recommend products based on precomputed features.
    """
    if product_id not in df['product_id'].values:
        return []

    # Load only required features
    features = df[['product_id', 'category_code', 'price', 'brand']].set_index('product_id')
    
    # Compute similarity
    similarity_matrix = cosine_similarity(features)
    
    # Find similar products
    product_index = features.index.get_loc(product_id)
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Return top recommendations
    top_products = [features.index[i[0]] for i in similarity_scores[1:top_n+1]]
    return top_products
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def collaborative_filtering_recommendations(user_id, top_n=5):
    """
    Recommend products using precomputed user-item interaction matrix.
    """
    user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0, )

    if user_id not in user_item_matrix.index:
        return []

    sparse_matrix = csr_matrix(user_item_matrix)

    # Train KNN model once
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
    model.fit(sparse_matrix)

    # Find similar users
    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = model.kneighbors(sparse_matrix[user_index], n_neighbors=top_n+1)

    # Recommend products
    similar_users = user_item_matrix.iloc[indices.flatten()[1:]]
    recommended_products = similar_users.mean().sort_values(ascending=False).head(top_n).index.tolist()

    return recommended_products
