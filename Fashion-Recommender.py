# --- Complete Nexora AI Assignment Script (Sentence-Transformers Version) ---

# 1. Install all necessary libraries
# Run this in your terminal or a Colab cell:
# pip install pandas scikit-learn numpy sentence-transformers

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # <-- CHANGED
import timeit

# --- 🎯 Step 1: Data Preparation ---
# (This step is unchanged)

print("Step 1: Preparing Data...")
# Create the mock data as a dictionary
data = {
    'product_id': [101, 102, 103, 104, 105],
    'name': [
        "Boho Dress",
        "Urban Tech Jacket",
        "Cozy Knit Sweater",
        "Classic Linen Shirt",
        "Streetwear Hoodie"
    ],
    'desc': [
        "Flowy, earthy tones for festival vibes. A relaxed, comfortable fit.",
        "Sleek, water-resistant jacket for city life. Energetic, chic, and modern.",
        "A soft, oversized wool sweater perfect for a quiet evening or reading by the fire.",
        "Lightweight and breathable, this shirt is a timeless staple for warm weather.",
        "A bold graphic hoodie representing urban street culture. Pure comfort."
    ],
    'tags': [
        ['boho', 'cozy', 'festival'],
        ['urban', 'modern', 'tech', 'chic'],
        ['cozy', 'warm', 'casual'],
        ['classic', 'casual', 'lightweight'],
        ['streetwear', 'urban', 'bold', 'cozy']
    ]
}

# Create the pandas DataFrame
df_products = pd.DataFrame(data)
print("Product catalog created successfully.")


# --- 🎯 Step 2: Embeddings (Setup & Generation) ---
# (This step is heavily modified)

print("\nStep 2: Setting up Embeddings...")

# 2a. Define and load the local model
# We use a popular, lightweight model from SentenceTransformers
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'... (This may take a moment on first run)")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'sentence-transformers' is installed: pip install sentence-transformers")
    exit()

# 2b. Generate embeddings for all products (one-time operation)
print("Generating embeddings for all product descriptions...")
try:
    # Get all descriptions as a list
    descriptions_list = df_products['desc'].tolist()
    
    # Generate embeddings for all descriptions in one batch
    embeddings_list = model.encode(descriptions_list)
    
    # Add the embeddings (as lists) to the DataFrame
    df_products['embedding'] = list(embeddings_list)
    
    print("Product embeddings generated and added to DataFrame.")
    # print(df_products.head())
except Exception as e:
    print(f"An error occurred during embedding generation: {e}")
    exit()


# --- 🎯 Step 3: Vector Search (Function Definition) ---
# (This step is unchanged - it just needs vectors)

print("\nStep 3: Defining Search Functions...")

def calculate_similarity(query_vec, product_vecs):
    """Calculates cosine similarity between a query vector and a matrix of product vectors."""
    # Ensure inputs are 2D numpy arrays
    query_2d = np.array(query_vec).reshape(1, -1)
    
    # product_vecs might be a Series of lists/arrays, so .tolist() is safest
    product_matrix = np.array(product_vecs.tolist())
    
    sim_scores = cosine_similarity(query_2d, product_matrix)
    
    # Return the 1D array of scores
    return sim_scores[0]

print("Similarity function defined.")


# --- 🎯 Step 4: Test & Eval (Function & Execution) ---
# (This step is modified to use the local model)

print("\nStep 4: Testing & Evaluation...")

def find_top_matches(query_text, product_df, model, top_n=3, threshold=0.7): # <-- Added 'model'
    """
    Full search pipeline:
    1. Gets embedding for the query using the local model.
    2. Calculates similarity against pre-embedded products.
    3. Ranks and returns top_n results, handling edge cases.
    """
    print(f"\n--- Testing Query: '{query_text}' ---")
    
    # 1. Get query embedding (now using the local model)
    try:
        query_vec = model.encode(query_text) # <-- CHANGED
    except Exception as e:
        print(f"Error getting embedding for query '{query_text}': {e}")
        return

    # 2. Calculate similarity (using pre-embedded product vectors)
    scores = calculate_similarity(query_vec, product_df['embedding'])
    
    # Add scores to a temp dataframe for ranking
    results_df = product_df.copy()
    results_df['sim_score'] = scores
    
    # 3. Sort and get top_n
    results_df = results_df.sort_values(by='sim_score', ascending=False)
    top_matches = results_df.head(top_n)

    # 4. Log metrics and handle edge cases
    # Note: Thresholds might need adjustment, as different models 
    # produce different similarity score ranges. 0.7 is a guess.
    # For 'all-MiniLM-L6-v2', a "good" score might be lower, e.g., > 0.4
    threshold = 0.4 # <-- Adjusted threshold for this model
    
    top_score = top_matches.iloc[0]['sim_score']
    
    if top_score < threshold:
        print(f"LOG: No good matches. (Top score: {top_score:.4f} < {threshold})")
        print("Fallback: Sorry, no strong matches found for that vibe.")
    else:
        print(f"LOG: Found {len(top_matches)} matches. (Top score: {top_score:.4f} > {threshold})")
        print("Top recommendations:")
        print(top_matches[['name', 'sim_score']])

# 4a. Run 3 Test Queries
queries_to_test = [
    "energetic urban chic",
    "cozy and warm for winter",
    "something for a music festival"
]

for q in queries_to_test:
    find_top_matches(q, df_products, model) # <-- Pass the model in

# 4b. Plot Latency
print("\n--- Measuring Search Latency ---")
# This times the *local* similarity search.

try:
    # Create a test vector using the local model
    test_query_vec = model.encode("test query") # <-- CHANGED
    
    def time_the_search():
        # Function to be timed (only the similarity calculation)
        calculate_similarity(test_query_vec, df_products['embedding'])

    number_of_runs = 100
    total_time = timeit.timeit(time_the_search, number=number_of_runs)
    avg_time_ms = (total_time / number_of_runs) * 1000 # in milliseconds

    print(f"Average search (cosine sim) latency over {number_of_runs} runs: {avg_time_ms:.6f} ms")

except Exception as e:
    print(f"Could not perform latency test: {e}")

print("\n--- Script execution complete. ---")
