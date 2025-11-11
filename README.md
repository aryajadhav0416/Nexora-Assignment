# Mini Fashion Recommendation System

This project is a simple, lightweight semantic search engine for a fashion product catalog. Instead of matching keywords, it uses **vector embeddings** to understand the "vibe" or "semantic meaning" of a user's query and find the most similar items.

This was built as a solution for the Nexora AI assignment.

## 🚀 How It Works

The core logic is a 3-step process:


1.  **Embed Products:** All product descriptions are converted into numerical vectors (embeddings) using the `sentence-transformers` library (`all-MiniLM-L6-v2` model). This is done once and stored.
2.  **Embed Query:** When a user types a query (e.g., "energetic urban chic"), that query is also converted into a vector using the same model.
3.  **Calculate Similarity:** We use **Cosine Similarity** (from `scikit-learn`) to calculate the mathematical "closeness" between the user's query vector and all of the product vectors.
4.  **Rank & Return:** The products are ranked by their similarity score, and the top 3 matches are returned.

## 🛠️ Tech Stack

* **Python 3**
* **Pandas:** For managing the product data.
* **Sentence-Transformers:** For generating the high-quality text embeddings.
* **Scikit-learn (sklearn):** For calculating cosine similarity.
* **Numpy:** For numerical operations.

## 🏃 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn sentence-transformers
    ```

3.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    The script will load the model, run three test queries, and print the results to the console.

## 💡 Reflection & Future Improvements

This project is a "brute-force" vector search, which is perfect for a small dataset. For a larger (e.g., 1M+ items) catalog, this would be too slow. The next steps would be to:

* **Implement a Vector Database:** Integrate a dedicated vector database like **Pinecone**, **Milvus**, or **Weaviate** to perform an ultra-fast **Approximate Nearest Neighbor (ANN)** search.
* **Build a Hybrid Search:** Combine this semantic search with traditional keyword (BM25) search to get the best of both worlds.
* **Deploy as an API:** Wrap the search logic in a simple **FastAPI** or **Flask** endpoint to make it available as a service.
