from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from pymongo import MongoClient
import torch
import numpy as np
from nltk.tokenize import sent_tokenize

# Initialize FastAPI
app = FastAPI()

# Initialize MongoDB client
client = MongoClient("mongodb://localhost:27017/")
db = client['category_db']
collection = db['category_trees']

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Pre-set root categories (trees)
predefined_roots = [
    "Technology",
    "Health",
    "Finance",
    "Education",
    "Environment"
]

# Pre-calculate and cache root embeddings
root_embeddings_cache = {root: None for root in predefined_roots}

# Request body model
class TextInput(BaseModel):
    text: str

# Function to get cosine similarity
def get_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to generate embedding for a category
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Initialize root embeddings cache if not already cached
def get_root_embeddings():
    for root in root_embeddings_cache:
        if root_embeddings_cache[root] is None:
            root_embeddings_cache[root] = get_embedding(root)
    return root_embeddings_cache

# Function to find the closest tree (root category)
def find_closest_tree(text_embedding):
    root_embeddings = get_root_embeddings()
    similarities = [get_similarity(text_embedding, root_embedding) for root_embedding in root_embeddings.values()]
    closest_tree_index = np.argmax(similarities)
    return closest_tree_index

# Function to generate hierarchical categories with sentence-level embeddings and hierarchical clustering
def generate_hierarchical_categories(text, distance_threshold=1.5):
    # Step 1: Sentence tokenization
    sentences = sent_tokenize(text)
    
    # Step 2: Generate embeddings for each sentence
    sentence_embeddings = []
    for sentence in sentences:
        embedding = get_embedding(sentence)
        sentence_embeddings.append(embedding)
    
    sentence_embeddings = np.vstack(sentence_embeddings)  # Stack all sentence embeddings into a matrix
    
    # Step 3: Perform Agglomerative Clustering on sentence embeddings
    clustering_model = AgglomerativeClustering(
        n_clusters=None,  # We set this to None to use the distance_threshold
        affinity='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    cluster_labels = clustering_model.fit_predict(sentence_embeddings)
    
    # Step 4: Group sentences based on clusters
    categories = {i: [] for i in np.unique(cluster_labels)}
    for i, cluster in enumerate(cluster_labels):
        categories[cluster].append(sentences[i])
    
    # Step 5: Return hierarchy as a list of categories
    return [f"Category {i}: {', '.join(sentences)}" for i, sentences in categories.items()]

# FastAPI route to categorize text
@app.post("/categorize")
def categorize(input_data: TextInput):
    text = input_data.text

    # Step 1: Generate embedding for the input text
    text_embedding = get_embedding(text)

    # Step 2: Find the closest tree based on similarity
    closest_tree_index = find_closest_tree(text_embedding)
    closest_tree = predefined_roots[closest_tree_index]

    # Step 3: Generate hierarchical categories using hierarchical clustering
    categories = generate_hierarchical_categories(text, distance_threshold=1.5)

    # Step 4: Retrieve the tree from MongoDB (or initialize if not present)
    tree = collection.find_one({"root": closest_tree})
    if not tree:
        tree = {"root": closest_tree, "categories": []}

    # Step 5: Add the new categories under the closest tree
    tree['categories'].append({
        "text": text,
        "generated_categories": categories
    })

    # Step 6: Save the updated tree back to MongoDB
    collection.update_one(
        {"root": closest_tree},
        {"$set": {"categories": tree['categories']}},
        upsert=True
    )

    # Step 7: Return the message, closest tree, and categories
    return {
        "message": "Categories successfully generated",
        "closest_tree": closest_tree,
        "categories": categories
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
