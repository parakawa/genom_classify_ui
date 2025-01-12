from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from utils.kmer_signature import compute_kmer_signature_bits

app = Flask(__name__)

# load grid search results
grid_search_results = pd.read_csv("grid_search_results.csv")

# create a dictionary with the best k for each chunk_size based on accuracy
best_k_for_chunk = (
    grid_search_results.sort_values(by="accuracy", ascending=False)
    .groupby("chunk_size")
    .first()["k"].to_dict()
)

# load all models from the "models" directory
models = {}
for file in os.listdir("models"):
    if file.endswith(".joblib"):
        parts = file.split("_")
        chunk_size = int(parts[1][5:])  # extract chunk_size from "chunk500"
        k = int(parts[2][1])  # extract k from "k2"
        models[(chunk_size, k)] = joblib.load(os.path.join("models", file))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    """classify a gene sequence provided by the user"""
    gene = request.form["gene"]
    gene_size = len(gene)

    # find the closest chunk_size
    chunk_sizes = list(best_k_for_chunk.keys())
    closest_chunk = min(chunk_sizes, key=lambda x: abs(x - gene_size))

    # get the best k for this chunk_size
    best_k = best_k_for_chunk[closest_chunk]

    # load the corresponding model
    model = models.get((closest_chunk, best_k))
    if not model:
        return "model not found for the selected chunk_size and k.", 400

    # compute the k-mer signature
    signature = compute_kmer_signature_bits(gene, best_k)

    # predict the organism
    prediction = model.predict([signature])[0]

    return render_template(
        "result.html",
        gene=gene,
        prediction=prediction,
        chunk_size=closest_chunk,
        k=best_k,
    )

if __name__ == "__main__":
    app.run(debug=True)
