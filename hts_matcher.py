# -*- coding: utf-8 -*-
"""
Matching funciton used to retrieve a stacked row from a description input by the user.
The function uses SentenceTransformer word embeddings to compute cosine similarity between
a description entered by the user and every stacked description field. This module when run
by itself returns the top n matches based on the input.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

class HTSMatcherEmbeddingsLocal:
    def __init__(self, file_path: Path, model_name="all-MiniLM-L6-v2"):
        """
        Load HTS data with embeddings if exists, otherwise compute embeddings.
        """
        if file_path.suffix == ".csv":
            self.df = pd.read_csv(file_path)
        elif file_path.suffix in [".parquet", ".pq"]:
            if file_path.exists():
                self.df = pd.read_parquet(file_path)
            else:
                # Try CSV fallback
                csv_path = file_path.with_suffix(".csv")
                if csv_path.exists():
                    print(f"Parquet not found, loading CSV {csv_path} and will rebuild Parquet.")
                    self.df = pd.read_csv(csv_path)
                else:
                    raise FileNotFoundError(f"Neither {file_path} nor {csv_path} exist.")
        else:
            raise ValueError("File must be CSV or Parquet.")
            
        # Normalize column names to safe names
        self.df.columns = [col.strip().replace(" ", "_") for col in self.df.columns]

        # Create readable description if not exists
        if "Readable_Description" not in self.df.columns:
            self.df["Readable_Description"] = self.df["Full_Description"].apply(
                lambda x: " → ".join([lvl.strip() for lvl in str(x).split(": :")])
            )

        # Load or compute embeddings
        if "Embedding" not in self.df.columns:
            print("Computing embeddings for HTS descriptions...")
            self.model = SentenceTransformer(model_name)
            embeddings = []
            for desc in tqdm(self.df["Full_Description"], desc="Embeddings"):
                embeddings.append(self.model.encode(str(desc)))
            self.df["Embedding"] = embeddings
            # Save as Parquet for future use
            parquet_path = file_path.with_suffix(".parquet")
            self.df.to_parquet(parquet_path, index=False)
            print(f"Embeddings saved to {parquet_path}")
        else:
            print(f"Loaded HTS data with embeddings from {file_path}")

        # Load model for similarity search
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(model_name)

    def match(self, query: str, top_n=5, interactive=True):
        query_embedding = self.model.encode(query)

        # Compute similarity scores
        scores = cosine_similarity(
            np.vstack(self.df["Embedding"].values), query_embedding.reshape(1, -1)
        ).flatten()
        self.df["score"] = scores

        # Sort top results
        results = self.df[self.df["score"] > 0.1].sort_values(by="score", ascending=False).head(top_n)

        if results.empty:
            print("No matches found.")
            return None

        # Display results interactively
        if interactive:
            for i, row in enumerate(results.itertuples(index=False), start=1):
                # Access via _asdict to avoid AttributeErrors
                row_dict = row._asdict()
                print(f"[{i}] Score: {row_dict['score']:.3f}\n    HTS Number: {row_dict.get('HTS_Number', 'N/A')}\n    {row_dict.get('Readable_Description', 'N/A')}\n")

            while True:
                choice = input(f"Select the best match (1-{len(results)}), or 0 to cancel: ")
                if choice.isdigit():
                    choice = int(choice)
                    if choice == 0:
                        return None
                    elif 1 <= choice <= len(results):
                        return results.iloc[choice - 1]
                print("Invalid selection. Try again.")

        return results.iloc[0]  # fallback return top match
    
# Example usage
if __name__ == "__main__":

    csv_path = Path(r"C:\Users\Carus\OneDrive\Desktop\Capstone\HTS_data\hts_1_97_stacked.parquet")
    matcher = HTSMatcherEmbeddingsLocal(csv_path)  # <-- this will load embeddings directly
    query = input("Enter product description to search: ")
    selected_row = matcher.match(query, top_n=10, interactive=True)

    if selected_row is not None:
        print("\nYou selected:")
        print(f"HTS Number: {selected_row['HTS_Number']}")
        print(f"Score: {selected_row['score']:.3f}")
        print(f"Full Description: {selected_row['Full_Description']}")