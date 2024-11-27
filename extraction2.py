from collections import Counter
from itertools import product
import pandas as pd

df = pd.read_excel("NSCLC_Biomarker_Features.xlsx")

df_normal = pd.read_excel("../normal.xlsx")
sequences = df_normal["Protein_Sequence"].to_list()

# Function to calculate k-mer composition
def kmer_composition(sequence, k=3):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    total_kmers = len(kmers)
    kmer_composition = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_composition

# Generate all possible k-mers of length k
def generate_all_kmers(k=3):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return [''.join(p) for p in product(amino_acids, repeat=k)]

# Create k-mer composition features for all NSCLC sequences
def create_kmer_features(sequences, k=3):
    all_kmers = generate_all_kmers(k)
    features = []
    
    for seq_data in sequences:
        sequence = seq_data
        kmer_comp = kmer_composition(sequence, k)
        feature_vector = {kmer: kmer_comp.get(kmer, 0) for kmer in all_kmers}
        features.append(feature_vector)
    
    return pd.DataFrame(features)


# Generate k-mer features
k = 3  # You can adjust this value to use different k-mers (e.g., 2, 4, etc.)
features_df = create_kmer_features(sequences, k=k)

# We will add the Disease label for NSCLC (which is typically 1 for NSCLC, but adjust as needed)
features_df["Disease_Index"] = 0 # Assuming all sequences here are for NSCLC

# Save the feature matrix to an Excel file
features_df.to_excel("Healthy_Biomarker_Features.xlsx", index=False)
print("NSCLC biomarker features saved to NSCLC_Biomarker_Features.xlsx")

