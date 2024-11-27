import requests
import pandas as pd
from collections import Counter
from itertools import product

# Function to fetch UniProt IDs for NSCLC biomarkers
def fetch_uniprot_ids(query, limit=250):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,id,protein_name,gene_names,organism_name",
        "size": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error if request fails
    return response.text

# Focus query on NSCLC biomarkers and proteins
query_nslc = "Non-Small Cell Lung Carcinoma AND (EGFR OR ALK OR KRAS OR PD-L1 OR VEGF) AND taxonomy_id:9606"



# Function to fetch protein sequences for specific UniProt IDs
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if request fails
    fasta_data = response.text
    sequence = "".join(fasta_data.split("\n")[1:])  # Skip the header line and join the sequence lines
    return sequence

# Fetch NSCLC biomarker data
def fetch_nslc_data(query, limit=250):
    print(f"Fetching NSCLC data for query: {query}")
    tsv_data = fetch_uniprot_ids(query, limit=limit)
    if not tsv_data.strip():
        print("No data found for NSCLC biomarkers")
        return []
    
    uniprot_sequences = []
    for line in tsv_data.splitlines()[1:]:
        uniprot_id = line.split("\t")[0]  # Get the UniProt ID (first column)
        try:
            sequence = fetch_uniprot_sequence(uniprot_id)
            uniprot_sequences.append({"UniProt_ID": uniprot_id, "Protein_Sequence": sequence})
        except Exception as e:
            print(f"Failed to fetch sequence for {uniprot_id}: {e}")
    return uniprot_sequences



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
        sequence = seq_data["Protein_Sequence"]
        kmer_comp = kmer_composition(sequence, k)
        feature_vector = {kmer: kmer_comp.get(kmer, 0) for kmer in all_kmers}
        features.append(feature_vector)
    
    return pd.DataFrame(features)

# Now, for each sequence, generate k-mer features and prepare for machine learning
nslc_data = fetch_nslc_data(query_nslc, limit=250)  # Fetch NSCLC biomarker data

# Generate k-mer features
k = 3  # You can adjust this value to use different k-mers (e.g., 2, 4, etc.)
features_df = create_kmer_features(nslc_data, k=k)

# We will add the Disease label for NSCLC (which is typically 1 for NSCLC, but adjust as needed)
features_df["Disease_Index"] = 1  # Assuming all sequences here are for NSCLC

# Save the feature matrix to an Excel file
features_df.to_excel("NSCLC_Biomarker_Features.xlsx", index=False)
print("NSCLC biomarker features saved to NSCLC_Biomarker_Features.xlsx")

