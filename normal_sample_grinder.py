import requests
import pandas as pd
import time

# Function to fetch UniProt IDs for a specific query
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

# Function to fetch protein sequence for a specific UniProt ID
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if request fails
    fasta_data = response.text
    sequence = "".join(fasta_data.split("\n")[1:])  # Skip the header line and join the sequence lines
    return sequence

# Query to fetch normal human protein sequences
query = 'taxonomy_id:9606 AND reviewed:true'

try:
    # Fetch UniProt data
    tsv_data = fetch_uniprot_ids(query, limit=250)
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
    tsv_data = ""

# Parse the TSV data and fetch sequences
normal_sequences = []
if tsv_data:
    for line in tsv_data.splitlines()[1:]:  # Skip header line
        uniprot_id = line.split("\t")[0]  # Get the UniProt ID (first column)
        try:
            sequence = fetch_uniprot_sequence(uniprot_id)
            normal_sequences.append({"UniProt_ID": uniprot_id, "Protein_Sequence": sequence, "Disease_Index": "Normal"})
        except Exception as e:
            print(f"Failed to fetch sequence for {uniprot_id}: {e}")
        time.sleep(1)  # Sleep to avoid hitting the server too frequently

# Convert data to DataFrame
df_normal = pd.DataFrame(normal_sequences)

# Save the data to an Excel file
output_file = "Normal_Human_Protein_Sequences.xlsx"
df_normal.to_excel(output_file, index=False)
print(f"Normal human protein sequences saved to {output_file}")