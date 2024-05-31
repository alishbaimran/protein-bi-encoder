from dataset import FastaDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Path to files
data_file = "/shared/amyxlu/data/uniref90/small.fasta"

# Initialize the FastaDataset
fasta_dataset = FastaDataset(data_file, cache_indices=True)

dataloader = DataLoader(fasta_dataset, batch_size=32, shuffle=True)

# each batch is a list of 2 tuples. the first tuple has all the descriptions stored as strings and the second tuple has all sequences stored as strings
for batch in dataloader:
    descs, seqs = batch
    for desc, seq in zip(descs, seqs):
        print(f"Description: {desc}")
        print(f"Sequence: {seq}")
  
#desc, seq = fasta_dataset.get(0)
#print(f"Description: {desc}")
#print(f"Sequence: {seq}")


