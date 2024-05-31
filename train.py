from bi_encoder import BiEncoder
from dataset import FastaDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import esm
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
import wandb 

wandb.init(project="protein-biencoder")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_file = "/home/alishbaimran/projects/protein_biencoder/uniref90.fasta"
fasta_dataset = FastaDataset(data_file, cache_indices=True)

# split dataset into training and validation sets
train_size = int(0.8 * len(fasta_dataset))
val_size = len(fasta_dataset) - train_size
train_dataset, val_dataset = random_split(fasta_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# load ESM-2 model
esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# instantiate the bi-encoder model
bi_encoder = BiEncoder(esm2_model, esm2_alphabet)
bi_encoder = bi_encoder.to(device)

# cosine similarity and loss function
def cosine_loss(protein_embedding, description_embedding):
    return 1 - cosine_similarity(protein_embedding, description_embedding).mean()

# Set up optimizer
optimizer = optim.Adam(bi_encoder.passage_encoder.parameters(), lr=1e-5)  # optimize BERT parameters

num_epochs = 100
for epoch in range(num_epochs):
    bi_encoder.train()
    train_loss = 0.0
    for batch in train_loader:
        descs, seqs = batch
        optimizer.zero_grad()

        # prepare data for ESM-2 input format
        protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]

        print(f"Batch size: {len(protein_sequences)}, Protein sequence lengths: {[len(seq) for _, seq in protein_sequences]}")

        protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))

        print(f"Protein embeddings shape: {protein_embeddings.shape}, Description embeddings shape: {description_embeddings.shape}")

        loss = cosine_loss(protein_embeddings, description_embeddings)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")

    # validation loop
    bi_encoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            descs, seqs = batch
            protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
            protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))

            loss = cosine_loss(protein_embeddings, description_embeddings)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

    # log losses to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }, step=epoch)

wandb.finish()
