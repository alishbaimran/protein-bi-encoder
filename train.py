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
import re

#wandb.init(project="protein-biencoder")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_file = "/home/alishbaimran/projects/protein_biencoder/output_small.fasta"
fasta_dataset = FastaDataset(data_file, cache_indices=True)

# split dataset into training and validation sets
train_size = int(0.8 * len(fasta_dataset))
val_size = len(fasta_dataset) - train_size
train_dataset, val_dataset = random_split(fasta_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# load ESM-2 model
esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# instantiate the bi-encoder model
bi_encoder = BiEncoder(esm2_model, esm2_alphabet)
bi_encoder = bi_encoder.to(device)

# cosine similarity and loss function
#def cosine_loss(protein_embedding, description_embedding):
    #return 1 - cosine_similarity(protein_embedding, description_embedding).mean()

# contrastive loss instead of cosine similarity 
# diagonals become correct, take softmax 

# contrastive loss function
def contrastive_loss(protein_embeddings, description_embeddings, margin=1.0):
    # pairwise cosine similarities
    similarities = torch.mm(protein_embeddings, description_embeddings.t())
    
    # labels, 1 for positive pairs, 0 for negative pairs
    batch_size = protein_embeddings.size(0)
    target_labels = torch.eye(batch_size, device=protein_embeddings.device)
    
    # contrastive loss
    pos_loss = target_labels * torch.pow(1 - similarities, 2)
    neg_loss = (1 - target_labels) * torch.pow(torch.clamp(similarities - margin, min=0.0), 2)
    loss = torch.mean(pos_loss + neg_loss)
    
    return loss

# Set up optimizer
optimizer = optim.Adam(bi_encoder.passage_encoder.parameters(), lr=1e-5)  # optimize BERT parameters

num_epochs = 50
for epoch in range(num_epochs):
    bi_encoder.train()
    train_loss = 0.0
    for batch in train_loader:
        descs, seqs = batch
        optimizer.zero_grad()

        # prepare data for ESM-2 input format
        protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]

        #for i, seq in enumerate(seqs[:3]): 
            #print(f"Original sequence {i}: {seq}")

        print(f"Batch size: {len(protein_sequences)}, Protein sequence lengths: {[len(seq) for _, seq in protein_sequences]}")

        #for i, desc in enumerate(descs[:3]):  
            #print(f"Original description {i}: {desc}")

        protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))

        #for i, desc in enumerate(descs[:3]):  
            #tokens = bi_encoder.tokenizer.tokenize(desc)
            #print(f"Tokenized description {i}: {tokens}")
        
        #inputs = bi_encoder.tokenizer(list(descs[:3]), return_tensors='pt', padding=True, truncation=True)
        #for i, desc in enumerate(descs[:3]):  
            #tokens = bi_encoder.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
            #print(f"Original description: {desc}")
            #print(f"Tokenized description {i}: {tokens}")

        print(f"Protein embeddings shape: {protein_embeddings.shape}, Description embeddings shape: {description_embeddings.shape}")

        loss = contrastive_loss(protein_embeddings, description_embeddings)
        
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

            loss = contrastive_loss(protein_embeddings, description_embeddings)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

    # log losses to wandb
    #wandb.log({
       #"epoch": epoch,
       # "train_loss": avg_train_loss,
       # "val_loss": avg_val_loss,
    #}, step=epoch)

#wandb.finish()
