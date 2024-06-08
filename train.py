from bi_encoder import BiEncoder
from dataset import FastaDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import esm
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb 
import re
from tqdm import tqdm

wandb.init(project="protein-biencoder")
print(torch.__version__)
print(torch.version.cuda)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_file = "/home/alishbaimran/projects/protein_biencoder/output_uniref90.fasta"
fasta_dataset = FastaDataset(data_file, cache_indices=True)

train_size = int(0.8 * len(fasta_dataset))
val_size = len(fasta_dataset) - train_size
train_dataset, val_dataset = random_split(fasta_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

tokenizer_path = "/home/alishbaimran/projects/protein_biencoder/updated_uniref_bpe_tokenizer.json"
bi_encoder = BiEncoder(esm2_model, esm2_alphabet, tokenizer_path)
bi_encoder = bi_encoder.to(device)

# contrastive loss function and accuracy calc 
# mult accuracy by 100
def contrastive_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    loss = F.cross_entropy(logits, torch.arange(b, device=device))

    # calc accuracy
    pred_positive_idx = torch.argmax(logits, dim=1)
    correct = torch.eq(pred_positive_idx, torch.arange(b, device=device)).type(torch.float)
    acc = correct.sum() / b

    return loss, acc

# Set up optimizer
optimizer = optim.Adam(bi_encoder.parameters(), lr=1e-5)  

num_epochs = 150
for epoch in range(num_epochs):
    bi_encoder.train()
    train_loss = 0.0
    train_acc = 0.0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
        for batch in train_loader:
            descs, seqs = batch
            
            optimizer.zero_grad()

            protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
            protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))

            loss, acc = contrastive_loss(protein_embeddings, description_embeddings)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            
            pbar.update(1)
            pbar.set_postfix({'loss': train_loss / pbar.n, 'accuracy': train_acc / pbar.n})

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Training Accuracy: {avg_train_acc}")

    # validation loop
    bi_encoder.eval()
    val_loss = 0.0
    val_acc = 0.0
    with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation") as pbar:
        with torch.no_grad():
            for batch in val_loader:
                descs, seqs = batch
                protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
                protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))

                loss, acc = contrastive_loss(protein_embeddings, description_embeddings)
                val_loss += loss.item()
                val_acc += acc.item()
                
                pbar.update(1)
                pbar.set_postfix({'loss': val_loss / pbar.n, 'accuracy': val_acc / pbar.n})
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    print(f"Epoch: {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_acc}")

    # log to wandb
    wandb.log({
       "epoch": epoch,
       "train_loss": avg_train_loss,
       "train_accuracy": avg_train_acc,
       "val_loss": avg_val_loss,
       "val_accuracy": avg_val_acc,
     }, step=epoch)

wandb.finish()
