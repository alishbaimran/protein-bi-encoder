from bi_encoder import BiEncoder
from dataset import FastaDataset
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import esm
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb 
import re
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from torch.cuda.amp import autocast, GradScaler

print(torch.__version__)
print(torch.version.cuda)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project="protein-biencoder")
    
    device = torch.device(f'cuda:{rank}')
    print(f'Using device: {device}')

    data_file = "/home/alishbaimran/projects/protein_biencoder/uniref90.fasta"
    fasta_dataset = FastaDataset(data_file, cache_indices=True)

    train_size = int(0.8 * len(fasta_dataset))
    val_size = len(fasta_dataset) - train_size
    train_dataset, val_dataset = random_split(fasta_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, sampler=train_sampler, num_workers=4)  
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, sampler=val_sampler, num_workers=4)  

    esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    tokenizer_path = "/home/alishbaimran/projects/protein_biencoder/second_uniref_bpe_tokenizer.json"
    bi_encoder = BiEncoder(esm2_model, esm2_alphabet, tokenizer_path).to(device)
    
    bi_encoder = DDP(bi_encoder, device_ids=[rank], find_unused_parameters=True)

    scaler = GradScaler()

    # contrastive loss function and accuracy calculation
    def contrastive_loss(queries, keys, temperature=0.1):
        b, device = queries.shape[0], queries.device
        logits = queries @ keys.t()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits /= temperature
        loss = F.cross_entropy(logits, torch.arange(b, device=device))

        # calculate accuracy
        pred_positive_idx = torch.argmax(logits, dim=1)
        correct = torch.eq(pred_positive_idx, torch.arange(b, device=device)).type(torch.float)
        acc = correct.sum() / b

        return loss, acc

    optimizer = optim.Adam(bi_encoder.parameters(), lr=1e-5)

    num_epochs = 100
    for epoch in range(num_epochs):
        bi_encoder.train()
        train_loss = 0.0
        train_acc = 0.0

        train_sampler.set_epoch(epoch)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training", disable=rank!=0) as pbar:
            for batch in train_loader:
                descs, seqs = batch

                optimizer.zero_grad()

                protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
                
                with autocast():
                    protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))
                    loss, acc = contrastive_loss(protein_embeddings, description_embeddings)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_acc += acc.item()

                pbar.update(1)
                if pbar.n > 0:
                    pbar.set_postfix({'loss': train_loss / pbar.n, 'accuracy': train_acc / pbar.n})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        if rank == 0:
            print(f"Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Training Accuracy: {avg_train_acc}")

        # validation loop
        bi_encoder.eval()
        val_loss = 0.0
        val_acc = 0.0
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation", disable=rank!=0) as pbar:
            with torch.no_grad():
                for batch in val_loader:
                    descs, seqs = batch
                    protein_sequences = [(f"protein_{i}", seq) for i, seq in enumerate(seqs)]
                    
                    with autocast():
                        protein_embeddings, description_embeddings = bi_encoder(protein_sequences, list(descs))
                        loss, acc = contrastive_loss(protein_embeddings, description_embeddings)

                    val_loss += loss.item()
                    val_acc += acc.item()

                    pbar.update(1)
                    if pbar.n > 0:
                        pbar.set_postfix({'loss': val_loss / pbar.n, 'accuracy': val_acc / pbar.n})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        if rank == 0:
            print(f"Epoch: {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_acc}")

        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_acc,
            }, step=epoch)
            
    if rank == 0:
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
