import torch
import esm
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, esm2_model, esm2_alphabet, bert_model_name="bert-base-uncased"):
        super(BiEncoder, self).__init__()
        self.query_encoder = esm2_model
        self.alphabet = esm2_alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.query_encoder.eval()  #  ESM2 to evaluation mode
        for param in self.query_encoder.parameters():
            param.requires_grad = False  # freeze ESM2 parameters
        
        self.passage_encoder = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # linear layer to project protein embeddings to the same dimension as BERT embeddings (TO-DO)
        self.projection = nn.Linear(1280, 768)

    def forward(self, protein_sequences, descriptions):
        # encode protein sequences with ESM-2
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_sequences)
        batch_tokens = batch_tokens.to(next(self.query_encoder.parameters()).device)  
        
        with torch.no_grad():  #  no gradients are calculated for ESM2
            results = self.query_encoder(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        protein_embeddings = []
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        for i, tokens_len in enumerate(batch_lens):
            protein_embeddings.append(token_representations[i, 1:tokens_len - 1].mean(0))
        protein_embeddings = torch.stack(protein_embeddings)
        
        # project protein embeddings to the same dimension as BERT embeddings
        protein_embeddings = self.projection(protein_embeddings)
        
        # tokenize and encode descriptions with BERT
        inputs = self.tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(protein_embeddings.device) for key, value in inputs.items()}
        description_embeddings = self.passage_encoder(**inputs).last_hidden_state[:, 0, :]
        
        return protein_embeddings, description_embeddings
