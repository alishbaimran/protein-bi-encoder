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

        # shape is batch size, sequence length (incl special tokens) - ends up being length of the longest sequence as smaller ones are padded, dim of each tokens embedding (1280)
        #print(f"Token representations shape: {token_representations.shape}")
        
        mask = (batch_tokens != self.alphabet.padding_idx).unsqueeze(-1).float()  
        # shape: (batch_size, seq_len, 1)
        # marks non-padding tokens with 1 & padding tokens with 0
        #print(f"Mask shape: {mask.shape}")

        # calculate sum of embeddings excluding padding
        token_representations = token_representations[:, 1:-1, :]  # not including [CLS] and [SEP]
        # shape is [batch_size, seq_len-2, 1280] 
        #print(f"excl cls and sep token rep: {token_representations.shape}")
        mask = mask[:, 1:-1, :]  # not including [CLS] and [SEP]
        # shape is  [batch_size, seq_len-2, 1]
        #print(f"excl cls and sep mask: {mask.shape}")

        # element wise multiplication of token_representations and mask. sums the embeddings along the sequence length dimension. 
        # sum_embeddings has the shape [batch_size, 1280]
        sum_embeddings = (token_representations * mask).sum(dim=1)
        
        # sums embeddings along sequence length 
        # shape is [batch_size, 1280]
        #print(f"Sum embeddings shape: {sum_embeddings.shape}")
        
        # shape is [batch_size, 1]
        count_non_padding = mask.sum(dim=1)
        #print(f"Count non-padding shape: {count_non_padding.shape}")
        
        # shape is [batch_size, 1280]
        protein_embeddings = sum_embeddings / count_non_padding
        
        #print(f"Protein embeddings shape after mean pooling: {protein_embeddings.shape}")
        
        # project protein embeddings to the match dimension as BERT embeddings
        protein_embeddings = self.projection(protein_embeddings)
        #print(f"Protein embeddings shape after projection: {protein_embeddings.shape}")

        # tokenize and encode descriptions with BERT
        inputs = self.tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(protein_embeddings.device) for key, value in inputs.items()}

        # debug: print original descriptions and their tokenized versions
        #for i, desc in enumerate(descriptions[:3]):  # Print the first 3 descriptions for debugging
            #print(f"Original description {i}: {desc}")
            #tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
            #print(f"Tokenized description {i}: {tokens}")

        # attention pool, embeddings of CLS for each sequence in batch
        description_embeddings = self.passage_encoder(**inputs).last_hidden_state[:, 0, :]

        #print(f"Description embeddings shape: {description_embeddings.shape}")
        
        return protein_embeddings, description_embeddings
