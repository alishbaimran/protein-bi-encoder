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
        self.query_encoder.eval()  # set ESM2 to evaluation mode to make sure no training (frozen)
        for param in self.query_encoder.parameters():
            param.requires_grad = False  # freeze ESM2 parameters
        
        self.passage_encoder = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, protein_sequences, descriptions):
        # encode protein sequences with ESM-2
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_sequences)
        with torch.no_grad():  # no gradients are calculated for ESM2
            results = self.query_encoder(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        protein_embeddings = []
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        for i, tokens_len in enumerate(batch_lens):
            protein_embeddings.append(token_representations[i, 1:tokens_len - 1].mean(0))
        protein_embeddings = torch.stack(protein_embeddings)
        
        # tokenize and encode descriptions with BERT
        inputs = self.tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True)
        description_embeddings = self.passage_encoder(**inputs).last_hidden_state[:, 0, :]
        
        return protein_embeddings, description_embeddings

# ESM-2 model
esm2_model, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# instantiate the bi-encoder model
bi_encoder = BiEncoder(esm2_model, esm2_alphabet)
