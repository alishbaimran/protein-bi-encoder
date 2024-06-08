import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from pathlib import Path
from tqdm import tqdm
from dataset import FastaDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stream_descriptions(fasta_dataset, log_every_n=1000):
    for idx in tqdm(range(len(fasta_dataset)), desc="Extracting descriptions"):
        desc, _ = fasta_dataset[idx]
        if idx % log_every_n == 0:
            logger.info(f"Processing description: {desc}")
        yield desc

def train_bpe_tokenizer_on_stream(fasta_file, output_path, vocab_size=30000, log_every_n=100000):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    fasta_dataset = FastaDataset(fasta_file, cache_indices=True)
    descriptions = stream_descriptions(fasta_dataset, log_every_n)
    
    logger.info("Starting tokenizer training")
    tokenizer.train_from_iterator(descriptions, trainer)
    logger.info("Tokenizer training completed")
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    tokenizer.save(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    data_file = Path("/shared/amyxlu/data/uniref90/uniref90.fasta")
    output_path = Path("/home/alishbaimran/projects/protein_biencoder/updated_uniref_bpe_tokenizer.json")
    train_bpe_tokenizer_on_stream(data_file, output_path)
