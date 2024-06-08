from Bio import SeqIO

input_file = "/shared/amyxlu/data/uniref90/uniref90.fasta"
output_file = "/home/alishbaimran/projects/protein_biencoder/uniref90.fasta"
max_length = 100  # specify your maximum length here

# function to filter sequences
def filter_sequences(input_file, output_file, max_length):
    with open(output_file, "w") as out_f:
        for record in SeqIO.parse(input_file, "fasta"):
            if len(record.seq) <= max_length:
                SeqIO.write(record, out_f, "fasta")

filter_sequences(input_file, output_file, max_length)
