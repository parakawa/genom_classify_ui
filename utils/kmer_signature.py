from collections import defaultdict
from itertools import product

def encode_kmer(kmer):
    """convert a k-mer to its integer representation"""
    encoding = {'A': 0b00, 'C': 0b01, 'G': 0b10, 'T': 0b11}
    result = 0
    for base in kmer:
        result = (result << 2) | encoding[base]
    return result 

def decode_kmer(encoded_kmer, k):
    """convert an encoded k-mer back to its string representation"""
    decoding = ['A', 'C', 'G', 'T']
    kmer = []
    for _ in range(k):
        base = encoded_kmer & 0b11  # take the last 2 bits
        kmer.append(decoding[base])
        encoded_kmer >>= 2
    return ''.join(reversed(kmer))

def reverse_complement_bits(kmer, k):
    """compute the reverse complement of a k-mer encoded as an integer"""
    mask = (1 << (2 * k)) - 1  # mask to keep only 2k bits
    complement = 0
    for _ in range(k):
        complement = (complement << 2) | ((kmer & 0b11) ^ 0b11)  # complement the base
        kmer >>= 2
    return complement & mask  # ensure there are no extra bits

def canonical_kmer_bits(kmer, k):
    """return the canonical k-mer in its integer form"""
    rev_comp = reverse_complement_bits(kmer, k)
    return min(kmer, rev_comp)

def generate_all_canonical_kmers_bits(k):
    """generate all canonical k-mers of length k as integers"""
    canonical_kmers = set()
    for kmer_tuple in product([0b00, 0b01, 0b10, 0b11], repeat=k):
        kmer = 0
        for base in kmer_tuple:
            kmer = (kmer << 2) | base
        canonical_kmers.add(canonical_kmer_bits(kmer, k))
    return sorted(canonical_kmers)

def compute_kmer_signature_bits(sequence, k = 2):
    """
    compute the k-mer signature vector for a sequence
    uses bitwise encoding to handle k-mers efficiently
    """
    if len(sequence) < k:
        raise ValueError("Sequence length must be at least k.")

    # dictionary to store frequencies (default value is 0)
    kmer_counts = defaultdict(int)

    # encode the entire sequence into bits
    encoded_sequence = encode_kmer(sequence)

    # mask to extract k-mers
    mask = (1 << (2 * k)) - 1  # mask for the last 2k bits
    for i in range(len(sequence) - k + 1):
        kmer = (encoded_sequence >> (2 * (len(sequence) - k - i))) & mask
        canonical = canonical_kmer_bits(kmer, k)
        kmer_counts[canonical] += 1

    # generate the signature vector
    possible_kmers = generate_all_canonical_kmers_bits(k)
    total_kmers = sum(kmer_counts.values())
    signature_vector = [
        kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0
        for kmer in possible_kmers
    ]

    return signature_vector
