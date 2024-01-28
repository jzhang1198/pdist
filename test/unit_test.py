#!/usr/bin/env python3

# get absolute path of module
import sys 
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(root_dir)
sys.path.append(module_dir)

from pdist import *
import numpy as np


def pwise_similarity_brute_force(seqs: np.ndarray, pairs: np.ndarray, subsmat: np.ndarray, tokenizer: dict):

    similarities = []
    for pair in pairs:
        seq1, seq2 =seqs[pair[0]], seqs[pair[1]]

        similarity = 0
        for i in range(len(seq1)):
            aa1, aa2 = seq1[i], seq2[i]

            if aa1 == '-' or aa1.islower() or aa2 == '-' or aa2.islower():
                continue
            else:
                similarity += subsmat[tokenizer.get(aa1), tokenizer.get(aa2)]

        similarities.append(similarity)

    return np.array(similarities)


def pwise_pid_brute_force(seqs: np.ndarray, pairs: np.ndarray):
    n_ids, seqlens = [], []
    for pair in pairs:
        seq1, seq2 = seqs[pair[0]], seqs[pair[1]]

        n_id, n1, n2 = 0, 0, 0
        for i in range(len(seq1)):
            aa1, aa2 = seq1[i], seq2[i]

            if aa1 != '-':
                n1 += 1
            
            if aa2 != '-':
                n2 += 1

            if aa1 != '-' and aa2 != '-' and not aa1.islower() and not aa2.islower() and aa1 == aa2:
                n_id += 1

        n_ids.append(n_id)
        seqlens.append(min([n1, n2]))

    return np.array(n_ids) / np.array(seqlens)


if __name__ == '__main__':

    msa_file = './test/test_msa.sto'
    msa = MSA(msa_file, 'stockholm')
    task_id, n_jobs = 0, 1

    blosum62_subsmat, blosum62_tokenizer = read_subsmat('blosum62')
    pid_subsmat, pid_tokenizer = read_subsmat('pid')

    # test tokenization
    msa.tokenize_seqs(blosum62_tokenizer)
    for col in msa._non_consensus_cols:
        characters = set(msa.get_col(col, raw=True))
        for c in characters:
            assert c == '-' or c.islower()
    print('Passed MSA tokenization test.')

    # test blosum similarity calculation
    pairs = distribute_pairs(msa.l, 0, 1)
    distance_calculator = pdist(blosum62_subsmat, 100000)
    distance_list = distance_calculator(msa.tokenized_seqs[:, ~msa.uncovered_mask], pairs, False)
    distance_list_test = pwise_similarity_brute_force(msa._raw_seqs, pairs, blosum62_subsmat, blosum62_tokenizer)
    assert set(distance_list[:, 2] - distance_list_test) == set([0])
    print('Passed similarity calculation test.')

    # test tokenization (again, but with pid tokenizer)
    msa.tokenize_seqs(pid_tokenizer)
    for col in msa._non_consensus_cols:
        characters = set(msa.get_col(col))
        for c in characters:
            assert c == '-' or c.islower()
    print('Passed secondary MSA tokenization test.')

    # test pid calculation
    pairs = distribute_pairs(msa.l, 0, 1)
    distance_calculator = pdist(pid_subsmat, 100000)
    distance_list = distance_calculator(msa.tokenized_seqs, pairs, False)

    seqlens = msa.get_seqlens()
    min_seqlens = np.stack([seqlens[pairs[:, 0]], seqlens[pairs[:, 1]]]).T.min(axis=1)
    distance_list[:, 2] = distance_list[:, 2] / min_seqlens

    distance_list_test = pwise_pid_brute_force(msa._raw_seqs, pairs)
    assert set(distance_list[:, 2] - distance_list_test) == set([0])
    print('Passed percent identity calculation test.')

    






