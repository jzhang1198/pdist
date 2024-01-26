
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import os 


def read_BLOSUM62(
        add_gap: bool = True
        ):
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    blosum62 = os.path.join(root_dir, 'BLOSUM62.txt')
    
    with open(blosum62, 'r') as f:
        blosum62 = f.readlines()

    blosum62 = [l.strip('\n') for l in blosum62]
    alphabet = blosum62[1].replace(' ', '')

    blosum62 = [blosum62[i][2:].split(' ') for i in range(2, len(blosum62))]
    blosum62 = [j for r in blosum62 for j in r if j != '']
    blosum62 = np.array(blosum62, dtype=int).reshape((len(alphabet), len(alphabet)))

    if add_gap:
        blosum62 = np.hstack([blosum62, np.zeros(len(alphabet))[:, np.newaxis]])
        alphabet += '-'
        blosum62 = np.vstack([blosum62, np.zeros(len(alphabet))])

    tok2index = {token:index for index, token in enumerate(alphabet)}

    return blosum62, tok2index


BLOSUM62, TOK2INDEX = read_BLOSUM62()


def distribute_pairs(l: int, task_id: int, n_jobs: int):
    pairs = np.vstack(np.tril_indices(l, k=0)).T
    sele = np.array([i for i in range(len(pairs)) if i % n_jobs == task_id])
    return pairs[sele]


def load_ali(file: str, fmt: str):

    aligned_seqs, ids = zip(*[(str(r.seq), r.id) for r in SeqIO.parse(file, fmt)])
    l, w = len(aligned_seqs), len(aligned_seqs[0])
    ali_arr = np.array(list(''.join([seq for seq in aligned_seqs]))).reshape((l, w))
    return ali_arr, ids


def clean_ali(ali: np.ndarray, remove_lowercase=True):

    # remove all gap columns
    non_gap_columns = np.where(~np.all(ali == '-', axis=0))[0]
    ali = ali[:, non_gap_columns]

    # remove unaligned columns
    if remove_lowercase:
        unaligned_columns = np.where(~np.any(np.char.islower(ali), axis=0))[0]
        ali = ali[:, unaligned_columns]
    else:
        ali = np.char.upper(ali)

    return ali


def pwise_blosum_worker(args):
    tokenized_ali, pair_batch = args
    seq1 = tokenized_ali[pair_batch[:, 0]]
    seq2 = tokenized_ali[pair_batch[:, 1]]
    return BLOSUM62[seq1, seq2].sum(axis=1)


def pwise_blosum(ali_arr: np.ndarray, pairs: np.ndarray, batch_size: int):

    # tokenize the alignment
    def map_func(element):
        return TOK2INDEX.get(element)
    vectorized_map_func = np.vectorize(map_func)
    tokenized_ali = vectorized_map_func(ali_arr)

    similarities = []
    for pair_batch in tqdm([pairs[i:i + int(batch_size), :] for i in range(0, len(pairs), int(batch_size))], desc='Computing pairwise BLOSUM similarities.', unit='batch'):

        # Compute the similarity matrix using broadcasting
        seq1 = tokenized_ali[pair_batch[:, 0]]
        seq2 = tokenized_ali[pair_batch[:, 1]]

        pair_similarities = BLOSUM62[seq1, seq2].sum(axis=1)
        similarities.append(pair_similarities)

    similarities = np.hstack(similarities)
    distance_list = np.hstack([pairs, similarities[:, np.newaxis]])
    return distance_list


def reformat_esl_output(esl_output, ids: np.ndarray):

    lines = esl_output.stdout.split('\n')[1:-1]

    # re-format esl-alipid output
    ids2index = dict([(id, i) for i, id in enumerate(ids)])
    def id_mapper(element):
        return ids2index.get(element)
    vectorized_id_mapper = np.vectorize(id_mapper)

    pid_matrix = np.zeros((len(ids), len(ids)))
    seqid1, seqid2, pid = zip(*[tuple(l.split()[0:3]) for l in lines])
    seqindex1, seqindex2 = vectorized_id_mapper(seqid1), vectorized_id_mapper(seqid2)

    pid_matrix[seqindex1, seqindex2] = pid
    pid_matrix = pid_matrix + pid_matrix.T - np.diag(np.diag(pid_matrix))
    np.fill_diagonal(pid_matrix, 1)
    return pid_matrix


def compute_pwise_identity(ali_arr: np.ndarray, tok2index: dict, batch_size=1e6):
    """ 
    Deprecated function. Now using esl-alipid, which is much more efficient.
    """

    pair_indices = np.vstack(np.tril_indices(len(ali_arr), k=-1)).T
    pid_matrix = np.zeros((len(ali_arr), len(ali_arr)))
    mlens_matrix = np.zeros((len(ali_arr), len(ali_arr)))
    match_matrix = np.zeros((len(ali_arr), len(ali_arr)))

    # obtain mapping between index and sequence length
    seqlens = ali_arr.shape[1] - np.sum(ali_arr == '-', axis=1)
    ind2seqlen = dict([(i,l) for i,l in enumerate(seqlens)])
    def length_map_func(element):
        return ind2seqlen.get(element)
    vectorized_length_map = np.vectorize(length_map_func)

    ali_arr = clean_ali(ali_arr, remove_lowercase=False)
    def map_func(element):
        return tok2index.get(element)
    vectorized_map_func = np.vectorize(map_func)
    tokenized_ali = vectorized_map_func(ali_arr)
    
    gap_token = tok2index['-']

    for pair_batch in tqdm([pair_indices[i:i + int(batch_size), :] for i in range(0, len(pair_indices), int(batch_size))], desc='Computing pairwise sequence identities.', unit='batch'):

        # Compute the similarity matrix using broadcasting
        seq1 = tokenized_ali[pair_batch[:, 0]]
        seq2 = tokenized_ali[pair_batch[:, 1]]

        min_lens = np.vstack([vectorized_length_map(pair_batch[:, 0]), vectorized_length_map(pair_batch[:, 1])]).min(axis=0)
        pair_identities = np.sum((seq1 == seq2) & (seq1 != gap_token) & (seq2 != gap_token), axis=1) / min_lens
        pid_matrix[pair_batch[:,1], pair_batch[:,0]] = pair_identities
        mlens_matrix[pair_batch[:,1], pair_batch[:,0]] = min_lens
        match_matrix[pair_batch[:,1], pair_batch[:,0]] = np.sum((seq1 == seq2) & (seq1 != gap_token) & (seq2 != gap_token), axis=1)

    pid_matrix = pid_matrix + pid_matrix.T - np.diag(np.diag(pid_matrix))
    np.fill_diagonal(pid_matrix, 1)
    
    return pid_matrix, mlens_matrix, match_matrix

