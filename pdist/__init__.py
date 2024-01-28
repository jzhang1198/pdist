
from multiprocessing import Pool
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import os 


class MSA:
    def __init__(
            self, 
            file: str, 
            fmt: str,
            remove_lowercase: bool = True
            ):
        """ 
        Data class for modeling multiple sequence alignments.

        NOTE: By default, this class will interpret lowercase characters in an
        alignment as unaligned residues and will remove them. This behaviour can
        be toggled by setting remove_lowercase to False.
        """

        seqs, ids = zip(*[(str(r.seq), r.id) for r in SeqIO.parse(file, fmt)])
        l, w = len(seqs), len(seqs[0])
        seqs = np.array(list(''.join([seq for seq in seqs]))).reshape((l, w))

        # clean the seqs 
        non_gap_columns = np.where(~np.all(seqs == '-', axis=0))[0]
        seqs = seqs[:, non_gap_columns]

        # remove unaligned columns
        if remove_lowercase:
            unaligned_columns = np.where(~np.any(np.char.islower(seqs), axis=0))[0]
            seqs = seqs[:, unaligned_columns]
        else:
            seqs = np.char.upper(seqs)

        self.seqs, self.ids = seqs, ids


class pdist:

    def __init__(
            self,
            subsmat: np.ndarray,
            tokenizer: dict, 
            batch_size: int,
            n_cpus: int,
            pid: bool
    ):
        
        self.subsmat = subsmat
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.pid = pid
        self.gap_token_idx = tokenizer.get('-')

    @staticmethod
    def _pdist(seqs: np.ndarray, pair_batch: np.ndarray, subsmat: np.ndarray):
        seqpairs = np.stack([seqs[pair_batch[:, 0]], seqs[pair_batch[:, 1]]])
        pair_distances = subsmat[seqpairs[0], seqpairs[1]].sum(axis=1)
        return pair_distances

    @staticmethod
    def _pdist_pid(seqs: np.ndarray, pair_batch: np.ndarray, subsmat: np.ndarray, gap_token_idx: int):
        seqpairs = np.stack([seqs[pair_batch[:, 0]], seqs[pair_batch[:, 1]]])
        seqlens = np.min((seqpairs != gap_token_idx).sum(axis=2), axis=0)
        pair_n_ids = subsmat[seqpairs[0], seqpairs[1]].sum(axis=1)
        return pair_n_ids / seqlens


    def __call__(self, seqs: np.ndarray, pairs: np.ndarray, verbose: bool):

        # tokenize seqs
        def map_func(element):
            return self.tokenizer.get(element)
        vectorized_map_func = np.vectorize(map_func)
        seqs = vectorized_map_func(seqs)

        log_str = 'BLOSUM similarities' if not self.pid else 'sequence identities'
        pair_batches = [pairs[i:i + int(self.batch_size), :] for i in range(0, len(pairs), int(self.batch_size))]
        pair_batches = tqdm(pair_batches, desc='Computing pairwise {}.'.format(log_str), unit='batch') if verbose else pairs
        
        if self.n_cpus == 1:
            distances = []
            for pair_batch in pair_batches:

                if not self.pid:
                    distances.append(pdist._pdist(seqs, pair_batch, self.subsmat)) 

                else:
                    distances.append(pdist._pdist_pid(seqs, pair_batch, self.subsmat, self.gap_token_idx))

        else:
            with Pool(self.n_cpus) as pool:

                if not self.pid:
                    args = [(s,b,m) for s, b, m in zip(list(pair_batch), )]

                else:
                    args = [() ]

                pass
            pass

        distances = np.hstack(distances)
        weighted_adj_list = np.hstack([pairs, distances[:, np.newaxis]])
        return weighted_adj_list


def read_subsmat(subsmat: str):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    subsmat_dir = os.path.join(os.path.dirname(root_dir), 'subsmats')
    available_subsmats = os.listdir(subsmat_dir)

    if '{}.npz'.format(subsmat) not in available_subsmats:
        raise Exception('ERROR: {} is not a valid substitution matrix. Ensure that your substitution matrix is saved as an npz file, per the instructions in the README. Available substitution matrices are: {}'.format(subsmat, ', '.join(available_subsmats)))

    subsmat = os.path.join(subsmat_dir, '{}.npz'.join())
    subsmat_data = np.load(subsmat)

    if 'alphabet' not in subsmat_data.keys() or 'subsmat' not in subsmat_data.keys():
        raise Exception('''ERROR: {} is not a valid substitution matrix. Ensure that your npz file includes 'alphabet' and 'subsmat' arrays, per the instructions in the README.''')

    tok2index = {tok:i for i, tok in enumerate(subsmat_data['alphabet'])}

    return subsmat_data['subsmat'], tok2index


def distribute_pairs(l: int, task_id: int, n_jobs: int):
    pairs = np.vstack(np.tril_indices(l, k=0)).T
    sele = np.array([i for i in range(len(pairs)) if i % n_jobs == task_id])
    return pairs[sele]


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

