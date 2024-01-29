
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import subprocess
import os 


format_dict = {
    'sto': 'stockholm',
    'st': 'stockholm',
    'fa': 'fasta',
    'fasta': 'fasta',
    'a2m': 'fasta',
    'clustal': 'clustal'
}


class MSA:
    def __init__(
            self, 
            file: str, 
            fmt: str                
        ):
        """ 
        Data class for modeling multiple sequence alignments.

        NOTE: By default, this class will interpret lowercase characters in an
        alignment as unaligned residues and will remove them. This behaviour can
        be toggled by setting remove_lowercase to False.
        """

        try:
            seqs, ids = zip(*[(str(r.seq), r.id) for r in SeqIO.parse(file, fmt)])
            ids = np.array(ids)
        except ValueError as e:
            print(f'''Issue parsing the input MSA file. Ensure that you are using a format compatible with BioPython's SeqIO.parse() function.''')
            raise 

        l, w = len(seqs), len(seqs[0])
        seqs = np.array(list(''.join([seq for seq in seqs]))).reshape((l, w))

        self._raw_seqs = np.copy(seqs)

        # remove columns containing all gaps or unaligned characters
        unaligned_indices = np.where(np.char.islower(seqs))
        seqs[unaligned_indices[0], unaligned_indices[1]] = np.array((['?'] * len(unaligned_indices[0])))

        self.unaligned_mask = seqs == '?'
        self.gap_mask = seqs == '-'
        self.uncovered_mask = np.all(np.logical_or(self.unaligned_mask, self.gap_mask), axis=0)

        self.seqs, self.ids, self.l = seqs, ids, l
        self.tokenized_seqs = None

        # attributes for internal testing purposes only
        self._non_consensus_cols = np.where(self.uncovered_mask)[0]
        
    def get_col(self, j: int, raw: bool = True):
        if raw:
            return self._raw_seqs[:, j]
        else:
            return self.seqs[:, j]

    def tokenize_seqs(self, tokenizer: dict):

        def map_func(element):
            return tokenizer.get(element)
        
        vectorized_map_func = np.vectorize(map_func)
        self.tokenized_seqs = vectorized_map_func(self.seqs)

    def get_seqlens(self):
        return (~self.gap_mask).sum(axis=1)


def split_batches(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def pwise_distance(seqs, pair_batches, subsmat, verbose, cpu_index):

    pair_batches = tqdm(pair_batches, desc=f'Computing pairwise distances on CPU {cpu_index}.', unit='batch') if verbose else pair_batches

    distances = []
    for pair_batch in pair_batches:
        seqpairs = np.stack([seqs[pair_batch[:, 0]], seqs[pair_batch[:, 1]]])
        distances.append(subsmat[seqpairs[0], seqpairs[1]].sum(axis=1))
    distances = np.hstack(distances)

    return distances

    
def get_subsmat_dir():
    root_dir = os.path.dirname(__file__)
    subsmat_dir = os.path.join(os.path.dirname(root_dir), 'subsmats')
    return subsmat_dir


def read_subsmat(subsmat: str):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    subsmat_dir = os.path.join(os.path.dirname(root_dir), 'subsmats')
    available_subsmats = os.listdir(subsmat_dir)

    if '{}.npz'.format(subsmat) not in available_subsmats:
        raise Exception('ERROR: {} is not a valid substitution matrix. Ensure that your substitution matrix is saved as an npz file, per the instructions in the README. Available substitution matrices are: {}'.format(subsmat, ', '.join(available_subsmats)))

    subsmat = os.path.join(subsmat_dir, '{}.npz'.format(subsmat))
    subsmat_data = np.load(subsmat)

    if 'alphabet' not in subsmat_data.keys() or 'subsmat' not in subsmat_data.keys():
        raise Exception('''ERROR: {} is not a valid substitution matrix. Ensure that your npz file includes 'alphabet' and 'subsmat' arrays, per the instructions in the README.''')

    subsmat, alphabet = subsmat_data['subsmat'], subsmat_data['alphabet']

    if '-' not in alphabet:
        alphabet =  np.append(alphabet, '-')
        subsmat = np.vstack([subsmat, np.zeros(len(subsmat))])
        subsmat = np.hstack([subsmat, np.zeros(len(subsmat))[:, np.newaxis]])

    if '?' not in alphabet:
        alphabet =  np.append(alphabet, '?')
        subsmat = np.vstack([subsmat, np.zeros(len(subsmat))])
        subsmat = np.hstack([subsmat, np.zeros(len(subsmat))[:, np.newaxis]])

    tok2index = {tok:i for i, tok in enumerate(alphabet)}

    return subsmat, tok2index


def distribute_pairs(l: int, task_id: int, n_jobs: int):
    pairs = np.vstack(np.tril_indices(l, k=0)).T
    sele = np.array([i for i in range(len(pairs)) if i % n_jobs == task_id])
    return pairs[sele]


def run_esl_alipid(msa_file: str, msa_fmt: str, ids: np.ndarray, outdir: str):
    """ 
    Deprecated function for running esl-alipid (a miniapp from the hmmer) and 
    reformatting its output into a matrix.
    """

    esl_command = ' '.join([
        'esl-alipid',
        '--informat',
        msa_fmt,
        msa_file
    ])
    esl_output = subprocess.run(esl_command, shell=True, capture_output=True, text=True)
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