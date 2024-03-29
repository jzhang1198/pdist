#!/usr/bin/env python3

from multiprocessing import Pool, RLock
from pdist import *
import numpy as np
import argparse
import os 


def init_parser():
    available_subsmats = [f.split('.')[0] for f in os.listdir(get_subsmat_dir())]
    parser = argparse.ArgumentParser(description='Calculates pairwise distances between aligned sequences.')
    parser.add_argument('outpath', type=str, help='Path to output directory.')
    parser.add_argument('msa_file', type=str, help='Path to multiple sequence alignment file.')
    parser.add_argument('--subsmat', type=str, default='pid', help='The substitution matrix to use. If left unassigned, the program will default to calculating percent identities. Available subsmats: {}'.format(', '.join(available_subsmats)))
    parser.add_argument('--msa-fmt', type=str, default='stockholm', help='File format of the MSA. Can be any format compatible with SeqIO.read() from BioPython.')
    parser.add_argument('--batch-size', type=int, default=1000000, help='Batch size of pairs used during calculation of BLOSUM62 similarities. If you have a ver large number of very long sequences, you might consider decreasing this to avoid memory issues.')
    parser.add_argument('--n-cpus', type=int, default=1, help='Number of CPUs (on the same device) to parallelize distance calculations over. Number of available CPUs can be determined with os.cpu_count().')
    parser.add_argument('--v', action='store_true', help='Turn on verbose output.')
    return parser


if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()

    msa_file, msa_fmt = args.msa_file, args.msa_fmt
    subsmat_type = args.subsmat
    calculate_pid = True if subsmat_type == 'pid' else False
    outdir = args.outpath 
    batch_size = int(args.batch_size)
    n_cpus = int(args.n_cpus)
    verbose = True if args.v else False

    task_id = int(os.environ.get("SGE_TASK_ID", 1)) - 1
    n_jobs = int(os.environ.get("SGE_TASK_LAST", 1)) 

    if 'SGE_TASK_ID' in os.environ:
        print('Running on SGE compute nodes.')

    msa = MSA(msa_file, msa_fmt)
    subsmat, tokenizer = read_subsmat(subsmat_type)
    msa.tokenize_seqs(tokenizer)

    pairs = distribute_pairs(msa.l, task_id, n_jobs)
    pair_batches = [pairs[i:i + int(batch_size), :] for i in range(0, len(pairs), int(batch_size))]

    if n_cpus > 1:

        if n_cpus > len(pair_batches):
            print('WARNING: The number of CPUs to distribute batches across is greater than the total number of batches. CPU count is being set to the total number of bacthes')
            n_cpus = len(pair_batches)

        pool = Pool(n_cpus, initargs=(RLock(), ), initializer=tqdm.set_lock)
        args = [(msa.tokenized_seqs[:, ~msa.uncovered_mask], batches, subsmat, verbose, i) for i, batches in enumerate(split_batches(pair_batches, n_cpus))]
        distances = list(pool.starmap(pwise_distance, args))
        distances = np.hstack(distances)

    else:
        distances = pwise_distance(msa.tokenized_seqs[:, ~msa.uncovered_mask], pair_batches, subsmat, verbose, 0)

    distance_list = np.hstack([pairs, distances[:, np.newaxis]])

    # divide by sequence length if doing pid calculation
    if calculate_pid:
        seqlens = msa.get_seqlens()
        min_seqlens = np.stack([seqlens[pairs[:, 0]], seqlens[pairs[:, 1]]]).T.min(axis=1)
        distance_list[:, 2] = distance_list[:, 2] / min_seqlens

    np.savez(os.path.join(outdir, f'pdist_output{task_id}.npz'), ids=msa.ids, distances=distance_list, distance_metric=np.array([subsmat_type]))
