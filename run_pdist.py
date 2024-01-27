#!/usr/bin/env python3

from pdist import *
import numpy as np
import subprocess
import argparse
import os 


def init_parser():
    parser = argparse.ArgumentParser(description='Calculates pairwise distances between aligned sequences. Supports usage of BLOSUM62 similarity or percent identity as the distance metric. \nNOTE: pairwise percent identities are calculated using esl-alipid from hmmer-3.3.2. You can either install this yourself and append the path to the compiled binaries in your PATH variable in bashrc. Alternatively, you can append the path to the binaries I have compiled to your PATH variable.')
    parser.add_argument('outpath', type=str, help='Path to output directory.')
    parser.add_argument('msa_file', type=str, help='Path to multiple sequence alignment file.')
    parser.add_argument('--msa-fmt', type=str, default='stockholm', help='File format of the MSA. Can be any format compatible with SeqIO.read() from BioPython.')
    parser.add_argument('--distance', type=str, default='pid', help='Sequence metric to compute. Can be percent identity (pid) or BLOSUM62 similarity (BLOSUM62).')
    parser.add_argument('--batch-size', type=int, default=1000000, help='Batch size of pairs used during calculation of BLOSUM62 similarities. If you have a large number of very long sequences, you might consider decreasing this to avoid issues with memory.')
    return parser


if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()

    msa_file, msa_fmt = args.msa_file, args.msa_fmt
    outdir = args.outpath 
    batch_size = int(args.batch_size)
    distance = args.distance

    task_id = int(os.environ.get("SGE_TASK_ID", 1)) - 1
    n_jobs = int(os.environ.get("SGE_TASK_LAST", 1)) 

    if 'SGE_TASK_ID' in os.environ:
        print('Running on SGE compute nodes.')

    aligned_seqs, ids = load_ali(msa_file, msa_fmt)

    if distance == 'BLOSUM62':
        pairs = distribute_pairs(len(aligned_seqs), task_id, n_jobs)
        cleaned_aligned_seqs = clean_ali(aligned_seqs, remove_lowercase=True)
        distance_list = pwise_blosum(cleaned_aligned_seqs, pairs, batch_size)
        np.savez(os.path.join(outdir, f'{BLOSUM_OUTPUT_NAME}{task_id}.npz'), ids=np.array(ids), distance_list=distance_list)

    else:

        print('Running esl-alipid')
        esl_command = ' '.join([
            'esl-alipid',
            '--informat',
            msa_fmt,
            msa_file
            ])
        esl_output = subprocess.run(esl_command, shell=True, capture_output=True, text=True)

        print('Reformatting output from esl-alipid into a matrix.')
        pid_matrix = reformat_esl_output(esl_output, ids)
        np.savez(os.path.join(outdir, f'{PID_OUTPUT_NAME}{task_id}.npz'), ids=np.array(ids), pid=pid_matrix)