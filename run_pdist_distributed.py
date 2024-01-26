#!/usr/bin/env python3

from PyClust import SGEJob
import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath', type=str, help='Path to output directory.')
    parser.add_argument('msa_file', type=str, help='Path to multiple sequence alignment file.')
    parser.add_argument('conda_env_name', type=str, help='Name of conda environment to run program in.')
    parser.add_argument('--msa-fmt', type=str, default='stockholm', help='File format of the MSA. Can be any format compatible with SeqIO.read() from BioPython.')
    parser.add_argument('--distance', type=str, default='pid', help='Sequence metric to compute. Can be percent identity (pid) or BLOSUM62 similarity (BLOSUM62).')
    parser.add_argument('--batch-size', type=int, default=1000000, help='Batch size of pairs used during calculation of BLOSUM62 similarities. If you have a large number of very long sequences, you might consider decreasing this to avoid issues with memory.')
    parser.add_argument('--job-name', type=str, default='pdist', help='Name of job.')
    parser.add_argument('--n', type=int, default=1, help='Number of tasks for array job.')
    parser.add_argument('--time-allocation', type=str, default='02:00:00', help='Time allocation for job, in HH:MM:SS format.')
    parser.add_argument('--memory-allocation', type=int, default=2, help='Memory allocation for the job (in GigaBytes). Think this only works with integers.')
    return parser


if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()

    outpath = args.outpath
    msa_file, msa_fmt = args.msa_file, args.msa_fmt
    conda_env_name = args.conda_env_name
    distance = args.distance
    batch_size = args.batch_size
    job_name = args.job_name
    n_jobs = args.n
    time_allocation = args.time_allocation
    memory_allocation = args.memory_allocation

    if distance == 'pid' and n_jobs > 1:
        print('WARNING: distributed calculation of percent identities is not supported. The number of array jobs has been set to 1.')
        n_jobs = 1

    job = SGEJob(
        outpath, 
        conda_env_name,
        job_name=job_name,
        n_tasks=n_jobs,
        time_allocation=time_allocation,
        memory_allocation=memory_allocation
    )
    job.submit(script='./run_pdist.py', arguments=[
        outpath,
        msa_file,
        '--msa-fmt',
        msa_fmt,
        '--distance',
        distance, 
        '--batch-size',
        batch_size
    ])
    
