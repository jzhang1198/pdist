#!/usr/bin/env python3

import os
import re
import sys
import argparse
import numpy as np 
from pdist import *
from tqdm import tqdm

def init_parser():
    parser = argparse.ArgumentParser(description='Merges outputs from an array job. Can also convert between output formats.')
    parser.add_argument('output_type', help='Designates whether the output is from sequence identity (pid) or similarity calculations (BLOSUM62).')
    parser.add_argument('--dir', default='.', help='Path to directory containing output files.')
    parser.add_argument('--convert-to-matrix', action='store_true', help='Converts the list representation to a distance matrix.')
    parser.add_argument('--remove-raw-outputs', action='store_true', help='Determines whether original files are kept or not.')
    return parser

if __name__ == '__main__':
    
    parser = init_parser()
    args = parser.parse_args()

    output_name = PID_OUTPUT_NAME if args.output_type == 'pid' else BLOSUM_OUTPUT_NAME
    dir = args.dir 
    convert_to_matrix = False if not args.convert_to_matrix else True
    remove_raw_outputs = False if not args.remove_raw_outputs else True

    if output_name == PID_OUTPUT_NAME:
        print('No outputs to merge for percent identity calculations! Exiting the program.')
        sys.exit()

    files = [os.path.join(dir, f) for f in os.listdir(dir) if output_name in f]
    files = [f for f in files if re.search(r'\d+', f)]
    distance_list = []
    for file in tqdm(files, desc='Reading in output files.', unit='file'):
        data = np.load(file)
        distance_list.append(data['distance_list'])
    distance_list = np.vstack(distance_list)

    ids = data['ids']

    if convert_to_matrix:
        similarity_matrix = np.zeros((len(ids), len(ids)))
        similarity_matrix[distance_list[:, 0].astype(int), distance_list[:, 1].astype(int)] = distance_list[:, 2]
        similarity_matrix = similarity_matrix + similarity_matrix.T - np.diag(np.diag(similarity_matrix))
        np.savez(os.path.join(dir, f'{BLOSUM_OUTPUT_NAME}.npz'), ids=ids, similarity_matrix=similarity_matrix)

    else:
        np.savez(os.path.join(dir, f'{BLOSUM_OUTPUT_NAME}.npz'), ids=ids, distance_list=distance_list)
