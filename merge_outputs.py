#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np 
from pdist import *
from tqdm import tqdm

def init_parser():
    parser = argparse.ArgumentParser(description='Merges outputs from an array job. Can also convert between output formats. Note that this will only work if all the output files are not renamed!')
    parser.add_argument('--dir', default='.', help='Path to directory containing output files.')
    parser.add_argument('--convert-to-matrix', action='store_true', help='Converts the list representation to a distance matrix.')
    parser.add_argument('--remove-raw-outputs', action='store_true', help='Determines whether original files are kept or not.')
    return parser

if __name__ == '__main__':
    
    parser = init_parser()
    args = parser.parse_args()

    dir = args.dir 
    convert_to_matrix = False if not args.convert_to_matrix else True
    remove_raw_outputs = False if not args.remove_raw_outputs else True

    files = [os.path.join(dir, f) for f in os.listdir(dir) if 'pdist_output' in f]
    files = [f for f in files if re.search(r'\d+', f)]
    distance_list, distance_metrics = [], {}
    for file in tqdm(files, desc='Reading in output files.', unit='file'):
        distance_metrics.add(data['distance_metric'][0])
        data = np.load(file)
        distance_list.append(data['distance_list'])
    distance_list = np.vstack(distance_list)

    ids = data['ids']
    if len(distance_metrics) > 1:
        raise Exception('ERROR: output files must have a consistent distance metric! The following distance metrics were detected: {}'.format(', '.join(list(distance_metrics))))

    if convert_to_matrix:
        distance_matrix = np.zeros((len(ids), len(ids)))
        distance_matrix[distance_list[:, 0].astype(int), distance_list[:, 1].astype(int)] = distance_list[:, 2]
        similarity_matrix = distance_matrix + distance_matrix.T - np.diag(np.diag(distance_matrix))
        np.savez(os.path.join(dir, f'pdist_output.npz'), ids=ids, distance_matrix=similarity_matrix, distance_metric=list(distance_metrics)[0])

    else:
        np.savez(os.path.join(dir, f'pdist_output.npz'), ids=ids, distance_list=distance_list, distance_metric=list(distance_metrics)[0])

    if remove_raw_outputs:
        pass
