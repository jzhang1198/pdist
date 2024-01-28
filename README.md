# pdist

A collection of Python scripts for scalable calculation of pairwise distances between aligned protein sequences. 

## Quickstart

Ensure that you have the dependencies within `requirements.txt` installed. If you are parallelizing your pairwise calculations across multiple cores on Wynton (or any other SGE supercomputing cluster), you will need to also clone [`PyClust`](https://github.com/jzhang1198/PyClust) and install it to your environment by running `pip install -e .` at the root of the repo.

## Types of distance calculations
`pdist` supports calculation of similarities and percent identities as sequence distance metrics. 

There are many ways to calculate percent identities. Here, I follow the convention used in the `esl-alipid` miniapp from HMMER. Briefly, pairwise percent identity is calculated by counting the number of identical residues in consensus columns of the alignment and then dividing this quantity by `min(len1, len2)`. The sequence lengths are calculated by counting all non-gap positions in the alignment.

For pairwise similarity calculations, I only consider the columns covered in the alignment. One can provide their own custom substitution matrix by following the instructions included in the `subsmats` folder in this repo.

## Running on a single device

`run_pdist.py` is designed to run on a single device. For larger alignments, pairwise distance calculations can be parallelized across different CPUs on the same device. Run the script with the `-h` flag for detailed usage instructions.

## Running on a cluster

`run_pdist_distributed.py` is a wrapper over `run_pdist.py` for submission of array jobs on a SGE supercomputing cluster. Run the script with the `-h` flag for detailed usage instructions.