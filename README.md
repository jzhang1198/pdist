# pdist

A collection of Python scripts for scalable calculation of pairwise distances between aligned protein sequences. `pdist` supports percent identity and BLOSUM62 similarity as distance metrics.  

## Quickstart

Ensure that you have the dependencies within `requirements.txt` installed. If you are parallelizing your pairwise calculations across multiple cores on Wynton (or any other SGE supercomputing cluster), you will need to also install clone [`PyClust`](https://github.com/jzhang1198/PyClust) and install it to your environment by running `pip install -e .` at the root of the repo.

You will need to also install `esl-alipid`, which is included as a mini-app within the [HMMER](http://hmmer.org) software. After installation, ensure that you have appended the path to the directory containing `esl-alipid` to your path variable in `.bashrc`. Alternatively, if you are running this on Wynton, you can just append this path to your path variable:

`/wynton/home/kortemme/jzhang1198/code/hmmer-3.3.2/easel/miniapps`

## Running on a single core

`run_pdist.py` will scale well up to 10K aligned sequences. 

## Running on a cluster

`run_pdist_distributed.py` is a wrapper over `run_pdist.py` for submission of array jobs on a SGE supercomputing cluster. Note that the pairwise sequence identity measurements cannot be parallelized across multiple cores! I have tried implementing my own pairwise sequence identity calculator, but I have found it to be much less efficient than the implementation in `esl-alipid`. 