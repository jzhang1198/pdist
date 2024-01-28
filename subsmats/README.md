# Adding your own substitution matrices

You can create your own substitution matrices by following the instructions below:

1. Create a numpy array containing an amino acid alphabet. This alphabet should cover all characters present in your input sequence alignments (i.e. standard amino acids, gap characters, etc.).
2. Create a symmetric `NxN` numpy array, where `N` is the length of the alphabet. The rows and columns should be indexed by the alphabet array created above.
3. Populate the NxN array such that the element at row index `i` and column index `j` represents the similarity of the amino acids at index `i` and `j` of the alphabet array. The substitution matrix should be symmetric or you may get weird results!
4. Save the alphabet array and the substitution matrix as a `.npz` archive file. For example:

```python
np.savez('./subsmats/new_subsmat.npz', subsmat=subsmat, alphabet=alphabet)
```

5. You can now specify your custom substitution matrix using the `--subsmat` flag in `run_pdist.py`. For example:

```bash
./run_pdist.py . msa_file --subsmat new_subsmat
```