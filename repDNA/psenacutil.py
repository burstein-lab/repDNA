__author__ = 'Fule Liu'

import sys
import os
import pickle
from math import pow

import numpy as np

from repDNA.util import frequency, batch_frequency
from repDNA.nacutil import make_kmer_list


ALPHABET = 'ACGT'


def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def get_phyche_factor_dic(k):
    """Get all {nucleotide: [(phyche, value), ...]} dict."""
    full_path = os.path.realpath(__file__)
    if 2 == k:
        file_path = "%s/data/mmc3.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic


def get_phyche_index(k, phyche_list):
    """get phyche_value according phyche_list."""
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_kmer_list(k, ALPHABET):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k)
    for nucleotide in make_kmer_list(k, ALPHABET):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value


def parallel_cor_function(nucleotide1, nucleotide2, phyche_index):
    """Get the cFactor.(Type1)"""
    temp_sum = 0.0
    phyche_index_values = list(phyche_index.values())
    len_phyche_index = len(phyche_index_values[0])
    for u in range(len_phyche_index):
        temp_sum += pow(float(phyche_index[nucleotide1][u]) - float(phyche_index[nucleotide2][u]), 2)

    return temp_sum / len_phyche_index


def series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value):
    """Get the series correlation Factor(Type 2)."""
    return float(phyche_value[nucleotide1][big_lamada]) * float(phyche_value[nucleotide2][big_lamada])


def get_parallel_factor(k, lamada, sequence, phyche_value):
    """Get the corresponding factor theta list."""
    theta = []
    l = len(sequence)

    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, l - k - i + 1):
            nucleotide1 = sequence[j: j+k]
            nucleotide2 = sequence[j+i: j+i+k]
            temp_sum += parallel_cor_function(nucleotide1, nucleotide2, phyche_value)

        theta.append(temp_sum / (l - k - i + 1))

    return theta


def get_series_factor(k, lamada, sequence, phyche_value):
    """Get the corresponding series factor theta list."""
    theta = []
    l_seq = len(sequence)
    temp_values = list(phyche_value.values())
    max_big_lamada = len(temp_values[0])

    for small_lamada in range(1, lamada + 1):
        for big_lamada in range(max_big_lamada):
            temp_sum = 0.0
            for i in range(0, l_seq - k - small_lamada + 1):
                nucleotide1 = sequence[i: i+k]
                nucleotide2 = sequence[i+small_lamada: i+small_lamada+k]
                temp_sum += series_cor_function(nucleotide1, nucleotide2, big_lamada, phyche_value)

            theta.append(temp_sum / (l_seq - k - small_lamada + 1))

    return theta


def make_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):
    """Generate the pseknc vector."""
    kmer = make_kmer_list(k, ALPHABET)
    vector = []

    for sequence in sequence_list:
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency(sequence, str(key)) for key in kmer]
        fre_sum = float(sum(fre_list))

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]

        # Get the theta_list according the Equation 5.
        if 1 == theta_type:
            theta_list = get_parallel_factor(k, lamada, sequence, phyche_value)
        elif 2 == theta_type:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation 9.
        denominator = 1 + w * theta_sum

        temp_vec = [round(f / denominator, 3) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 4))

        vector.append(temp_vec)

    return vector


def get_parallel_factor_psednc(lamada, sequence, phyche_matrix, dinuc_to_idx):
    l = len(sequence)

    if l < 2 or lamada <= 0:
        return [0.0] * lamada

    # Extract all dinucleotides and convert to indices
    dinucleotides = [sequence[j:j + 2] for j in range(l - 1)]
    dinuc_indices = np.array([dinuc_to_idx.get(dinuc, 0) for dinuc in dinucleotides])

    # Get property vectors for all dinucleotides
    all_properties = phyche_matrix[dinuc_indices]  # Shape: (l-1, n_properties)

    theta = []

    for i in range(1, lamada + 1):
        max_j = l - 1 - lamada
        if max_j <= 0:
            theta.append(0.0)
            continue

        # Vectorized calculation for this lag
        props1 = all_properties[:max_j]  # First set of properties
        props2 = all_properties[i:i + max_j]  # Second set (shifted by i)

        # Vectorized squared differences
        squared_diffs = (props1 - props2) ** 2
        correlations = np.mean(squared_diffs, axis=1)  # Mean over properties
        temp_sum = np.sum(correlations)
        theta.append(temp_sum / (l - i - 1))

    return theta


def make_old_pseknc_vector(sequence_list, lamada, w, k, phyche_value, theta_type=1):
    """Generate the pseknc vector."""
    kmer = make_kmer_list(k, ALPHABET)
    num_sequences = len(sequence_list)
    num_kmers = len(kmer)

    # Pre-allocate result array
    # Each vector has num_kmers + lamada elements
    vector_length = num_kmers + lamada
    result = np.zeros((num_sequences, vector_length), dtype=np.float64)

    # Convert phyche_value to numpy arrays once, which is faster to work with
    dinuc_list = sorted(phyche_value.keys())
    dinuc_to_idx = {dinuc: i for i, dinuc in enumerate(dinuc_list)}
    phyche_matrix = np.array([phyche_value[dinuc] for dinuc in dinuc_list])

    for i, sequence in enumerate(sequence_list):
        seq_len = len(sequence)
        if seq_len < k or lamada + k > seq_len:
            error_info = f"Sorry, the sequence length must be larger than {lamada + k}"
            sys.stderr.write(error_info)
            sys.exit(0)

        fre_list = batch_frequency(sequence, kmer)
        fre_array = np.array(fre_list, dtype=np.float64)

        # Normalize frequencies
        fre_sum = np.sum(fre_array)
        fre_array /= fre_sum

        # Get theta values
        if theta_type == 1:
            theta_list = get_parallel_factor_psednc(lamada, sequence, phyche_matrix, dinuc_to_idx)
        elif theta_type == 2:
            theta_list = get_series_factor(k, lamada, sequence, phyche_value)

        theta_array = np.array(theta_list, dtype=np.float64)
        theta_sum = np.sum(theta_array)

        # Vectorized final calculation
        denominator = 1 + w * theta_sum

        # Fill result array directly
        result[i, :num_kmers] = np.round(fre_array / denominator, 3)
        result[i, num_kmers:] = np.round(w * theta_array / denominator, 4)

    return result.tolist()


if __name__ == '__main__':
    # get_phyche_index(2, ['Base stacking'])
    extra_phyche_index = {'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11, 1],
                          'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04, 1],
                          'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62, 1],
                          'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17, 1],
                          'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25, 1],
                          'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24, 1],
                          'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39, 1],
                          'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62, 1],
                          'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71, 1],
                          'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59, 1],
                          'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24, 1],
                          'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04, 1],
                          'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39, 1],
                          'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71, 1],
                          'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25, 1],
                          'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11, 1]}
    phyche_index = extend_phyche_index(get_phyche_index(k=2, phyche_list=['Base stacking', 'DNA denaturation']),
                                       extra_phyche_index)
    print(phyche_index)