__author__ = 'Fule Liu'

ALPHABET = 'ACGT'


import numpy as np


def extend_phyche_index(original_index, extend_index):
    """Extend {phyche:[value, ... ]}"""
    if 0 == len(extend_index):
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def make_ac_vector(sequence_list, lag, phyche_value, k):
    """
    Optimized version that processes all sequences and lags simultaneously.
    This version provides maximum performance for large datasets.
    """
    # Convert phyche_value dict to numpy array
    kmer_keys = list(phyche_value.keys())
    phyche_array = np.array([phyche_value[kmer] for kmer in kmer_keys], dtype=float)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmer_keys)}

    vec_ac = []

    for sequence in sequence_list:
        len_seq = len(sequence)

        # Pre-extract all k-mers and convert to indices
        kmers = [sequence[i:i + k] for i in range(len_seq - k + 1)]
        kmer_indices = np.array([kmer_to_idx[kmer] for kmer in kmers], dtype=int)
        seq_phyche_values = phyche_array[kmer_indices]

        each_vec = []

        for temp_lag in range(1, lag + 1):
            max_i = len_seq - temp_lag - k + 1

            # Get phyche values for this lag
            phyche_vals_1 = seq_phyche_values[:max_i]
            phyche_vals_2 = seq_phyche_values[temp_lag:temp_lag + max_i]

            # Vectorized calculation for all j values at once
            ave_phyche_values = np.sum(phyche_vals_1, axis=0) / len_seq
            diff_vals = phyche_vals_1 - ave_phyche_values
            temp_sums = np.sum(diff_vals * phyche_vals_2, axis=0)

            # Add all j values for this lag
            each_vec.extend(np.round(temp_sums / max_i, 3).tolist())

        vec_ac.append(each_vec)

    return vec_ac


def make_cc_vector(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_cc = []
    for sequence in sequence_list:
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for i1 in range(len_phyche_value):
                for i2 in range(len_phyche_value):
                    if i1 != i2:
                        # Calculate average phyche_value for a nucleotide.
                        ave_phyche_value1 = 0.0
                        ave_phyche_value2 = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide = sequence[j: j + k]
                            ave_phyche_value1 += float(phyche_value[nucleotide][i1])
                            ave_phyche_value2 += float(phyche_value[nucleotide][i2])
                        ave_phyche_value1 /= len_seq
                        ave_phyche_value2 /= len_seq

                        # Calculate the vector.
                        temp_sum = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide1 = sequence[j: j + k]
                            nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
                            temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
                                        (float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
                        each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 3))

        vec_cc.append(each_vec)

    return vec_cc


if __name__ == '__main__':
    pass