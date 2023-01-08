import argparse
from itertools import groupby
from hmmlearn import hmm
import gzip as gz
import pandas as pd
import numpy as np

BASES_INDICES = {"A": 0, "C": 1, "G": 2, "T": 3}
TRAIN_DATA_FILES = ["train_background.fa.gz", "train_cpg_island.fa.gz"]

def get_initial_transitions():
    initial_transitions = pd.DataFrame(index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"],
                                       columns=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"], dtype=float)
    p_stay_in_island = 0.995
    p_stay_outside_island = 0.9999

    transition_within_island = np.array([
        [0.18, 0.27, 0.43, 0.12],
        [0.17, 0.37, 0.27, 0.19],
        [0.16, 0.34, 0.38, 0.12],
        [0.08, 0.36, 0.38, 0.18],
    ])

    transition_outside_island = np.array([
        [0.3, 0.2, 0.29, 0.21],
        [0.32, 0.3, 0.08, 0.3],
        [0.25, 0.25, 0.3, 0.2],
        [0.18, 0.24, 0.29, 0.29]
    ])

    initial_transitions.loc[["A+", "C+", "G+", "T+"], ["A+", "C+", "G+", "T+"]] = \
        transition_within_island * p_stay_in_island
    initial_transitions.loc[["A-", "C-", "G-", "T-"], ["A-", "C-", "G-", "T-"]] = \
        transition_outside_island * p_stay_outside_island

    initial_transitions.loc[["A+", "C+", "G+", "T+"], ["A-", "C-", "G-", "T-"]] = \
        (1 - p_stay_in_island) / 4
    initial_transitions.loc[["A-", "C-", "G-", "T-"], ["A+", "C+", "G+", "T+"]] = \
        (1 - p_stay_outside_island) / 4

    return initial_transitions


def get_initial_startprob():
    human_genome_length = 3.8e9
    human_cpg_island_count = 45000
    average_island_length = 200

    p_is_island = (human_cpg_island_count * average_island_length) / human_genome_length

    base_probs = np.array([0.15, 0.35, 0.35, 0.15, 0.25, 0.25, 0.25, 0.25])
    mask = np.array(([p_is_island] * 4) + ([1 - p_is_island] * 4))

    return pd.DataFrame({"startprob": base_probs * mask},
                        index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"])


def get_initial_data():
    initial_emission = pd.DataFrame({"A": [1, 0, 0, 0, 1, 0, 0, 0], "C": [0, 1, 0, 0, 0, 1, 0, 0],
                                     "G": [0, 0, 1, 0, 0, 0, 1, 0], "T": [0, 0, 0, 1, 0, 0, 0, 1]},
                                    index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"])

    return initial_emission, get_initial_transitions(), get_initial_startprob()


def preprocess_data(fasta: str) -> np.ndarray:
    reader = fastaread_gz if is_gz_file(fasta) else fastaread
    result = np.array([[BASES_INDICES[c]] for _, seq in reader(fasta) for c in seq])
    lengths = np.array([len(seq) for _, seq in reader(fasta)])
    return result, lengths


def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def fastaread_gz(fasta_name):
    """
    Read a gzip compressed fasta file. For each sequence in the file, yield the header and the actual sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = gz.open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.decode().startswith(">")))
    for header in faiter:
        header = next(header)[1:].decode().strip()
        seq = "".join(s.decode().strip() for s in next(faiter))
        yield header, seq


def train(model):
    seqs_0, lengths_0 = preprocess_data(TRAIN_DATA_FILES[0])
    seqs_1, lengths_1 = preprocess_data(TRAIN_DATA_FILES[1])
    train_seqs = np.concatenate([seqs_0, seqs_1])
    train_lengths = np.concatenate([lengths_0, lengths_1])

    model.fit(train_seqs, train_lengths)

    return model


def predict_fasta(model, fasta):
    states = []
    likelihoods = []
    lengths = []

    reader = fastaread_gz if is_gz_file(fasta) else fastaread
    for _, seq in reader(fasta):
        parsed_seq = np.array([[BASES_INDICES[c]] for c in seq])
        seq_len = np.array([len(seq)])
        ll, hidden_states = model.decode(parsed_seq, seq_len)
        hidden_states = np.where(hidden_states >= 4, 'N', 'I')
        states.append(hidden_states)
        likelihoods.append(ll)
        lengths.append(len(seq))

    states = [''.join(s.astype(str)) for s in states]

    return states, likelihoods, lengths

def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    parser.add_argument('decodeAlg',
                        help='The algorithm name for predicting/decoding hidden states. Can be "posterior" or "viterbi".')
    parser.add_argument('outputPrefix', help='The prefix of output.')
    return parser.parse_args()


def display_model_parameters(model: hmm.CategoricalHMM, label: str):
    transitions = pd.DataFrame(index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"],
                                       columns=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"],
                                       data=model.transmat_)
    emission = pd.DataFrame(index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"],
                                       columns=["A", "C", "G", "T"],
                                       data=model.emissionprob_)
    startprob = pd.DataFrame(index=["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"],
                                       columns=["p"],
                                       data=model.startprob_)
    
    for mat, mat_name in zip((transitions, emission, startprob),("transition", "emission", "startprob")):
        mat.to_html(f"{label}_{mat_name}.html")


def init_model(algorithm, convergence_threshold):
    model = hmm.CategoricalHMM(n_components=8, algorithm=algorithm, n_iter=120,
                               tol=convergence_threshold, params="st",  init_params="")
    emission, transition, startprob = get_initial_data()

    model.emissionprob_ = emission
    model.startprob_ = np.squeeze(startprob.values)
    model.transmat_ = transition

    return model


def evaluate_model(model):
    test_seqs = ["30_cpg_island.padded.fa.gz", "70_non_cpg_island.padded.fa.gz"]
    test_labels = ["30_cpg_island.label.fa.gz", "70_non_cpg_island.label.fa.gz"]

    # Question 2 - calculate average likelihood per base
    for ts in test_seqs:
        _, likelihoods, lengths = predict_fasta(model, ts)
        likelihoods = np.array(likelihoods)
        lengths = np.array(lengths)
        likelihood_per_base = np.exp(likelihoods / lengths)
        print(f"Average likelihood per base for {ts}: {np.mean(likelihood_per_base)}")

    # Question 3 - evaluate model preformance


def main(visualize: bool):
    args = parse_args()
    prediction_output_file = args.outputPrefix + "cpg_island_predictions.txt"
    likelihood_output_file = args.outputPrefix + "likelihood.txt"
    learned_params_output_file = args.outputPrefix + "params.txt"

    model = init_model(args.decodeAlg, args.convergenceThr)

    if visualize:
        display_model_parameters(model, "Before")
        for tdf in TRAIN_DATA_FILES:
            _, likelihoods_before, _ = predict_fasta(model, tdf)
            print(f"Average log-likelihood before training for {tdf}: {np.mean(likelihoods_before)}")
    
    model = train(model)

    if visualize:
        display_model_parameters(model, "After")
        for tdf in TRAIN_DATA_FILES:
            _, likelihoods_after, _ = predict_fasta(model, tdf)
            print(f"Average log-likelihood after training for {tdf}: {np.mean(likelihoods_after)}")
        evaluate_model(model)
    
    states, likelihoods, _ = predict_fasta(model, args.fasta)

    with open(prediction_output_file, 'w') as outfile:
        for s in states:
            outfile.write(f"{s}\n")

    with open(likelihood_output_file, 'w') as outfile:
        for ll in likelihoods:
            outfile.write(f"{ll}\n")

    with open(learned_params_output_file, 'w') as outfile:
        for row in model.transmat_:
            row_txt = '\t'.join(str(c) for c in row)
            outfile.write(f"{row_txt}\n")


if __name__ == '__main__':
    main(True)
