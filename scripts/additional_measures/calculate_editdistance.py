import thebeat
import os
import pandas as pd
import numpy as np
import warnings
import Levenshtein


warnings.filterwarnings("ignore")


ITIs = pd.read_csv(os.path.join("data", "experiment", "processed", "ITIs.csv"))
ITIs_bytrial = pd.read_csv(os.path.join("data", "experiment", "processed", "ITIs_bytrial.csv"))
ITIs['edit_distance'] = np.nan

fourier_df = pd.read_csv(os.path.join("data", "experiment", "processed", "fourier.csv"))


for sequence_id, sequence_df in ITIs.groupby("sequence_id"):
    resp_tempo = np.mean(sequence_df["resp_iti"])
    stim_seq = thebeat.Sequence(sequence_df["stim_ioi"].values)
    resp_seq = thebeat.Sequence(sequence_df["resp_iti"].values)

    # get the fourier value for determining tempo
    sixteenth_duration_stim = round(fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_stim'].values[0])
    sixteenth_duration_resp = round(fourier_df[fourier_df.sequence_id == sequence_id]['fourier_16th_duration_resp'].values[0])

    # quantize sequences
    stim_seq = stim_seq.quantize_iois(to=sixteenth_duration_stim)
    resp_seq = resp_seq.quantize_iois(to=sixteenth_duration_resp)

    # get binary sequences
    stim_seq_binary = thebeat.helpers.sequence_to_binary(stim_seq, sixteenth_duration_stim).astype(int)
    resp_seq_binary = thebeat.helpers.sequence_to_binary(resp_seq, sixteenth_duration_resp).astype(int)

    # get actual edit distance
    edit_distance = Levenshtein.distance(stim_seq_binary, resp_seq_binary)

    # get theoretically worst edit distance
    resp_seq_binary_inverse = 1 - resp_seq_binary
    worst_edit_distance = Levenshtein.distance(stim_seq_binary, resp_seq_binary_inverse)

    # normalize
    edit_distance_normalized = edit_distance / worst_edit_distance

    # add to dataframe
    ITIs.loc[ITIs["sequence_id"] == sequence_id, "edit_distance"] = edit_distance
    ITIs_bytrial.loc[ITIs_bytrial["sequence_id"] == sequence_id, "edit_distance"] = edit_distance
    ITIs.loc[ITIs["sequence_id"] == sequence_id, "edit_distance_normalized"] = edit_distance_normalized
    ITIs_bytrial.loc[ITIs_bytrial["sequence_id"] == sequence_id, "edit_distance_normalized"] = edit_distance_normalized


# change col data types
ITIs.edit_distance = ITIs.edit_distance.astype(int)
ITIs_bytrial.edit_distance = ITIs_bytrial.edit_distance.astype(int)

# sort by pp_id, then stim_id
ITIs = ITIs.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)
ITIs_bytrial = ITIs_bytrial.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)

print(ITIs)

# write out
ITIs.to_csv(os.path.join("data", "experiment", "processed", "ITIs.csv"), index = False)
ITIs_bytrial.to_csv(os.path.join("data", "experiment", "processed", "ITIs_bytrial.csv"), index = False)
