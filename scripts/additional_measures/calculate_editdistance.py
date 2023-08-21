import thebeat
import os
import pandas as pd
import numpy as np
import warnings


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

    # quantize resp_seq to stim_seq
    relationship_between_stim_and_resp = sixteenth_duration_stim / sixteenth_duration_resp
    resp_seq.iois *= relationship_between_stim_and_resp
    stim_seq.iois = np.round(stim_seq.iois, 0)
    resp_seq.iois = np.round(resp_seq.iois, 0)

    # calculate edit distance
    edit_distance = thebeat.stats.edit_distance_sequence(stim_seq, resp_seq, resolution=sixteenth_duration_stim)

    # add to dataframe
    ITIs.loc[ITIs["sequence_id"] == sequence_id, "edit_distance"] = edit_distance
    ITIs_bytrial.loc[ITIs_bytrial["sequence_id"] == sequence_id, "edit_distance"] = edit_distance


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
