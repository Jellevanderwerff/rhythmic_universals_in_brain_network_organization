import thebeat
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


ITIs = pd.read_csv(os.path.join("data", "experiment", "processed", "ITIs.csv"))
ITIs_bytrial = pd.read_csv(os.path.join("data", "experiment", "processed", "ITIs_bytrial.csv"))
ITIs['edit_distance'] = np.nan


for sequence_id, sequence_df in ITIs.groupby("sequence_id"):
    resp_tempo = np.mean(sequence_df["resp_iti"])
    stim_seq = thebeat.Sequence(sequence_df["stim_ioi"].values)
    resp_seq = thebeat.Sequence(sequence_df["resp_iti"].values)

    # calculate edit distance
    edit_distance = thebeat.stats.edit_distance_sequence(resp_seq, stim_seq, resolution = resp_tempo / 4)
    
    # add to dataframe
    ITIs.loc[ITIs["sequence_id"] == sequence_id, "edit_distance"] = edit_distance
    ITIs_bytrial.loc[ITIs_bytrial["sequence_id"] == sequence_id, "edit_distance"] = edit_distance


# change col data types 
ITIs.edit_distance = ITIs.edit_distance.astype(int)
ITIs_bytrial.edit_distance = ITIs_bytrial.edit_distance.astype(int)

# sort by pp_id, then stim_id
ITIs = ITIs.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)
ITIs_bytrial = ITIs_bytrial.sort_values(by = ["pp_id", "stim_id"]).reset_index(drop = True)

# write out
ITIs.to_csv(os.path.join("data", "experiment", "processed", "ITIs.csv"), index = False)
ITIs_bytrial.to_csv(os.path.join("data", "experiment", "processed", "ITIs_bytrial.csv"), index = False)
