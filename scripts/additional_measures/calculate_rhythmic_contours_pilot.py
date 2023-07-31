import Levenshtein
import pandas as pd
import os

def rhythmic_contours(iois):
    output = ''
    for i in range(len(iois) - 1):
        if iois[i + 1] > iois[i]:
            output += 'A'
        elif iois[i + 1] < iois[i]:
            output += 'B'
        else:
            output += 'C'
    return output

def rhythmic_contours_levenshtein(stim_sequence, resp_sequence):
    stim_str = rhythmic_contours(stim_sequence)
    resp_str = rhythmic_contours(resp_sequence)
    return Levenshtein.distance(stim_str, resp_str)


ITIs = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'))


for sequence_id, stim_df in ITIs.groupby('sequence_id'):
    stim_iois = stim_df.stim_ioi.values
    resp_itis = stim_df.resp_iti.values
    rhythmic_contours_distance = rhythmic_contours_levenshtein(stim_iois, resp_itis)

    ITIs.loc[ITIs.sequence_id == sequence_id, 'rhythmic_contours_edit_distance'] = rhythmic_contours_distance
    ITIs_bytrial.loc[ITIs_bytrial.sequence_id == sequence_id, 'rhythmic_contours_edit_distance'] = rhythmic_contours_distance


# change data type
ITIs.rhythmic_contours_edit_distance = ITIs.rhythmic_contours_edit_distance.astype(int)
ITIs_bytrial.rhythmic_contours_edit_distance = ITIs_bytrial.rhythmic_contours_edit_distance.astype(int)

# write out
ITIs.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs.csv'), index = False)
ITIs_bytrial.to_csv(os.path.join('data', 'pilot', 'processed', 'ITIs_bytrial.csv'), index = False)
