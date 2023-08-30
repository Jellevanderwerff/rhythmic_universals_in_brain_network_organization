import Levenshtein
import pandas as pd
import os
import numpy as np
import scipy.stats

def rhythmic_contours(iois):
    output = ''
    for i in range(len(iois) - 1):
        if iois[i + 1] > iois[i]:
            output += 'L'
        elif iois[i + 1] < iois[i]:
            output += 'S'
        else:
            output += 'E'
    return output

def rhythmic_contours_levenshtein(stim_sequence, resp_sequence):
    stim_str = rhythmic_contours(stim_sequence)
    resp_str = rhythmic_contours(resp_sequence)

    return Levenshtein.distance(stim_str, resp_str)


ITIs_quantized = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_quantized.csv'))
ITIs = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'))
ITIs_bytrial = pd.read_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'))

count = 0

for sequence_id, seq_df in ITIs_quantized.groupby('sequence_id'):
    print(count)
    stim_iois = list(seq_df.stim_ioi_q.values)
    resp_itis = list(seq_df.resp_iti_q.values)
    rhythmic_contours_distance = rhythmic_contours_levenshtein(stim_iois, resp_itis)

    # Add rhythmic contour edit distance
    ITIs_quantized.loc[ITIs_quantized.sequence_id == sequence_id, 'rhythmic_contours_edit_distance_quantized'] = rhythmic_contours_distance
    ITIs.loc[ITIs.sequence_id == sequence_id, 'rhythmic_contours_edit_distance_quantized'] = rhythmic_contours_distance
    ITIs_bytrial.loc[ITIs_bytrial.sequence_id == sequence_id, 'rhythmic_contours_edit_distance_quantized'] = rhythmic_contours_distance

    count += 1


# change data type
ITIs_quantized.rhythmic_contours_edit_distance_quantized = ITIs_quantized.rhythmic_contours_edit_distance_quantized.astype(int)
ITIs_bytrial.rhythmic_contours_edit_distance_quantized = ITIs_bytrial.rhythmic_contours_edit_distance_quantized.astype(int)
ITIs.rhythmic_contours_edit_distance_quantized = ITIs.rhythmic_contours_edit_distance_quantized.astype(int)

# write out
ITIs_quantized.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index = False)
ITIs_bytrial.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs_bytrial.csv'), index = False)
ITIs.to_csv(os.path.join('data', 'experiment', 'processed', 'ITIs.csv'), index = False)
