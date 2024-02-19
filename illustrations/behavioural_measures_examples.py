import pandas as pd
import numpy as np
from thebeat.music import Rhythm
from pprint import pprint
import os

# Make a list of dictionaries with all combinations of the variables, which can have the value low or high
combinations = []
for entropy in ['negative', 'zero', 'positive']:
    for redundancy in ['zero', 'high']:
        for isochrony in ['negative', 'zero', 'high']:
            for binary_ternary in ['negative', 'zero', 'high']:
                combinations.append({
                    'Entropy difference': entropy,
                    'Grammatical redundancy': redundancy,
                    'Isochrony introduced': isochrony,
                    'Binary and ternary ratios introduced': binary_ternary,
                })

# Make a dataframe with the combinations
df = pd.DataFrame(combinations).transpose()
df.to_csv(os.path.join('illustrations', 'behavioural_measures_combinations.csv'))



"""
Entropy difference: negative
Grammatical redundancy: zero
Isochrony introduced: negative
Binary and ternary ratios introduced: negative
"""
stim = Rhythm([])