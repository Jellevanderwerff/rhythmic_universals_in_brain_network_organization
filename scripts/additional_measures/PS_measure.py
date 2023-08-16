import pandas as pd
import thebeat
import numpy.typing as npt
import re

def get_binary_sequence(ratios: npt.ArrayLike):
    output = []

    for ratio in ratios:
        output += [1]
        output += [0] * (ratio - 1)


    return output

def add_accents(binary_sequence: str):
    """
    See Povel & Essens (1985). 0 means silence, 1 means unaccented event, 2 means accented event.
    """
    output = []
    count = 0

    while count < len(binary_sequence):
        # If there's three consecutive 1s:
        if binary_sequence[count:count+3] == '111':
            count += 3
            output += '212'
        # If there's two consecutive 1s:
        elif binary_sequence[count:count+2] == '11':
            count += 2
            output += '12'
        # If an event is isolated
        elif binary_sequence[count:count+3] == '010':
            count += 3
            output += '020'
        # Otherwise, add the event to the output, and move on
        else:
            output += binary_sequence[count]
            count += 1

    return ''.join(output)



def calculate_PS(sequence, grid_ioi):
    """
    Calculates the PS measure for a given Sequence object.

    Assumptions:

    - We assume that the sequence starts at t=0, so there can be no silence at the beginning
    - We assume that the provided Sequence constitutes one 'period'.

    Parameters
    ----------

    sequence : thebeat.core.Sequence
        The sequence for which to calculate the PS-measure.
    grid_ioi : float
        This number indicates the ioi of the underlying temporal grid.

    Returns
    -------

    ps : dict
        A dictionary containing the PS measure, the corresponding unit size and location, and the corresponding C, D, and m values
        used during calculation.
    """

    # Definitions
    WEIGHTS = {
        'lambda': 0.2223,
        'W': 1.1695,
        'd1': 0.0235,
        'd2': 1.2722,
        'd3': 1.2955,
        'd4': 0.0736,
        'd5': 0.7931
    }

    # Checks
    if sequence.duration % grid_ioi != 0:
        raise ValueError('Sequence duration must be a multiple of the period IOI divided by n_period_subdivisions. Probably, you need to quantize the sequence \
                         first, e.g. using Sequence.quantize().')

    if sequence.onsets[0] != 0:
        raise ValueError('Sequence must start at time 0.')

    # Get the binary sequence
    binary_sequence = ''.join([str(x) for x in get_binary_sequence(sequence.integer_ratios)])
    accented_sequence = add_accents(binary_sequence)

    currently_lowest_C = None
    corresponding_u = None
    corresponding_loc = None

    min_unit_size = 1
    max_unit_size = int((sequence.duration / grid_ioi / 2) - 1)  # because everything _smaller_ than 1/2 period

    # Determine the best clock
    for unit_size in range(min_unit_size, max_unit_size + 1):
        for loc in range(1, unit_size + 1):
            if len(accented_sequence) % unit_size != 0:  # If it is not a divisor
                break

            # Make a sequence of 0's and 1' with the length of accented_sequence,
            # placing ones at multiples of unit_size, with the first 1 at loc
            clock = ['1' if i % unit_size == loc - 1 else '0' for i in range(len(accented_sequence))]

            # Check that the clock and the accented sequence are the same length
            assert len(clock) == len(accented_sequence)

            # u is the number of clock ticks coinciding with unaccented events
            u = 0
            # s is the number of clock ticks coinciding with silence
            s = 0

            # Loop over the clock ticks and the accented sequence
            for clock_tick, accent in zip(clock, accented_sequence):
                if clock_tick == '1' and accent == '0':  # clock tick + silence
                    s += 1
                elif clock_tick == '1' and accent == '1':  # clock tick + unaccented event
                    u += 1

            # Calculate C
            C = WEIGHTS['W'] * s + u

            # Replace resulting C if it is lower than the current lowest C, and save unit size and location
            if currently_lowest_C == None or C < currently_lowest_C:
                currently_lowest_C = C
                corresponding_u = unit_size
                corresponding_loc = loc

    # Save final values for clarity
    final_C = currently_lowest_C
    final_u = corresponding_u
    final_loc = corresponding_loc

    ## Calculate D
    # Split up the original sequence into final_u-sized segments
    segments = [binary_sequence[i:i + final_u] for i in range(0, len(binary_sequence), final_u)]

    # calculate the sum of the c_i's
    ci_sum = 0
    for segment in segments:
        # segment starts with silence (type N)
        if segment.startswith('0'):
            ci_sum += WEIGHTS['d4']
        # segment is empty (type E)
        elif re.search(r"^10*1$", segment):
            ci_sum += WEIGHTS['d1']
        else:  # segment is either equally or unequally subdivided
            segment_durations = [len(x) for x in segment.split('0') if x != '']
            if len(segment) % 2 == 0:  # if even
                if all(x == segment_durations[0] for x in segment_durations):  # equally subdivided if all values are the same
                    ci_sum += WEIGHTS['d2']
                else:  # segment is unequally subdivided
                    ci_sum += WEIGHTS['d3']
            else:  # if uneven, cannot be subdivided equally
                ci_sum += WEIGHTS['d3']


    # calculate m (i.e. number of 'new' strings)
    # for each segment check whether the subsequent segment is the same:
    m = 0
    for i in range(len(segments) - 1):
        if segments[i] != segments[i + 1]:
            m += 1

    # Calculate D
    D = ci_sum + m * WEIGHTS['d5']

    # Calculate PS
    PS = WEIGHTS['lambda'] * final_C + (1 - WEIGHTS['lambda']) * D

    return {
        'PS': PS,
        'C': final_C,
        'unit': final_u,
        'loc': final_loc,
        'D': D,
        'm': m
    }


seq = thebeat.Sequence.from_integer_ratios([2, 2, 4, 3, 1, 4], value_of_one=150)
print(seq.duration)
print(calculate_PS(seq, 150))







# tests:
"""
patterns = [
    [1, 1, 1, 1, 3, 1, 2, 2, 4],
    [1, 1, 2, 2, 1, 1, 3, 1, 4],
    [2, 1, 1, 2, 1, 1, 3, 1, 4],
    [2, 2, 1, 1, 1, 1, 3, 1, 4],
    [3, 1, 2, 2, 1, 1, 1, 1, 4],  # 5
    [1, 1, 2, 1, 1, 2, 1, 3, 4],
    [2, 1, 1, 1, 2, 1, 3, 1, 4],
    [1, 3, 1, 1, 1, 1, 2, 2, 4],
    [1, 3, 2, 1, 1, 2, 1, 1, 4],
    [2, 1, 1, 2, 1, 1, 1, 3, 4],  # 10
    [1, 1, 2, 1, 3, 1, 2, 1, 4],
    [1, 2, 1, 1, 1, 2, 3, 1, 4],
    [1, 2, 1, 2, 1, 1, 1, 3, 4],
    [1, 3, 1, 2, 1, 2, 1, 1, 4],
    [3, 1, 1, 2, 1, 1, 2, 1, 4],  # 15
    [1, 2, 1, 1, 1, 2, 1, 3, 4],
    [1, 2, 1, 1, 2, 1, 1, 3, 4],
    [1, 2, 1, 1, 3, 1, 2, 1, 4],
    [1, 3, 1, 2, 1, 1, 1, 2, 4],
    [1, 3, 1, 2, 1, 1, 2, 1, 4],  # 20
    [1, 1, 1, 1, 2, 1, 2, 3, 4],
    [1, 1, 1, 1, 2, 1, 2, 3, 4],
    [1, 1, 3, 1, 2, 1, 1, 2, 4],
    [2, 1, 1, 3, 2, 1, 1, 1, 4],
    [2, 3, 1, 1, 1, 2, 1, 1, 4],  # 25
    [1, 1, 1, 2, 2, 3, 1, 1, 4],
    [1, 2, 1, 1, 2, 3, 1, 1, 4],
    [1, 2, 3, 1, 1, 2, 1, 1, 4],
    [2, 1, 1, 1, 2, 3, 1, 1, 4],
    [3, 1, 1, 1, 1, 2, 1, 2, 4],  # 30
    [1, 1, 1, 2, 1, 1, 3, 2, 4],
    [1, 1, 1, 3, 1, 2, 1, 2, 4],
    [1, 2, 1, 1, 1, 3, 1, 2, 4],
    [1, 2, 3, 1, 1, 1, 1, 2, 4],
    [2, 3, 1, 1, 2, 1, 1, 1, 4],  # 35

]
judged_scores = [1.56, 2.12, 2.08, 1.88, 1.80, 2.44, 2.20, 2.56, 3.00, 2.04, 2.76, 2.72, 3.00, 3.16, 2.04, 2.88, 2.60, 2.60, 2.64, 3.24, 3.08, 3.04, 3.04,
                 2.56, 2.56, 2.84, 3.60, 2.68, 3.28, 3.08, 3.52, 3.60, 3.04, 2.88, 3.08]

PS_scores = [calculate_PS(thebeat.Sequence.from_integer_ratios(pattern, value_of_one=125), 125)['PS'] for pattern in patterns]
"""