import pandas as pd
import thebeat
import numpy.typing as npt
import re
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

def get_binary_sequence(iois: npt.ArrayLike, grid_ioi: float):

    output = []

    for ioi in iois:
        if ioi == grid_ioi:
            output += [1]
        else:
            output += [1]
            output += [0] * (int(ioi / grid_ioi) - 1)  # can be zero

    return output

def add_accents(binary_sequence: str):
    """
    See Povel & Essens (1985). 0 means silence, 1 means unaccented event, 2 means accented event.
    """
    output = []
    count = 0

    while count < len(binary_sequence):
        # If an event is isolated (anywhere in the sequence)
        if binary_sequence[count:count+3] == '010':
            output += '020'
            count += 3
        # If an event is isolated at the beginning of the sequence
        elif count == 0 and binary_sequence[count:count+2] == '10':
            output += '20'
            count += 2
        # If an event is isolated at the end of the sequence
        elif count == len(binary_sequence) - 2 and binary_sequence[count:count+2] == '01':
            output += '02'
            count += 2
        # If there's three OR MORE consecutive 1s:
        elif binary_sequence[count:count+3] == '111':
            m = re.search(r"111+", binary_sequence[count:])
            n_ones = len(m.group(0))
            output += '2'
            output += '1' * (n_ones - 2)
            output += '2'
            count += n_ones
        # If there's two consecutive 1s:
        elif binary_sequence[count:count+2] == '11':
            output += '12'
            count += 2
        # Otherwise, add the event to the output, and move on
        else:
            output += binary_sequence[count]
            count += 1

    return ''.join(output)



def calculate_PS(sequence: Union[thebeat.core.Sequence, thebeat.music.Rhythm], grid_ioi: float):
    """
    Calculates the PS measure for a given Sequence object.

    Assumptions:

    - We assume that the sequence starts at t=0, so there can be no silence at the beginning
    - We assume that the provided Sequence constitutes one 'period'.

    Parameters
    ----------

    sequence : thebeat.core.Sequence or thebeat.music.Rhythm
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

    if np.any(sequence.iois < grid_ioi):
        raise ValueError("One of the inter-onset intervals (IOIs) is shorter than the grid_ioi.")

    # Get the binary sequence
    binary_sequence = ''.join([str(x) for x in get_binary_sequence(iois=sequence.iois, grid_ioi=grid_ioi)])
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
    # Split up the binary sequence into parts of size final_u, starting at final_loc
    segments = [binary_sequence[i:i+final_u] for i in range(final_loc - 1, len(binary_sequence), final_u)]
    # Discard segments that are shorter than final_u
    segments = [segment for segment in segments if len(segment) == final_u]
    # calculate the sum of the c_i's
    ci_sum = 0
    for segment in segments:
        # segment starts with silence (type N)
        if segment.startswith('0'):
            ci_sum += WEIGHTS['d4']
        else:  # segment is either empty, or equally or unequally subdivided
            # get the durations of the segment (e.g. 1-2, 1-1, etc.)
            segment_durations = re.findall(r"10*", segment)
            segment_durations = [len(x) for x in segment_durations]
            if len(segment_durations) == 1:  # segment is empty if there is only one value (type E)
                ci_sum += WEIGHTS['d1']
            elif all(x == segment_durations[0] for x in segment_durations):  # equally subdivided if all durations are the same
                ci_sum += WEIGHTS['d2']
            else:  # unequally subdivided if not all durations are the same
                ci_sum += WEIGHTS['d3']

    # calculate m (i.e. number of 'new' strings when going from left to right)
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


