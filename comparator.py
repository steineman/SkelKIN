# The comparator should get the books and compare a standard simulation result to a reduced option
# The comparator should return possible errors either in linear time or in logarithmic time
# The comparator should NOT remove a species or reaction and should only give an error back
# The comparator doesn't care about units, but assumes they are the same for the reduced and standard data

import warnings
import bisect
import random
from math import log10
from numba import njit
import numpy as np
import time

@njit
def _interp_sweep(source_time, source_values, target_time):
    n = source_time.shape[0]
    m = target_time.shape[0]

    result = np.empty(m)
    j = 0

    for i in range(m):
        t = target_time[i]

        if t <= source_time[0]:
            result[i] = source_values[0]
            continue
        if t >= source_time[n - 1]:
            result[i] = source_values[n - 1]
            continue

        while j + 1 < n and source_time[j + 1] < t:
            j += 1

        t0 = source_time[j]
        t1 = source_time[j + 1]
        v0 = source_values[j]
        v1 = source_values[j + 1]

        result[i] = v0 + (t - t0) * (v1 - v0) / (t1 - t0)

    return result

@njit
def _compare_lin_kernel(std_time, red_time,
                        std_vals, red_vals,
                        extended_time):

    orig_interp = _interp_sweep(std_time, std_vals, extended_time)
    red_interp  = _interp_sweep(red_time, red_vals, extended_time)

    cumulative_error = 0.0

    for i in range(1, extended_time.shape[0]):
        dt = extended_time[i] - extended_time[i - 1]
        diff = abs(orig_interp[i] - red_interp[i])
        cumulative_error += diff * dt

    return cumulative_error

@njit
def _compare_log_kernel(std_time, red_time,
                        std_vals, red_vals,
                        extended_time):

    orig_interp = _interp_sweep(std_time, std_vals, extended_time)
    red_interp  = _interp_sweep(red_time, red_vals, extended_time)

    cumulative_error = 0.0

    for i in range(1, extended_time.shape[0]):

        t0 = extended_time[i - 1]
        t1 = extended_time[i]

        if t0 == 0.0 or t1 == 0.0:
            dt = 0.0
        else:
            dt = log10(t1) - log10(t0)

        diff = abs(orig_interp[i] - red_interp[i])
        cumulative_error += diff * dt

    return cumulative_error

@njit
def _compare_max_kernel(std_time, red_time,
                        std_vals, red_vals,
                        extended_time):

    orig_interp = _interp_sweep(std_time, std_vals, extended_time)
    red_interp  = _interp_sweep(red_time, red_vals, extended_time)

    maximum_error = 0.0

    for i in range(extended_time.shape[0]):
        diff = abs(orig_interp[i] - red_interp[i])
        if diff > maximum_error:
            maximum_error = diff

    return maximum_error

def _interpolate_to_grid(source_time: list, source_values: list, target_time: list) -> list:
    """
    Interpolates source_values defined on source_time onto target_time.
    Uses constant extrapolation outside bounds.
    Assumes source_time is sorted, which it is
    """

    n = len(source_time)
    result = [0.0] * len(target_time)

    # Precompute index lookup for exact matches
    time_to_index = {t: i for i, t in enumerate(source_time)}

    for i, t in enumerate(target_time):

        # Exact match (O(1))
        if t in time_to_index:
            result[i] = source_values[time_to_index[t]]
            continue

        # Constant extrapolation
        if t <= source_time[0]:
            result[i] = source_values[0]
            continue
        if t >= source_time[-1]:
            result[i] = source_values[-1]
            continue

        # Proper interpolation (O(log n))
        idx = bisect.bisect_left(source_time, t)
        lower = idx - 1
        upper = idx

        t0 = source_time[lower]
        t1 = source_time[upper]
        v0 = source_values[lower]
        v1 = source_values[upper]

        result[i] = v0 + (t - t0) * (v1 - v0) / (t1 - t0)

    return result

# unused original code, my O(n^2*log(n)) genius crashed the pc, again...
def extend_and_interpolate_gpt(
    standard_time: list,
    standard_dictionary: dict[str, list],
    reduced_time: list,
    reduced_dictionary: dict[str, list]
) -> tuple[list, dict[str, list], dict[str, list]]:

    # --- sanity checks ---
    if standard_dictionary:
        if len(standard_time) != len(next(iter(standard_dictionary.values()))):
            warnings.warn("Standard time array does not match standard species array")

        if len({len(v) for v in standard_dictionary.values()}) != 1:
            warnings.warn("Standard dictionary species have different lengths")

    if reduced_dictionary:
        if len(reduced_time) != len(next(iter(reduced_dictionary.values()))):
            warnings.warn("Reduced time array does not match reduced species array")

        if len({len(v) for v in reduced_dictionary.values()}) != 1:
            warnings.warn("Reduced dictionary species have different lengths")

    # --- unified sorted time grid ---
    extended_time = sorted(set(standard_time) | set(reduced_time))

    total_extended_original = {}
    total_extended_reduced = {}

    # Iterate only over standard species
    for species, original_values in standard_dictionary.items():

        reduced_values = reduced_dictionary.get(
            species,
            [0.0] * len(reduced_time)
        )

        total_extended_original[species] = _interpolate_to_grid(
            standard_time, original_values, extended_time
        )

        total_extended_reduced[species] = _interpolate_to_grid(
            reduced_time, reduced_values, extended_time
        )

    return extended_time, total_extended_original, total_extended_reduced

# unused original code, my O(n^2) genius crashed the pc
def extend_and_interpolate(standard_time: list, standard_dictionary: dict[str, list],
                           reduced_time: list, reduced_dictionary: dict[str, list]) -> tuple[list, dict[str, list], dict[str, list]]:

    # standard warnings
    if len(standard_time) != len(random.choice(list(standard_dictionary.values()))):
        warnings.warn("Standard time array does not match standard species array")
    if len(reduced_time) != len(random.choice(list(reduced_dictionary.values()))):
        warnings.warn("Reduced time array does not match reduced species array")
    if len({len(sublist) for sublist in standard_dictionary.values()}) != 1:
        warnings.warn("Standard dictionary species have different lengths")
    if len({len(sublist) for sublist in reduced_dictionary.values()}) != 1:
        warnings.warn("Reduced dictionary species have different lengths")

    # Create a longer time list if needed

    extended_time = list(set(standard_time + reduced_time))
    extended_time.sort()

    total_extended_original = {}
    total_extended_reduced = {}

    for original_species in standard_dictionary.keys():
        original_values = standard_dictionary[original_species]
        if original_species in reduced_dictionary.keys():
            reduced_values = reduced_dictionary[original_species]
        else:
            reduced_values = len(reduced_time)*[0.0]

        extended_original_values = len(extended_time)*[0.0]
        extended_reduced_values = len(extended_time)*[0.0]

        # Linear interpolation of missing data btw, a quick btw.
        for i, time in enumerate(extended_time):
            if time in standard_time:
                extended_original_values[i] = original_values[standard_time.index(time)]
            if time not in standard_time:
                if time <= standard_time[0]: # constant extrapolation
                    extended_original_values[i] = original_values[0]
                elif time >= standard_time[-1]: # constant extrapolation
                    extended_original_values[i] = original_values[-1]
                else: # interpolate
                    idx = bisect.bisect_left(standard_time, time)
                    lower_idx = idx - 1
                    upper_idx = idx

                    # linear interpolation
                    input_value = original_values[lower_idx] + (time - standard_time[lower_idx])*((original_values[upper_idx] - original_values[lower_idx])/(standard_time[upper_idx] - standard_time[lower_idx]))

                    extended_original_values[i] = input_value

        # and for the reduced
        for i, time in enumerate(extended_time):
            if time in reduced_time:
                extended_reduced_values[i] = reduced_values[reduced_time.index(time)]
            if time not in reduced_time:
                if time <= reduced_time[0]:  # constant extrapolation
                    extended_reduced_values[i] = reduced_values[0]
                elif time >= reduced_time[-1]:  # constant extrapolation
                    extended_reduced_values[i] = reduced_values[-1]
                else:  # interpolate
                    idx = bisect.bisect_left(reduced_time, time)
                    lower_idx = idx - 1
                    upper_idx = idx

                    # linear interpolation
                    input_value = reduced_values[lower_idx] + (time - reduced_time[lower_idx]) * (
                                (reduced_values[upper_idx] - reduced_values[lower_idx]) / (
                                    reduced_time[upper_idx] - reduced_time[lower_idx]))

                    extended_reduced_values[i] = input_value

        total_extended_original[original_species] = extended_original_values
        total_extended_reduced[original_species] = extended_reduced_values

        # Time integrated difference added to error
        # All errors are absolute, cry me a river

    return extended_time, total_extended_original, total_extended_reduced


def compare_lin(standard_time : list, standard_dictionary : dict[str, list],
                reduced_time : list, reduced_dictionary : dict[str, list]) -> float:

    extended_time = np.array(
        sorted(set(standard_time) | set(reduced_time)),
        dtype=np.float64
    )

    std_time = np.array(standard_time, dtype=np.float64)
    red_time = np.array(reduced_time, dtype=np.float64)

    cumulative_error = 0.0

    for species, std_vals_list in standard_dictionary.items():

        red_vals_list = reduced_dictionary.get(
            species,
            [0.0] * len(reduced_time)
        )

        std_vals = np.array(std_vals_list, dtype=np.float64)
        red_vals = np.array(red_vals_list, dtype=np.float64)

        cumulative_error += _compare_lin_kernel(
            std_time, red_time,
            std_vals, red_vals,
            extended_time
        )

    modelled_time = extended_time[-1] - extended_time[0]
    return cumulative_error / modelled_time



def compare_log(standard_time : list, standard_dictionary : dict[str, list],
                reduced_time : list, reduced_dictionary : dict[str, list]) -> float:

    extended_time = np.array(
        sorted(set(standard_time) | set(reduced_time)),
        dtype=np.float64
    )

    std_time = np.array(standard_time, dtype=np.float64)
    red_time = np.array(reduced_time, dtype=np.float64)

    cumulative_error = 0.0

    for species, std_vals_list in standard_dictionary.items():

        red_vals_list = reduced_dictionary.get(
            species,
            [0.0] * len(reduced_time)
        )

        std_vals = np.array(std_vals_list, dtype=np.float64)
        red_vals = np.array(red_vals_list, dtype=np.float64)

        cumulative_error += _compare_log_kernel(
            std_time, red_time,
            std_vals, red_vals,
            extended_time
        )

    start_time = extended_time[0] if extended_time[0] != 0 else extended_time[1]
    modelled_log_time = abs(
        np.log10(extended_time[-1]) - np.log10(start_time)
    )

    return cumulative_error / modelled_log_time


def compare_max(standard_time : list, standard_dictionary : dict[str, list],
                reduced_time : list, reduced_dictionary :  dict[str, list]) -> float:

    extended_time = np.array(
        sorted(set(standard_time) | set(reduced_time)),
        dtype=np.float64
    )

    std_time = np.array(standard_time, dtype=np.float64)
    red_time = np.array(reduced_time, dtype=np.float64)

    maximum_error = 0.0

    for species, std_vals_list in standard_dictionary.items():

        red_vals_list = reduced_dictionary.get(
            species,
            [0.0] * len(reduced_time)
        )

        std_vals = np.array(std_vals_list, dtype=np.float64)
        red_vals = np.array(red_vals_list, dtype=np.float64)

        err = _compare_max_kernel(
            std_time, red_time,
            std_vals, red_vals,
            extended_time
        )

        if err > maximum_error:
            maximum_error = err

    return maximum_error

#len_lists = 700000
#
#dict_een = {'CH4' : [7]*len_lists, 'H2' : [3]*len_lists}
#dict_twee = {'CH4' : [8]*len_lists, 'H2' : [2]*len_lists}
#time_een = [i/(len_lists*100) for i in list(range(0, len_lists, 1))]
#time_twee = [(i+0.5)/(len_lists*100) for i in list(range(0, len_lists, 1))]
#
#start_time = time.time()
#
#print(compare_log(time_een, dict_een, time_twee, dict_twee))
#print(compare_lin(time_een, dict_een, time_twee, dict_twee))
#print(compare_max(time_een, dict_een, time_twee, dict_twee))
#
#print(time.time()-start_time, 'seconds')