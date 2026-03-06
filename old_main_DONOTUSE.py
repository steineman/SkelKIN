# This is it, your final scene begins
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import fromchemKIN
import condition_objects
import comparator
import project_handler
import os
import time as timing

# user input, to be manipulated using GUI? xd
project_name = "CRECK_test"
input_folder = "CRECK_conditions/"
sensitivity_tolerance = 1e-3
excluded_species = ['CH4', 'CH3', 'H2', 'C2H2', 'C2H4', 'C2H6', 'C6H6']
model_type = "ChemKIN"
comparator_type = "max" # "lin", "log", "max"

# Optional files, depending on model choice
chemkin_kinet = "CRECK_2003_TOT_HT_SOOT_pyro_COMSOL.CKI.txt"
chemkin_thermo = "CRECK_2003_TOT_HT_SOOT_therm_pyro_COMSOL.CKT.txt"
comsol_therm_model = "thermo_model.mph"
comsol_plasma_model = "plasma_model.mph"

# Code start, good luck
def check_project_similarity():
    # check if it's the same project
    (file_model_type, file_comparator_type, file_sensitivity, file_excluded_species, file_test_conditions_folder,
     file_len_original_species_list, file_len_original_reactions_list) = project_handler.get_header_data()
    if file_model_type != model_type or file_comparator_type != comparator_type or file_sensitivity != sensitivity_tolerance or file_excluded_species != excluded_species or file_len_original_species_list != len(
            all_species) or file_len_original_reactions_list != len(
            all_reactions) or file_test_conditions_folder != input_folder:
        sys.exit('Current .skn file does not match your input conditions, please delete or rename the project file or open the .skn file and match conditions')

# All workers can only take one argument to ensure parallel runtimes, hence why it is in a set

def _run_standard_single_condition(condition):
    """
    Worker function for multiprocessing.
    Runs the standard model for one condition.
    """

    match model_type:
        case "ChemKIN":
            time_vals, result_dict = fromchemKIN.run_standard_model(condition)
            return time_vals, result_dict
        case "COMSOL_plasma":
            raise NotImplementedError("I didnt implement COMSOL plasma yet lole")
        case "COMSOL_thermal":
            raise NotImplementedError("I didnt implement COMSOL thermal yet lole")
        case _:
            raise ValueError(f"{model_type} doesn't exist")


def _run_reduced_species_worker(args):
    """
    Worker for reduced species simulations.
    args = (condition_index, condition, omitted_species)
    """
    condition_index, condition, omitted_species = args

    time_vals, result_dict = fromchemKIN.run_reduced_species_model(
        condition,
        [omitted_species] if isinstance(omitted_species, str) else omitted_species
    )

    return omitted_species, condition_index, time_vals, result_dict


def _run_reduced_reactions_worker(args):
    """
    Worker for reduced reactions simulations.
    args = (condition_index, condition, omitted_species, omitted_reactions)
    """
    condition_index, condition, omitted_species, omitted_reactions = args

    time_vals, result_dict = fromchemKIN.run_reduced_reactions_model(
        condition,
        [omitted_species] if isinstance(omitted_species, str) else omitted_species,
        [omitted_reactions] if isinstance(omitted_reactions, str) else omitted_reactions
    )

    return omitted_reactions, condition_index, time_vals, result_dict

# Boss of the workers

def run_standard_models_parallel(conditions):
    """
    Runs standard ChemKIN model for all conditions in parallel.
    Prints progress when each simulation finishes.
    If this doesn't run smooth, nothing will.
    """

    cpu_count = multiprocessing.cpu_count()
    total_jobs = len(conditions)

    print(f"Running {total_jobs} simulations using {cpu_count} CPU cores")

    results_time = [[]] * total_jobs
    results_dict = [{}] * total_jobs

    start_total = timing.time()

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Keep track of which future belongs to which index
        futures = {
            executor.submit(_run_standard_single_condition, cond): i
            for i, cond in enumerate(conditions)
        }

        completed = 0

        for future in as_completed(futures):
            index = futures[future]

            try:
                time_vals, result_dict = future.result()
                results_time[index] = time_vals
                results_dict[index] = result_dict
            except Exception as e:
                print(f"[ERROR] Simulation {index} failed: {e}")
                raise

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"Condition {index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

    print(f"All simulations finished in {timing.time() - start_total:.2f} seconds")

    return results_time, results_dict

def run_reduced_species_models_parallel(conditions, omitted_species_list):
    """
    Runs reduced species models in parallel.

    Total jobs:
        len(conditions) * len(omitted_species_list)

    Returns:
        dict[species][condition_index] = (time, result_dict)
    """

    cpu_count = multiprocessing.cpu_count()
    total_jobs = len(conditions) * len(omitted_species_list)

    print(f"Running {total_jobs} reduced-species simulations "
          f"using {cpu_count} CPU cores")

    start_total = timing.time()

    # Pre-build result structure
    results = {
        tuple(species) if isinstance(species, list) else species: {} for species in omitted_species_list
    }

    # Build job list
    job_args = []

    for species in omitted_species_list:
        for cond_index, condition in enumerate(conditions):
            job_args.append((cond_index, condition, species))

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:

        futures = [
            executor.submit(_run_reduced_species_worker, args)
            for args in job_args
        ]

        completed = 0

        for future in as_completed(futures):

            try:
                species, cond_index, time_vals, result_dict = future.result()
            except Exception as e:
                print(f"[ERROR] Reduced species job failed: {e}")
                raise

            results[tuple(species) if isinstance(species, list) else species][cond_index] = (time_vals, result_dict)

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"Species {species}, Condition {cond_index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

    print(f"All reduced-species simulations finished in "
          f"{timing.time() - start_total:.2f} seconds")

    return results

def run_reduced_reactions_models_parallel(conditions, final_omitted_species_list, omitted_reactions_list):
    """
    Runs reduced species models in parallel.

    Total jobs:
        len(conditions) * len(omitted_species_list)

    Returns:
        dict[species][condition_index] = (time, result_dict)
    """

    cpu_count = multiprocessing.cpu_count()
    total_jobs = len(conditions) * len(omitted_reactions_list)

    print(f"Running {total_jobs} reduced-species simulations "
          f"using {cpu_count} CPU cores")

    start_total = timing.time()

    # Pre-build result structure
    results = {
        reactions: {} for reactions in omitted_reactions_list
    }

    # Build job list
    job_args = []

    for reactions in omitted_reactions_list:
        for cond_index, condition in enumerate(conditions):
            job_args.append((cond_index, condition, final_omitted_species_list, reactions))

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:

        futures = [
            executor.submit(_run_reduced_reactions_worker, args)
            for args in job_args
        ]

        completed = 0

        for future in as_completed(futures):

            try:
                reactions, cond_index, time_vals, result_dict = future.result()
            except Exception as e:
                print(f"[ERROR] Reduced species job failed: {e}")
                raise

            results[reactions][cond_index] = (time_vals, result_dict)

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"Reactions {reactions}, Condition {cond_index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

    print(f"All reduced-species simulations finished in "
          f"{timing.time() - start_total:.2f} seconds")

    return results

def error_and_sort(standard : tuple[list[list], list[dict]], test_cases : dict[tuple | str, dict], comparator_mode) -> condition_objects.ItemErrorList:
    # make it make sense :S

    compare_timer_start = timing.time()

    errorList = condition_objects.ItemErrorList([])

    for item in test_cases:
        item_error = condition_objects.ItemError(item, 0.0, 0.0, 0.0)
        for i, item_condition in enumerate(conditions_list.conditions):
            match comparator_mode:
                case "max":
                    max_value = comparator.compare_max(standard[0][i], standard[1][i], test_cases[item][i][0],
                                                       test_cases[item][i][1])
                    item_error.add_to_value(max_value, 1)
                case "lin":
                    lin_value = comparator.compare_lin(standard[0][i], standard[1][i], test_cases[item][i][0],
                                                       test_cases[item][i][1])
                    item_error.add_to_value(lin_value, 1)
                case "log":
                    log_value = comparator.compare_log(standard[0][i], standard[1][i], test_cases[item][i][0],
                                                       test_cases[item][i][1])
                    item_error.add_to_value(log_value, 1)
                case _:
                    raise ValueError(f"Unknown comparator mode: {comparator_mode}")
        errorList.add_to_list(item_error)

        elapsed = timing.time() - compare_timer_start
        print(
            f"Item {item} added to error list "
            f"(total elapsed: {elapsed:.2f} s)"
        )

    # now sort it
    errorList.sort_items()

    elapsed = timing.time() - compare_timer_start
    print(
        f"All items added to error list "
        f"(total elapsed: {elapsed:.2f} s)"
    )

    return errorList


def inverse_item_pyramid_builder(item_error_list : condition_objects.ItemErrorList):
    # The inverse aliens wrote this one
    inverse_pyramid = []

    too_much_voodoo = []

    # I am not documenting this, figure it out, good luck
    for item in item_error_list.items[::-1]:
        too_much_voodoo.append(item.item)
        inverse_pyramid.append(too_much_voodoo.copy())

    return inverse_pyramid

if __name__ == "__main__":

    # Get the project file ready, see what is up
    project_handler.project_identifier = project_name

    status = project_handler.where_are_we() # "nothing", "initialized", "step 1", "step 2", "step 3", "step 4"

    # Reinitialize the project OR initialize it
    all_species = []
    all_reactions = []
    single_species_error = condition_objects.ItemErrorList
    grouped_species_error = condition_objects.ItemErrorList
    single_reaction_error = condition_objects.ItemErrorList
    grouped_reaction_error = condition_objects.ItemErrorList

    match model_type:
        case "ChemKIN":
            fromchemKIN.load_model_to_yaml(chemkin_kinet, chemkin_thermo)
            all_species = fromchemKIN.get_species()
            all_reactions = fromchemKIN.get_reactions()
        case "COMSOL_thermal":
            raise NotImplementedError("I didnt implement COMSOL thermal yet lole")
        case "COMSOL_plasma":
            raise NotImplementedError("I didnt implement COMSOL plasma yet lole")
        case _:
            raise ValueError("This model type doesn't exist, sorry :'(")

    # Get Error lists
    match status:
        case "nothing":
            project_handler.write_header(model_type, comparator_type, sensitivity_tolerance, excluded_species, input_folder, all_species, all_reactions)
        case "initialized":
            check_project_similarity()
        case "step 1":
            check_project_similarity()
            single_species_error = project_handler.get_me_data("step 1")
        case "step 2":
            check_project_similarity()
            single_species_error = project_handler.get_me_data("step 1")
            grouped_species_error = project_handler.get_me_data("step 2")
        case "step 3":
            check_project_similarity()
            single_species_error =  project_handler.get_me_data("step 1")
            grouped_species_error = project_handler.get_me_data("step 2")
            single_reaction_error = project_handler.get_me_data("step 3")
        case "step 4":
            check_project_similarity()
            single_species_error = project_handler.get_me_data("step 1")
            grouped_species_error = project_handler.get_me_data("step 2")
            single_reaction_error = project_handler.get_me_data("step 3")
            grouped_reaction_error = project_handler.get_me_data("step 4")

    # Take conditions from folder :)
    conditions_list = condition_objects.TestConditions([])

    for filename in os.listdir(input_folder):
        new_condition = condition_objects.load_thermal_condition(input_folder + "/" + filename)
        conditions_list.add_condition(new_condition)

    # Run standard models

    time_standard, dict_standard = run_standard_models_parallel(conditions_list.conditions)
    standard_data = (time_standard, dict_standard)

    # ------------------------------------------------------------------------------------------------------- #
    list_species = fromchemKIN.get_species()
    list_species = list(set(list_species) - set(excluded_species))

    #test_list_species = ['C', 'C3H4A', 'C6H5C4H9']
    # ------------------------------------------------------------------------------------------------------- #


    # --------------- STEP 1 -------------
    if status == "step 1" or status == "step 2" or status == "step 3" or status == "step 4":
        pass
    else:
        step_one_results = run_reduced_species_models_parallel(conditions_list.conditions, list_species)
        single_species_error = error_and_sort(standard_data, step_one_results, comparator_type)
        project_handler.write_step_error(1, single_species_error)


    # --------------- STEP 2 -------------
    if status == "step 2" or status == "step 3" or status == "step 4":
        pass
    else:
        # Group species from the previous error list
        # Most important species are on top, hence the inverse pyramid
        list_grouped_species = inverse_item_pyramid_builder(single_species_error)
        # step two
        step_two_results = run_reduced_species_models_parallel(conditions_list.conditions, list_grouped_species)
        grouped_species_error = error_and_sort(standard_data, step_two_results, comparator_type)
        project_handler.write_step_error(2, grouped_species_error)
