# This is it, your final scene begins
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

import fromchemKIN
import fromcomsolRE
import condition_objects
import comparator
import project_handler
import os
import time as timing

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# user input, to be manipulated using GUI? xd
project_name = "COMSOL_260309b"
input_folder = str(BASE_DIR / "COMSOL_260309b_conditions")
sensitivity_tolerance = 1e-3
excluded_species = ['CH4', 'CO2','H','H2','CO','H2O','CH3','O2','O','H2O']
model_type = "COMSOL_thermal"
comparator_type = "max" # "lin", "log", "max"
comsol_parallel_standard_runs = True

# Optional files, depending on model choice
chemkin_kinet = str(BASE_DIR / "CRECK_2003_TOT_HT_SOOT_pyro_COMSOL.CKI.txt")
chemkin_thermo = str(BASE_DIR / "CRECK_2003_TOT_HT_SOOT_therm_pyro_COMSOL.CKT.txt")
comsol_therm_model = str(BASE_DIR / "0D_260309b.mph")
comsol_plasma_model = str(BASE_DIR / "plasma_model.mph")


def _initialize_worker(comsol_cores_per_task: int = 1):
    """Load the selected model once per worker process."""

    match model_type:
        case "ChemKIN":
            fromchemKIN.load_model_to_yaml(chemkin_kinet, chemkin_thermo)
        case "COMSOL_thermal":
            fromcomsolRE.set_comsol_cores(comsol_cores_per_task)
            fromcomsolRE.load_comsol_model(comsol_therm_model)
        case "COMSOL_plasma":
            raise NotImplementedError("I didnt implement COMSOL plasma yet lole")
        case _:
            raise ValueError(f"{model_type} doesn't exist")


def _pool_configuration(total_jobs: int, *, default_cap: int | None = None) -> tuple[int, int]:
    """Choose worker count and per-task CPU allocation.

    Rules:
    - reserve at least one CPU outside the worker pool whenever possible
    - if there is enough headroom, assign multiple CPUs per COMSOL task
    - if there are more jobs than workers, the executor queue becomes the schedule
    """

    cpu_count = multiprocessing.cpu_count()
    reserved_cpus = 1 if cpu_count > 1 else 0
    available_cpus = max(1, cpu_count - reserved_cpus)
    cap = available_cpus if default_cap is None else min(available_cpus, default_cap)

    if model_type == "COMSOL_thermal":
        cap = min(cap, 10)

    worker_count = max(1, min(total_jobs, cap))
    comsol_cores_per_task = 1

    if model_type == "COMSOL_thermal":
        comsol_cores_per_task = max(1, available_cpus // worker_count)

    return worker_count, comsol_cores_per_task
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

    try:
        match model_type:
            case "ChemKIN":
                time_vals, result_dict = fromchemKIN.run_standard_model(condition)
                return time_vals, result_dict
            case "COMSOL_plasma":
                raise NotImplementedError("I didnt implement COMSOL plasma yet lole")
            case "COMSOL_thermal":
                time_vals, result_dict = fromcomsolRE.run_standard_model(condition)
                return time_vals, result_dict
            case _:
                raise ValueError(f"{model_type} doesn't exist")
    except Exception as exc:
        raise RuntimeError(str(exc)) from None


def _run_standard_condition_group(group):
    """Run one ordered group of COMSOL conditions inside a single worker."""

    indexed_conditions = list(group)
    ordered_conditions = [condition for _, condition in indexed_conditions]

    try:
        match model_type:
            case "COMSOL_thermal":
                time_vals_list, result_dicts = fromcomsolRE.run_standard_models(ordered_conditions)
                return [
                    (index, time_vals, result_dict)
                    for (index, _), time_vals, result_dict in zip(indexed_conditions, time_vals_list, result_dicts)
                ]
            case _:
                raise ValueError("Grouped standard runs are only used for COMSOL thermal models.")
    except Exception as exc:
        raise RuntimeError(str(exc)) from None


def _group_comsol_standard_conditions(conditions):
    """Group COMSOL standard conditions to preserve useful continuation paths."""

    groups: dict[float, list[tuple[int, object]]] = {}

    for index, condition in enumerate(conditions):
        key = float(condition.get_molar_fraction("CH4") or 0.0)
        groups.setdefault(key, []).append((index, condition))

    ordered_groups = []
    for key in sorted(groups):
        ordered_groups.append(
            sorted(
                groups[key],
                key=lambda item: (
                    item[1].get_temperature()[1][0],
                    item[1].get_temperature()[1][-1],
                    item[0],
                ),
            )
        )

    return ordered_groups


def _ordered_comsol_study_conditions(conditions):
    """Return the conditions in the original COMSOL study order."""

    return sorted(
        list(enumerate(conditions)),
        key=lambda item: (
            0 if item[1].get_temperature()[1][0] < item[1].get_temperature()[1][-1] else 1,
            float(item[1].get_molar_fraction("CH4") or 0.0),
            item[0],
        ),
    )


def _run_standard_condition_with_gui_continuation(target_index, conditions, comsol_cores_per_task: int):
    """Replay the saved COMSOL sweep order up to one target condition."""

    ordered_conditions = _ordered_comsol_study_conditions(conditions)
    target_position = next(
        position
        for position, (index, _) in enumerate(ordered_conditions)
        if index == target_index
    )
    replay_conditions = [condition for _, condition in ordered_conditions[:target_position + 1]]

    fromcomsolRE.set_comsol_cores(comsol_cores_per_task)
    time_values, result_dicts = fromcomsolRE.run_standard_models(replay_conditions)
    return time_values[-1], result_dicts[-1]


def _reduced_species_error_worker(args):
    """
    args = (condition_index, condition, omitted_species,
            standard_time, standard_dict, comparator_mode)
    """
    (cond_index,
     condition,
     omitted_species,
     standard_time,
     standard_dict,
     comparator_mode) = args

    # Run reduced simulation
    try:
        match model_type:
            case "ChemKIN":
                time_vals, result_dict = fromchemKIN.run_reduced_species_model(
                    condition,
                    [omitted_species] if isinstance(omitted_species, str) else omitted_species
                )
            case "COMSOL_thermal":
                time_vals, result_dict = fromcomsolRE.run_reduced_species_model(
                    condition,
                    [omitted_species] if isinstance(omitted_species, str) else omitted_species
                )
            case _:
                raise ValueError(f"{model_type} doesn't exist")
    except Exception as exc:
        raise RuntimeError(str(exc)) from None

    # Compute error immediately
    match comparator_mode:
        case "max":
            error_value = comparator.compare_max(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case "lin":
            error_value = comparator.compare_lin(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case "log":
            error_value = comparator.compare_log(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case _:
            raise ValueError(f"Unknown comparator mode: {comparator_mode}")

    # Drop big arrays automatically when function exits
    return omitted_species, cond_index, error_value


def _reduced_reactions_error_worker(args):
    """
    args = (condition_index,
            condition,
            omitted_species,
            omitted_reactions,
            standard_time,
            standard_dict,
            comparator_mode)
    """

    (cond_index,
     condition,
     omitted_species,
     omitted_reactions,
     standard_time,
     standard_dict,
     comparator_mode) = args

    # Run reduced reaction simulation
    try:
        match model_type:
            case "ChemKIN":
                time_vals, result_dict = fromchemKIN.run_reduced_reactions_model(
                    condition,
                    [omitted_species] if isinstance(omitted_species, str) else omitted_species,
                    [omitted_reactions] if isinstance(omitted_reactions, str) else omitted_reactions
                )
            case "COMSOL_thermal":
                time_vals, result_dict = fromcomsolRE.run_reduced_reactions_model(
                    condition,
                    [omitted_species] if isinstance(omitted_species, str) else omitted_species,
                    [omitted_reactions] if isinstance(omitted_reactions, str) else omitted_reactions
                )
            case _:
                raise ValueError(f"{model_type} doesn't exist")
    except Exception as exc:
        raise RuntimeError(str(exc)) from None

    # Compute error immediately
    match comparator_mode:
        case "max":
            error_value = comparator.compare_max(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case "lin":
            error_value = comparator.compare_lin(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case "log":
            error_value = comparator.compare_log(
                standard_time,
                standard_dict,
                time_vals,
                result_dict
            )
        case _:
            raise ValueError(f"Unknown comparator mode: {comparator_mode}")

    # Big arrays die here when function exits
    return omitted_reactions, cond_index, error_value

# Boss of the workers

def run_standard_models_parallel(conditions):
    """
    Runs standard ChemKIN model for all conditions in parallel.
    Prints progress when each simulation finishes.
    If this doesn't run smooth, nothing will.
    """

    total_jobs = len(conditions)

    if model_type == "COMSOL_thermal" and comsol_parallel_standard_runs:
        condition_groups = _group_comsol_standard_conditions(conditions)
        worker_count, comsol_cores_per_task = _pool_configuration(len(condition_groups))

        print(
            f"Running {total_jobs} simulations using {worker_count} worker processes"
            f" with {comsol_cores_per_task} CPU cores per COMSOL task"
        )
        if len(condition_groups) < total_jobs:
            print(f"Bundled the conditions into {len(condition_groups)} ordered COMSOL sweep groups")

        results_time = [[]] * total_jobs
        results_dict = [{}] * total_jobs
        start_total = timing.time()
        failed_group_indices = []

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_worker,
            initargs=(comsol_cores_per_task,),
        ) as executor:
            futures = {
                executor.submit(_run_standard_condition_group, group): [index for index, _ in group]
                for group in condition_groups
            }

            completed = 0

            for future in as_completed(futures):
                group_indices = futures[future]

                try:
                    grouped_results = future.result()
                    for index, time_vals, result_dict in grouped_results:
                        results_time[index] = time_vals
                        results_dict[index] = result_dict
                        completed += 1
                        elapsed = timing.time() - start_total
                        print(
                            f"[{completed}/{total_jobs}] "
                            f"Condition {index} finished "
                            f"(elapsed: {elapsed:.2f} s)"
                        )
                except Exception as exc:
                    print(f"[ERROR] Condition group {group_indices} failed: {exc}")
                    failed_group_indices.extend(group_indices)

        if failed_group_indices:
            for index in sorted(set(failed_group_indices)):
                elapsed = timing.time() - start_total
                print(
                    f"Retrying condition {index} in GUI-order continuation mode "
                    f"(elapsed: {elapsed:.2f} s)"
                )
                time_vals, result_dict = _run_standard_condition_with_gui_continuation(
                    index,
                    conditions,
                    comsol_cores_per_task,
                )
                results_time[index] = time_vals
                results_dict[index] = result_dict

                completed = sum(1 for item in results_time if item)
                elapsed = timing.time() - start_total
                print(
                    f"[{completed}/{total_jobs}] "
                    f"Condition {index} finished via continuation replay "
                    f"(elapsed: {elapsed:.2f} s)"
                )

        print(f"All simulations finished in {timing.time() - start_total:.2f} seconds")
        return results_time, results_dict

    if model_type == "COMSOL_thermal" and not comsol_parallel_standard_runs:
        print(f"Running {total_jobs} simulations in one COMSOL parametric study")
        start_total = timing.time()
        try:
            results_time, results_dict = fromcomsolRE.run_standard_models(conditions)
        except Exception as exc:
            print(f"[ERROR] Standard COMSOL study failed: {exc}")
            raise

        for index in range(total_jobs):
            elapsed = timing.time() - start_total
            print(
                f"[{index + 1}/{total_jobs}] "
                f"Condition {index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

        print(f"All simulations finished in {timing.time() - start_total:.2f} seconds")
        return results_time, results_dict

    worker_count, comsol_cores_per_task = _pool_configuration(total_jobs)

    print(
        f"Running {total_jobs} simulations using {worker_count} worker processes"
        + (f" with {comsol_cores_per_task} CPU cores per COMSOL task" if model_type == "COMSOL_thermal" else "")
    )
    if total_jobs > worker_count:
        print(f"Scheduling {total_jobs - worker_count} simulations in the queue")

    results_time = [[]] * total_jobs
    results_dict = [{}] * total_jobs

    start_total = timing.time()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(comsol_cores_per_task,),
    ) as executor:
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


def run_reduced_species_error_parallel(
        conditions,
        omitted_species_list,
        standard_data,
        comparator_mode):
    total_jobs = len(conditions) * len(omitted_species_list)
    worker_count, comsol_cores_per_task = _pool_configuration(total_jobs)

    print(f"Running {total_jobs} reduced-species simulations "
          f"using {worker_count} worker processes"
          + (f" with {comsol_cores_per_task} CPU cores per COMSOL task" if model_type == "COMSOL_thermal" else ""))
    if total_jobs > worker_count:
        print(f"Scheduling {total_jobs - worker_count} reduced-species simulations in the queue")

    start_total = timing.time()

    # Accumulate errors per species
    species_errors = {
        tuple(s) if isinstance(s, list) else s: condition_objects.ItemError(tuple(s) if isinstance(s, list) else s, 0, 0 ,0)
        for s in omitted_species_list
    }

    job_args = []

    for species in omitted_species_list:
        for cond_index, condition in enumerate(conditions):
            job_args.append((
                cond_index,
                condition,
                species,
                standard_data[0][cond_index],  # standard time
                standard_data[1][cond_index],  # standard dict
                comparator_mode
            ))

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(comsol_cores_per_task,),
    ) as executor:

        futures = [
            executor.submit(_reduced_species_error_worker, args)
            for args in job_args
        ]

        completed = 0

        for future in as_completed(futures):
            species, cond_index, error_value = future.result()

            key = tuple(species) if isinstance(species, list) else species
            species_errors[key].add_to_value(error_value)

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"{species}, Condition {cond_index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

    # Build ItemErrorList
    errorList = condition_objects.ItemErrorList([])

    for species in species_errors:
        errorList.add_to_list(species_errors[species])

    errorList.sort_items()

    print(f"All reduced-species simulations finished in "
          f"{timing.time() - start_total:.2f} seconds")

    return errorList

def run_reduced_reactions_error_parallel(
        conditions,
        final_omitted_species_list,
        omitted_reactions_list,
        standard_data,
        comparator_mode):
    total_jobs = len(conditions) * len(omitted_reactions_list)
    worker_count, comsol_cores_per_task = _pool_configuration(total_jobs, default_cap=8)

    print(f"Running {total_jobs} reduced-reaction simulations "
          f"using {worker_count} worker processes"
          + (f" with {comsol_cores_per_task} CPU cores per COMSOL task" if model_type == "COMSOL_thermal" else ""))
    if total_jobs > worker_count:
        print(f"Scheduling {total_jobs - worker_count} reduced-reaction simulations in the queue")

    start_total = timing.time()

    reaction_errors = {
        reactions: 0.0
        for reactions in omitted_reactions_list
    }

    reaction_counts = {
        reactions: 0
        for reactions in omitted_reactions_list
    }

    job_args = []

    for reactions in omitted_reactions_list:
        for cond_index, condition in enumerate(conditions):
            job_args.append((
                cond_index,
                condition,
                final_omitted_species_list,
                reactions,
                standard_data[0][cond_index],
                standard_data[1][cond_index],
                comparator_mode
            ))

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_worker,
        initargs=(comsol_cores_per_task,),
    ) as executor:

        futures = [
            executor.submit(_reduced_reactions_error_worker, args)
            for args in job_args
        ]

        completed = 0

        for future in as_completed(futures):
            reactions, cond_index, error_value = future.result()

            reaction_errors[reactions] += error_value
            reaction_counts[reactions] += 1

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"{reactions}, Condition {cond_index} finished "
                f"(elapsed: {elapsed:.2f} s)"
            )

    # Build ItemErrorList
    errorList = condition_objects.ItemErrorList([])

    for reactions in reaction_errors:
        item_error = condition_objects.ItemError(
            reactions,
            reaction_errors[reactions],
            0.0,
            0.0
        )
        errorList.add_to_list(item_error)

    errorList.sort_items()

    print(f"All reduced-reaction simulations finished in "
          f"{timing.time() - start_total:.2f} seconds")

    return errorList

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
    project_handler.project_identifier = str(BASE_DIR / project_name)

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
            fromcomsolRE.load_comsol_model(comsol_therm_model)
            all_species = fromcomsolRE.get_species()
            all_reactions = fromcomsolRE.get_reactions()
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

    for filename in sorted(os.listdir(input_folder)):
        condition_path = Path(input_folder) / filename
        new_condition = condition_objects.load_thermal_condition(str(condition_path))
        conditions_list.add_condition(new_condition)

    # Run standard models

    time_standard, dict_standard = run_standard_models_parallel(conditions_list.conditions)
    standard_data = (time_standard, dict_standard)

    # ------------------------------------------------------------------------------------------------------- #
    list_species = list(set(all_species) - set(excluded_species))
    # ------------------------------------------------------------------------------------------------------- #


    # --------------- STEP 1 -------------
    if status == "step 1" or status == "step 2" or status == "step 3" or status == "step 4":
        pass
    else:
        single_species_error = run_reduced_species_error_parallel(
            conditions_list.conditions,
            list_species,
            standard_data,
            comparator_type
        )

        project_handler.write_step_error(1, single_species_error)


    # --------------- STEP 2 -------------
    if status == "step 2" or status == "step 3" or status == "step 4":
        pass
    else:
        # Group species from the previous error list
        # Most important species are on top, hence the inverse pyramid
        list_grouped_species = inverse_item_pyramid_builder(single_species_error)
        # step two
        grouped_species_error = run_reduced_species_error_parallel(
            conditions_list.conditions,
            list_grouped_species,
            standard_data,
            comparator_type
        )

        project_handler.write_step_error(2, grouped_species_error)
