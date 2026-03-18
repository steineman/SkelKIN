# This is it, your final scene begins
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import queue

import fromchemKIN
import fromcomsolRE
import condition_objects
import comparator
import project_handler
import os
import time as timing

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# user input, to be manipulated using GUI?
project_name = "COMSOL_260312"
input_folder = str(BASE_DIR / "COMSOL_260312_conditions")
sensitivity_tolerance = 1e-3
excluded_species = ['CH4', 'CO2','H','H2','CO','H2O','CH3','CH2','O2','O']
model_type = "COMSOL_thermal"
comparator_type = "max" # "lin", "log", "max"
comsol_parallel_standard_runs = True
COMSOL_REDUCTION_WORKER_CAP = 5
COMSOL_MAX_CORES_PER_TASK = 8
COMSOL_REDUCED_SPECIES_TIMEOUT_SECONDS = 7200
TIMEOUT_ERROR_VALUE = float("1.0")

# Optional files, depending on model choice
chemkin_kinet = str(BASE_DIR / "CRECK_2003_TOT_HT_SOOT_pyro_COMSOL.CKI.txt")
chemkin_thermo = str(BASE_DIR / "CRECK_2003_TOT_HT_SOOT_therm_pyro_COMSOL.CKT.txt")
comsol_therm_model = str(BASE_DIR / "0D_260312.mph")
comsol_plasma_model = str(BASE_DIR / "plasma_model.mph")


def _initialize_worker(comsol_cores_per_task: int = 1):
    """Load the selected model once per worker process."""

    match model_type:
        case "ChemKIN":
            fromchemKIN.load_model_to_yaml(chemkin_kinet, chemkin_thermo)
        case "COMSOL_thermal":
            fromcomsolRE.set_comsol_session_mode("client-server")
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
        raw_cores_per_task = min(COMSOL_MAX_CORES_PER_TASK, max(1, available_cpus // worker_count))
        if raw_cores_per_task > 1 and raw_cores_per_task % 2 == 1:
            raw_cores_per_task -= 1
        comsol_cores_per_task = max(1, raw_cores_per_task)

    return worker_count, comsol_cores_per_task
# Code start, good luck
def check_project_similarity():
    # check if it's the same project
    (
        file_model_type,
        file_comparator_type,
        file_sensitivity,
        file_excluded_species,
        file_test_conditions_folder,
        file_len_original_species_list,
        file_len_original_reactions_list,
    ) = project_handler.get_header_data()
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


def _worker_identifier() -> int:
    """Return a stable worker number for progress reporting."""

    identity = multiprocessing.current_process()._identity
    if identity:
        return int(identity[0])
    return 0



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
    worker_id = _worker_identifier()
    start_worker = timing.perf_counter()

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
    return omitted_species, cond_index, error_value, worker_id, timing.perf_counter() - start_worker


def _compare_error_value(standard_time, standard_dict, time_vals, result_dict, comparator_mode):
    """Compute one comparison value for a reduced run."""

    match comparator_mode:
        case "max":
            return comparator.compare_max(standard_time, standard_dict, time_vals, result_dict)
        case "lin":
            return comparator.compare_lin(standard_time, standard_dict, time_vals, result_dict)
        case "log":
            return comparator.compare_log(standard_time, standard_dict, time_vals, result_dict)
        case _:
            raise ValueError(f"Unknown comparator mode: {comparator_mode}")


def _item_key(item):
    """Normalize a progress item so completed work can be matched on resume."""

    if isinstance(item, list):
        return tuple(item)
    return item


def _timeout_comment(timeout_seconds: float) -> str:
    """Return the `.skn` timeout annotation for one stalled simulation."""

    return f"TIMEOUT SPECIES : {int(timeout_seconds)} s"


def _failed_species_comment(reason: str) -> str:
    """Return the `.skn` annotation for one non-convergent reduced-species case."""

    del reason
    return "Failed Species"


def _is_nonremovable_species_error(item_error: condition_objects.ItemError) -> bool:
    """Return whether a step-1 species result should be kept out of step 2."""

    comment = getattr(item_error, "comment", "") or ""
    return (
        comment.startswith("TIMEOUT SPECIES")
        or comment.startswith("FAILED SPECIES")
        or comment.startswith("Failed Species")
    )


def _reduced_species_error_worker_all_conditions_inner(
    result_queue,
    conditions,
    omitted_species,
    standard_times,
    standard_dicts,
    comparator_mode,
    comsol_cores_per_task,
):
    """Run one COMSOL reduced-species candidate in an isolated child process."""

    try:
        _initialize_worker(comsol_cores_per_task)
        time_vals_list, result_dicts = fromcomsolRE.run_reduced_species_models(
            conditions,
            [omitted_species] if isinstance(omitted_species, str) else omitted_species,
        )
    except Exception as exc:
        result_queue.put(("error", str(exc)))
        return

    item_error = condition_objects.ItemError(
        tuple(omitted_species) if isinstance(omitted_species, list) else omitted_species,
        0.0,
        0.0,
        0.0,
    )
    for standard_time, standard_dict, time_vals, result_dict in zip(
        standard_times,
        standard_dicts,
        time_vals_list,
        result_dicts,
    ):
        error_value = _compare_error_value(
            standard_time,
            standard_dict,
            time_vals,
            result_dict,
            comparator_mode,
        )
        item_error.add_to_value(error_value)

    result_queue.put(
        (
            "ok",
            item_error.get_item(),
            item_error.get_value(),
            item_error.get_max_value(),
            item_error.weight,
            item_error.get_comment(),
        )
    )


def _reduced_species_error_worker_all_conditions(args):
    """Run one reduced-species COMSOL mechanism over the full condition set."""

    (
        conditions,
        omitted_species,
        standard_times,
        standard_dicts,
        comparator_mode,
        comsol_cores_per_task,
        timeout_seconds,
    ) = args
    worker_id = _worker_identifier()
    start_worker = timing.perf_counter()

    match model_type:
        case "COMSOL_thermal":
            ctx = multiprocessing.get_context("spawn")
            result_queue = ctx.Queue()
            child = ctx.Process(
                target=_reduced_species_error_worker_all_conditions_inner,
                args=(
                    result_queue,
                    conditions,
                    omitted_species,
                    standard_times,
                    standard_dicts,
                    comparator_mode,
                    comsol_cores_per_task,
                ),
            )
            child.start()
            child.join(timeout_seconds)

            if child.is_alive():
                child.terminate()
                child.join(10)
                if child.is_alive():
                    child.kill()
                    child.join(5)

                timeout_item = tuple(omitted_species) if isinstance(omitted_species, list) else omitted_species
                return (
                    timeout_item,
                    TIMEOUT_ERROR_VALUE,
                    TIMEOUT_ERROR_VALUE,
                    float(len(conditions)),
                    worker_id,
                    timing.perf_counter() - start_worker,
                    True,
                    _timeout_comment(timeout_seconds),
                )

            try:
                status, *payload = result_queue.get_nowait()
            except queue.Empty:
                raise RuntimeError("Reduced-species worker exited without returning a result.") from None

            if status == "error":
                failed_item = tuple(omitted_species) if isinstance(omitted_species, list) else omitted_species
                return (
                    failed_item,
                    TIMEOUT_ERROR_VALUE,
                    TIMEOUT_ERROR_VALUE,
                    float(len(conditions)),
                    worker_id,
                    timing.perf_counter() - start_worker,
                    False,
                    _failed_species_comment(payload[0]),
                )

            species, mean_error, max_error, weight, comment = payload
            return (
                species,
                mean_error,
                max_error,
                weight,
                worker_id,
                timing.perf_counter() - start_worker,
                False,
                comment,
            )
        case _:
            raise ValueError("Grouped reduced-species runs are only used for COMSOL thermal models.")


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
            raise ValueError(f"Unknown comparator mode: {comparator_mode}.")

    # Big arrays die here when function exits
    return omitted_reactions, cond_index, error_value


def _reduced_reactions_error_worker_all_conditions(args):
    """Run one reduced-reaction COMSOL mechanism over the full condition set."""

    (
        conditions,
        omitted_species,
        omitted_reactions,
        standard_times,
        standard_dicts,
        comparator_mode,
    ) = args

    try:
        match model_type:
            case "COMSOL_thermal":
                time_vals_list, result_dicts = fromcomsolRE.run_reduced_reactions_models(
                    conditions,
                    [omitted_species] if isinstance(omitted_species, str) else omitted_species,
                    [omitted_reactions] if isinstance(omitted_reactions, str) else omitted_reactions,
                )
            case _:
                raise ValueError("Grouped reduced-reaction runs are only used for COMSOL thermal models.")
    except Exception as exc:
        raise RuntimeError(str(exc)) from None

    item_error = condition_objects.ItemError(
        omitted_reactions,
        0.0,
        0.0,
        0.0,
    )
    for standard_time, standard_dict, time_vals, result_dict in zip(
        standard_times,
        standard_dicts,
        time_vals_list,
        result_dicts,
    ):
        error_value = _compare_error_value(
            standard_time,
            standard_dict,
            time_vals,
            result_dict,
            comparator_mode,
        )
        item_error.add_to_value(error_value)

    return item_error.get_item(), item_error.get_value(), item_error.get_max_value(), item_error.weight

# Boss of the workers

def run_standard_models_parallel(conditions):
    """
    Runs standard ChemKIN model for all conditions in parallel.
    Prints progress when each simulation finishes.
    If this doesn't run smooth, nothing will.
    """

    total_jobs = len(conditions)

    if model_type == "COMSOL_thermal" and not comsol_parallel_standard_runs:
        print(f"Running {total_jobs} simulations sequentially in one COMSOL worker session")
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
        comparator_mode,
        checkpoint_step_number: int | None = None,
        existing_error_list: condition_objects.ItemErrorList | None = None):
    if model_type == "COMSOL_thermal":
        total_jobs = len(omitted_species_list)
        completed_error_list = condition_objects.ItemErrorList([])
        allowed_keys = {_item_key(species) for species in omitted_species_list}
        if existing_error_list is not None:
            completed_error_list.items = [
                item
                for item in existing_error_list.items
                if _item_key(item.item) in allowed_keys
            ]
        completed_keys = {_item_key(item.item) for item in completed_error_list.items}
        pending_species = [species for species in omitted_species_list if _item_key(species) not in completed_keys]
        pending_jobs = len(pending_species)

        if pending_jobs == 0:
            completed_error_list.sort_items()
            return completed_error_list

        worker_count, comsol_cores_per_task = _pool_configuration(
            pending_jobs,
            default_cap=COMSOL_REDUCTION_WORKER_CAP,
        )

        print(f"Running {pending_jobs} reduced-species simulations "
              f"using {worker_count} worker processes"
              f" with {comsol_cores_per_task} CPU cores per COMSOL task")
        if pending_jobs > worker_count:
            print(f"Scheduling {pending_jobs - worker_count} reduced-species simulations in the queue")
        if completed_error_list.items:
            print(f"Resuming with {len(completed_error_list.items)} completed and {pending_jobs} remaining")

        start_total = timing.time()
        job_args = [
            (
                conditions,
                species,
                standard_data[0],
                standard_data[1],
                comparator_mode,
                comsol_cores_per_task,
                COMSOL_REDUCED_SPECIES_TIMEOUT_SECONDS,
            )
            for species in pending_species
        ]

        errorList = completed_error_list

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_reduced_species_error_worker_all_conditions, args)
                for args in job_args
            ]

            completed = len(errorList.items)

            for future in as_completed(futures):
                species, mean_error, max_error, weight, worker_id, worker_elapsed, timed_out, comment = future.result()
                item_error = condition_objects.ItemError(species, mean_error, max_error, weight, comment=comment)
                errorList.add_to_list(item_error)
                if checkpoint_step_number is not None:
                    errorList.sort_items()
                    project_handler.write_step_progress(checkpoint_step_number, errorList)

                completed += 1
                elapsed = timing.time() - start_total

                if timed_out:
                    print(f"Timeout on simulation without {species}. Cannot afford to remove it")
                elif comment.startswith("FAILED SPECIES") or comment.startswith("Failed Species"):
                    print(f"Solver failure on simulation without {species}. Cannot afford to remove it")

                print(
                    f"[{completed}/{total_jobs}] "
                    f"{species} finished by worker {worker_id} in {worker_elapsed:.2f} s. "
                    f"(Total time elapsed: {elapsed:.2f} s)"
                )

        errorList.sort_items()
        if checkpoint_step_number is not None:
            project_handler.clear_step_progress(checkpoint_step_number)
        print(f"All reduced-species simulations finished in "
              f"{timing.time() - start_total:.2f} seconds")
        return errorList

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
    if existing_error_list is not None:
        for item in existing_error_list.items:
            if _item_key(item.item) not in species_errors:
                continue
            species_errors[_item_key(item.item)] = condition_objects.ItemError(
                item.item,
                item.value,
                item.max_value,
                item.weight,
                comment=getattr(item, "comment", ""),
            )

    job_args = []
    existing_keys = {_item_key(item.item) for item in (existing_error_list.items if existing_error_list is not None else [])}

    for species in omitted_species_list:
        if _item_key(species) in existing_keys:
            continue
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

        completed = int(sum(item.weight for item in species_errors.values() if item.weight > 0))

        for future in as_completed(futures):
            species, cond_index, error_value, worker_id, worker_elapsed = future.result()

            key = tuple(species) if isinstance(species, list) else species
            species_errors[key].add_to_value(error_value)
            if checkpoint_step_number is not None:
                checkpoint_errors = condition_objects.ItemErrorList([
                    item for item in species_errors.values() if item.weight > 0
                ])
                checkpoint_errors.sort_items()
                project_handler.write_step_progress(checkpoint_step_number, checkpoint_errors)

            completed += 1
            elapsed = timing.time() - start_total

            print(
                f"[{completed}/{total_jobs}] "
                f"{species}, Condition {cond_index} finished by worker {worker_id} in {worker_elapsed:.2f} s. "
                f"(Total time elapsed: {elapsed:.2f} s)"
            )

    # Build ItemErrorList
    errorList = condition_objects.ItemErrorList([])

    for species in species_errors:
        if species_errors[species].weight > 0:
            errorList.add_to_list(species_errors[species])

    errorList.sort_items()
    if checkpoint_step_number is not None:
        project_handler.clear_step_progress(checkpoint_step_number)

    print(f"All reduced-species simulations finished in "
          f"{timing.time() - start_total:.2f} seconds")

    return errorList

def run_reduced_reactions_error_parallel(
        conditions,
        final_omitted_species_list,
        omitted_reactions_list,
        standard_data,
        comparator_mode):
    if model_type == "COMSOL_thermal":
        total_jobs = len(omitted_reactions_list)
        worker_count, comsol_cores_per_task = _pool_configuration(total_jobs, default_cap=COMSOL_REDUCTION_WORKER_CAP)

        print(f"Running {total_jobs} reduced-reaction simulations "
              f"using {worker_count} worker processes"
              f" with {comsol_cores_per_task} CPU cores per COMSOL task")
        if total_jobs > worker_count:
            print(f"Scheduling {total_jobs - worker_count} reduced-reaction simulations in the queue")

        start_total = timing.time()
        job_args = [
            (
                conditions,
                final_omitted_species_list,
                reactions,
                standard_data[0],
                standard_data[1],
                comparator_mode,
            )
            for reactions in omitted_reactions_list
        ]

        errorList = condition_objects.ItemErrorList([])

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_worker,
            initargs=(comsol_cores_per_task,),
        ) as executor:
            futures = [
                executor.submit(_reduced_reactions_error_worker_all_conditions, args)
                for args in job_args
            ]

            completed = 0

            for future in as_completed(futures):
                reactions, mean_error, max_error, weight = future.result()
                item_error = condition_objects.ItemError(reactions, mean_error, max_error, weight)
                errorList.add_to_list(item_error)

                completed += 1
                elapsed = timing.time() - start_total

                print(
                    f"[{completed}/{total_jobs}] "
                    f"{reactions} finished "
                    f"(elapsed: {elapsed:.2f} s)"
                )

        errorList.sort_items()
        print(f"All reduced-reaction simulations finished in "
              f"{timing.time() - start_total:.2f} seconds")
        return errorList

    total_jobs = len(conditions) * len(omitted_reactions_list)
    worker_count, comsol_cores_per_task = _pool_configuration(total_jobs, default_cap=COMSOL_REDUCTION_WORKER_CAP)

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
    removable_items = [
        item
        for item in item_error_list.items
        if not _is_nonremovable_species_error(item)
    ]

    # I am not documenting this, figure it out, good luck
    for item in removable_items[::-1]:
        too_much_voodoo.append(item.item)
        inverse_pyramid.append(too_much_voodoo.copy())

    return inverse_pyramid


def _species_group_from_item(item: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize one step-2 item into a plain species list."""

    if isinstance(item, str):
        return [item]
    return list(item)


def _select_step2_omitted_species(
    grouped_species_error: condition_objects.ItemErrorList,
    tolerance: float,
) -> list[str]:
    """Choose the largest removable step-2 species group within tolerance."""

    best_group: list[str] = []
    best_error = float("inf")
    best_max_error = float("inf")

    for item_error in grouped_species_error.items:
        if item_error.get_value() > tolerance:
            continue

        species_group = _species_group_from_item(item_error.get_item())
        if len(species_group) > len(best_group):
            best_group = species_group
            best_error = item_error.get_value()
            best_max_error = item_error.get_max_value()
            continue

        if len(species_group) == len(best_group):
            if item_error.get_value() < best_error:
                best_group = species_group
                best_error = item_error.get_value()
                best_max_error = item_error.get_max_value()
            elif item_error.get_value() == best_error and item_error.get_max_value() < best_max_error:
                best_group = species_group
                best_error = item_error.get_value()
                best_max_error = item_error.get_max_value()

    return best_group


def _export_step2_comsol_copy(grouped_species_error: condition_objects.ItemErrorList) -> None:
    """Export the step-2 reduced COMSOL model copy for the current project."""

    if model_type != "COMSOL_thermal":
        return
    if not isinstance(grouped_species_error, condition_objects.ItemErrorList):
        return
    if not grouped_species_error.items:
        return

    omitted_species = _select_step2_omitted_species(grouped_species_error, sensitivity_tolerance)
    exported_model_path = fromcomsolRE.export_reduced_comsol_model(omitted_species)
    print(
        f"Saved reduced COMSOL model copy with {len(omitted_species)} omitted species: "
        f"{exported_model_path}"
    )

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
            fromcomsolRE.set_comsol_session_mode("stand-alone")
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
            project_handler.write_header(
                model_type,
                comparator_type,
                sensitivity_tolerance,
                excluded_species,
                input_folder,
                all_species,
                all_reactions,
            )
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
        step1_progress = project_handler.get_step_progress(1)
        single_species_error = run_reduced_species_error_parallel(
            conditions_list.conditions,
            list_species,
            standard_data,
            comparator_type,
            checkpoint_step_number=1,
            existing_error_list=step1_progress,
        )

        project_handler.write_step_error(1, single_species_error)


    # --------------- STEP 2 -------------
    if status == "step 2" or status == "step 3" or status == "step 4":
        pass
    else:
        # Group species from the previous error list
        # Most important species are on top, hence the inverse pyramid
        list_grouped_species = inverse_item_pyramid_builder(single_species_error)
        step2_progress = project_handler.get_step_progress(2)
        # step two
        grouped_species_error = run_reduced_species_error_parallel(
            conditions_list.conditions,
            list_grouped_species,
            standard_data,
            comparator_type,
            checkpoint_step_number=2,
            existing_error_list=step2_progress,
        )

        project_handler.write_step_error(2, grouped_species_error)

    _export_step2_comsol_copy(grouped_species_error)
