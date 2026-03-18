import os
import re
import json
import condition_objects

project_identifier = "dummy"


# The project handler should write a few things down.
# It should write down the modelled conditions
# The four (4) major steps and the sorted order of these steps with errors
# 1. individual species omitted
# 2. grouped species omitted
# 3. individual reactions omitted
# 4. grouped reactions omitted
# FINAL: reduced species and reactions


def _step_block_tags(step_number: int, *, progress: bool = False) -> tuple[str, str]:
    """Return the start/end tags for one step block."""

    if step_number not in {1, 2, 3, 4}:
        raise ValueError("step_number must be 1, 2, 3, 4")
    if progress:
        return f"STEP {step_number} PROGRESS", f"END STEP {step_number} PROGRESS"
    return f"STEP {step_number} ERROR", f"END STEP {step_number} ERROR"


def _step_header_line(step_number: int) -> str:
    """Return the column header for one step block."""

    header_map = {
        1: "SPECIES ERROR MAX_ERROR WEIGHT",
        2: "SPECIES_LIST ERROR MAX_ERROR WEIGHT",
        3: "REACTIONS ERROR MAX_ERROR WEIGHT",
        4: "REACTION_LIST ERROR MAX_ERROR WEIGHT",
    }
    return header_map[step_number]


def _replace_or_append_block(filepath: str, start_tag: str, end_tag: str, text_str: str) -> None:
    """Replace an existing tagged block or append it if missing."""

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if start_tag in raw_text:
            updated_text = re.sub(
                rf"{start_tag}.*?{end_tag}\n\n",
                text_str,
                raw_text,
                flags=re.DOTALL,
            )
        else:
            updated_text = raw_text + text_str

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(updated_text)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_str)


def _remove_block(filepath: str, start_tag: str, end_tag: str) -> None:
    """Remove a tagged block if it exists."""

    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if start_tag not in raw_text:
        return

    updated_text = re.sub(
        rf"{start_tag}.*?{end_tag}\n\n",
        "",
        raw_text,
        flags=re.DOTALL,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(updated_text)

def write_header(
    model_type: str,
    comparator_type: str,
    sensitivity: float,
    excluded_species: list[str],
    test_conditions_folder: str,
    original_species_list: list[str],
    original_reactions_list: list[str],
):
    text_str = ""
    text_str += f"HEADER {project_identifier}\n"
    text_str += f"Model type: {model_type}\n"
    text_str += f"Comparator type: {comparator_type}\n"
    text_str += f"Sensitivity: {str(sensitivity)}\n"
    text_str += f"Excluded species: {str(excluded_species)}\n"
    text_str += f"Test conditions folder: {str(test_conditions_folder)}\n"
    text_str += f"Original amount of species: {str(len(original_species_list))}\n"
    text_str += f"Original amount of reactions: {str(len(original_reactions_list))}\n"
    text_str += f"END HEADER\n\n"
    if os.path.exists(project_identifier + '.skn'):
        # it exists add or modify
        with open(project_identifier + '.skn', "r") as f:
            raw_text = f.read()
        if "HEADER" in raw_text:
            # change the text
            update_header = re.sub(r'HEADER.*?END HEADER\n\n', text_str, raw_text, flags=re.DOTALL)
            with open(project_identifier + '.skn', "w", encoding="utf-8") as f:
                f.write(update_header)
        else:
            with open(project_identifier + ".skn", "a") as f:
                f.write(raw_text)
    else:
        with open(project_identifier + ".skn", "w") as skn_file:
            skn_file.write(text_str)


def write_step_error(step_number: int, item_errors: condition_objects.ItemErrorList):
    start_tag, end_tag = _step_block_tags(step_number, progress=False)
    column_header = _step_header_line(step_number)

    text_str = ""
    text_str += f"{start_tag}\n"
    text_str += f"{column_header}\n"

    for error in item_errors.items:
        item_serialized = json.dumps(error.item)
        line = f"{item_serialized} {error.value} {error.max_value} {error.weight}"
        if getattr(error, "comment", ""):
            line += f" {error.comment}"
        text_str += line + "\n"

    text_str += f"{end_tag}\n\n"

    filepath = project_identifier + ".skn"
    _replace_or_append_block(filepath, start_tag, end_tag, text_str)
    clear_step_progress(step_number)


def write_step_progress(step_number: int, item_errors: condition_objects.ItemErrorList):
    """Checkpoint one in-progress step without marking it complete."""

    start_tag, end_tag = _step_block_tags(step_number, progress=True)
    column_header = _step_header_line(step_number)

    text_str = ""
    text_str += f"{start_tag}\n"
    text_str += f"{column_header}\n"

    for error in item_errors.items:
        item_serialized = json.dumps(error.item)
        line = f"{item_serialized} {error.value} {error.max_value} {error.weight}"
        if getattr(error, "comment", ""):
            line += f" {error.comment}"
        text_str += line + "\n"

    text_str += f"{end_tag}\n\n"
    _replace_or_append_block(project_identifier + ".skn", start_tag, end_tag, text_str)


def clear_step_progress(step_number: int):
    """Remove any saved in-progress checkpoint for one step."""

    start_tag, end_tag = _step_block_tags(step_number, progress=True)
    _remove_block(project_identifier + ".skn", start_tag, end_tag)


def write_verdict(species_list : list[str], reaction_list : list[str]):
    text_str = ""
    text_str += f"VERDICT\n"
    text_str += f"SPECIES KEPT\n"
    for species in species_list:
        text_str += f"{str(species)}\n"
    text_str += f"REACTIONS KEPT\n"
    for reaction in reaction_list:
        text_str += f"{str(reaction)}\n"
    text_str += f"END VERDICT\n\n"

    if os.path.exists(project_identifier + '.skn'):
        with open(project_identifier + '.skn', "r") as f:
            raw_text = f.read()

        if "VERDICT" in raw_text:
            update_verdict = re.sub(r'VERDICT.*?END VERDICT\n\n', text_str, raw_text, flags=re.DOTALL)
            with open(project_identifier + '.skn', "w", encoding="utf-8") as f:
                f.write(update_verdict)
        else:
            with open(project_identifier + ".skn", "a") as f:
                f.write(text_str)


def where_are_we() -> str:
    if not os.path.exists(project_identifier + '.skn'):
        return "nothing"
    with open(project_identifier + '.skn', "r") as f:
        raw_text = f.read()
    if "VERDICT" in raw_text:
        return "done"
    elif "END STEP 4 ERROR" in raw_text:
        return "step 4"
    elif "END STEP 3 ERROR" in raw_text:
        return "step 3"
    elif "END STEP 2 ERROR" in raw_text:
        return "step 2"
    elif "END STEP 1 ERROR" in raw_text:
        return "step 1"
    elif "END HEADER" in raw_text:
        return "initialized"
    else:
        return "nothing"


def _read_step_block(start_tag: str, end_tag: str) -> condition_objects.ItemErrorList:
    """Read one tagged step block from the project file."""

    if not os.path.exists(project_identifier + '.skn'):
        raise FileNotFoundError("No project file found.")

    with open(project_identifier + '.skn', "r", encoding="utf-8") as f:
        raw_text = f.read()

    pattern = rf"{start_tag}\n.*?\n(.*?){end_tag}"
    match = re.search(pattern, raw_text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No data found for block {start_tag}.")

    block = match.group(1).strip().splitlines()

    item_error_list = condition_objects.ItemErrorList([])

    for line in block:
        match = re.match(
            r"^(.*) "
            r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan)) "
            r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan)) "
            r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan))(?: (.*))?$",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        item = json.loads(match.group(1))
        value = float(match.group(2))
        max_value = float(match.group(3))
        weight = float(match.group(4))
        comment = match.group(5) or ""

        error_obj = condition_objects.ItemError(
            item=item,
            value=value,
            max_value=max_value,
            weight=weight,
            comment=comment
        )

        item_error_list.items.append(error_obj)

    return item_error_list


def get_me_data(step: str) -> condition_objects.ItemErrorList:
    step_map = {
        "step 1": _step_block_tags(1, progress=False),
        "step 2": _step_block_tags(2, progress=False),
        "step 3": _step_block_tags(3, progress=False),
        "step 4": _step_block_tags(4, progress=False),
    }

    if step not in step_map:
        raise ValueError("Step must be 'step 1', 'step 2', 'step 3', or 'step 4'.")

    start_tag, end_tag = step_map[step]
    return _read_step_block(start_tag, end_tag)


def get_step_progress(step_number: int) -> condition_objects.ItemErrorList:
    """Return the in-progress checkpoint for one step, if any."""

    start_tag, end_tag = _step_block_tags(step_number, progress=True)
    try:
        return _read_step_block(start_tag, end_tag)
    except (FileNotFoundError, ValueError):
        return condition_objects.ItemErrorList([])

def get_header_data():
    if not os.path.exists(project_identifier + '.skn'):
        raise FileNotFoundError("No project file found.")

    start_tag = "HEADER"
    end_tag = "END HEADER"

    with open(project_identifier + '.skn', "r", encoding="utf-8") as f:
        raw_text = f.read()

    pattern = rf"{start_tag}(.*?){end_tag}"
    match = re.search(pattern, raw_text, flags=re.DOTALL)

    if not match:
        raise ValueError("No HEADER section found.")

    lines = match.group(1).strip().splitlines()

    model_type = None
    sensitivity = None
    comparator_type = None
    excluded_species = []
    test_conditions_folder = None
    len_original_species_list = 0
    len_original_reactions_list = 0

    for line in lines:
        if line.startswith("Model type:"):
            model_type = line.split(":", 1)[1].strip()

        elif line.startswith("Comparator type:"):
            comparator_type = line.split(":", 1)[1].strip()

        elif line.startswith("Sensitivity:"):
            sensitivity = float(line.split(":", 1)[1].strip())

        elif line.startswith("Excluded species:"):
            excluded_species = eval(line.split(":", 1)[1].strip())

        elif line.startswith("Test conditions folder:"):
            test_conditions_folder = line.split(":", 1)[1].strip()

        elif line.startswith("Original amount of species:"):
            len_original_species_list = int(line.split(":", 1)[1].strip())

        elif line.startswith("Original amount of reactions:"):
            len_original_reactions_list = int(line.split(":", 1)[1].strip())

    return (
        model_type,
        comparator_type,
        sensitivity,
        excluded_species,
        test_conditions_folder,
        len_original_species_list,
        len_original_reactions_list,
    )


def get_verdict_data():
    if not os.path.exists(project_identifier + '.skn'):
        raise FileNotFoundError("No project file found.")

    start_tag = "VERDICT"
    end_tag = "END VERDICT"

    with open(project_identifier + '.skn', "r", encoding="utf-8") as f:
        raw_text = f.read()

    pattern = rf"{start_tag}\n(.*?){end_tag}"
    match = re.search(pattern, raw_text, flags=re.DOTALL)

    if not match:
        raise ValueError("No VERDICT section found.")

    lines = match.group(1).strip().splitlines()

    species_list = []
    reaction_list = []

    mode = None

    for line in lines:
        line = line.strip()

        if line == "SPECIES KEPT":
            mode = "species"
            continue

        elif line == "REACTIONS KEPT":
            mode = "reactions"
            continue

        if mode == "species":
            species_list.append(line)

        elif mode == "reactions":
            reaction_list.append(line)

    return species_list, reaction_list
