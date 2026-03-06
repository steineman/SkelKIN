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

def write_header(model_type: str, comparator_type: str, sensitivity: float, excluded_species: list[str], test_conditions_folder: str, original_species_list: list[str], original_reactions_list: list[str]):
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
    if step_number not in {1, 2, 3, 4}:
        raise ValueError("step_number must be 1, 2, 3, 4")

    header_map = {
        1: "SPECIES ERROR MAX_ERROR WEIGHT",
        2: "SPECIES_LIST ERROR MAX_ERROR WEIGHT",
        3: "REACTIONS ERROR MAX_ERROR WEIGHT",
        4: "REACTION_LIST ERROR MAX_ERROR WEIGHT",
    }

    start_tag = f"STEP {step_number} ERROR"
    end_tag = f"END STEP {step_number} ERROR"
    column_header = header_map[step_number]

    text_str = ""
    text_str += f"{start_tag}\n"
    text_str += f"{column_header}\n"

    for error in item_errors.items:
        item_serialized = json.dumps(error.item)
        text_str += f"{item_serialized} {error.value} {error.max_value} {error.weight}\n"

    text_str += f"{end_tag}\n\n"

    filepath = project_identifier + ".skn"

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        if start_tag in raw_text:
            updated_text = re.sub(
                rf"{start_tag}.*?{end_tag}\n\n",
                text_str,
                raw_text,
                flags=re.DOTALL
            )
        else:
            updated_text = raw_text + text_str

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(updated_text)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_str)


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
    elif "END STEP 4" in raw_text:
        return "step 4"
    elif "END STEP 3" in raw_text:
        return "step 3"
    elif "END STEP 2" in raw_text:
        return "step 2"
    elif "END STEP 1" in raw_text:
        return "step 1"
    elif "END HEADER" in raw_text:
        return "initialized"
    else:
        return "nothing"


def get_me_data(step: str) -> condition_objects.ItemErrorList:
    if not os.path.exists(project_identifier + '.skn'):
        raise FileNotFoundError("No project file found.")

    step_map = {
        "step 1": ("STEP 1 ERROR", "END STEP 1 ERROR"),
        "step 2": ("STEP 2 ERROR", "END STEP 2 ERROR"),
        "step 3": ("STEP 3 ERROR", "END STEP 3 ERROR"),
        "step 4": ("STEP 4 ERROR", "END STEP 4 ERROR"),
    }

    if step not in step_map:
        raise ValueError("Step must be 'step 1', 'step 2', 'step 3', or 'step 4'.")

    start_tag, end_tag = step_map[step]

    with open(project_identifier + '.skn', "r", encoding="utf-8") as f:
        raw_text = f.read()

    pattern = rf"{start_tag}\n.*?\n(.*?){end_tag}"
    match = re.search(pattern, raw_text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No data found for {step}.")

    block = match.group(1).strip().splitlines()

    item_error_list = condition_objects.ItemErrorList([])

    for line in block:
        parts = line.strip().split(" ", 3)
        if len(parts) != 4:
            continue

        item = json.loads(parts[0])
        value = float(parts[1])
        max_value = float(parts[2])
        weight = float(parts[3])

        error_obj = condition_objects.ItemError(
            item=item,
            value=value,
            max_value=max_value,
            weight=weight
        )

        item_error_list.items.append(error_obj)

    return item_error_list

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