"""Build a truly reduced COMSOL GRI30 model from a saved ``.skn`` file.

This script reads the edited SkelKIN project file, picks the best step-2
omission group within the saved sensitivity tolerance, and rebuilds the COMSOL
Reaction Engineering mechanism so that:

1. omitted species are removed from the RE interface
2. reactions touching omitted species are removed
3. forward-rate expressions are regenerated without obsolete ``re.c_*`` terms
4. the saved project file gets a clear verdict section listing what was kept

It is intentionally separate from ``fromcomsolRE.py`` because the live solver
workflow still relies on placeholder zero-valued species for reduced sweeps.
For an exported standalone COMSOL model, rebuilding the mechanism is cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import shutil
import xml.etree.ElementTree as ET
import zipfile

import cantera as ct
import mph

from build_gri30_comsol_model import (
    THERMO_EXTENSION_TMAX_K,
    contains_nitrogen,
    default_initial_value_expressions,
    fmt,
    format_reaction_equation,
    forward_rate_expression,
    species_identifier_map,
    thermo_expressions,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_FILE = ROOT / "COMSOL_260312_gri30_noN_formula.skn"
DEFAULT_TEMPLATE_MODEL = ROOT / "0D_260312_gri30_noN_formula.mph"
DEFAULT_OUTPUT_MODEL = ROOT / "0D_260312_gri30_noN_formula_reduced.mph"
MANDATORY_EXCLUSIONS = {"CH2(S)", "AR"}
PREFERRED_PLOT_SPECIES = ["CH4", "CO2", "H2", "CO", "O", "O2", "H", "H2O", "HO", "CH2O"]


@dataclass
class Step2Entry:
    omitted_species: list[str]
    error: float
    max_error: float
    weight: float


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _project_sensitivity(project_file: Path) -> float:
    for line in _read_text(project_file).splitlines():
        if line.startswith("Sensitivity:"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not find sensitivity in {project_file}")


def _step2_entries(project_file: Path) -> list[Step2Entry]:
    lines = _read_text(project_file).splitlines()
    inside_step2 = False
    entries: list[Step2Entry] = []

    for raw_line in lines:
        line = raw_line.strip()
        if line == "STEP 2 ERROR":
            inside_step2 = True
            continue
        if line == "END STEP 2 ERROR":
            break
        if not inside_step2 or not line or line.startswith("SPECIES_LIST"):
            continue
        if not line.startswith("["):
            continue

        list_end = line.rfind("]")
        if list_end < 0:
            continue
        group = ast.literal_eval(line[: list_end + 1])
        numbers = line[list_end + 1 :].strip().split()
        if len(numbers) < 3:
            continue
        entries.append(
            Step2Entry(
                omitted_species=list(group),
                error=float(numbers[0]),
                max_error=float(numbers[1]),
                weight=float(numbers[2]),
            )
        )

    if not entries:
        raise ValueError(f"No step-2 entries found in {project_file}")
    return entries


def select_omitted_species(project_file: Path) -> list[str]:
    """Return the largest valid omission group still present in the project file."""

    tolerance = _project_sensitivity(project_file)
    best_group: list[str] = []
    best_error = float("inf")
    best_max_error = float("inf")

    for entry in _step2_entries(project_file):
        if entry.error > tolerance:
            continue
        group = list(entry.omitted_species)
        if len(group) > len(best_group):
            best_group = group
            best_error = entry.error
            best_max_error = entry.max_error
            continue
        if len(group) == len(best_group):
            if entry.error < best_error:
                best_group = group
                best_error = entry.error
                best_max_error = entry.max_error
            elif entry.error == best_error and entry.max_error < best_max_error:
                best_group = group
                best_error = entry.error
                best_max_error = entry.max_error

    if not best_group:
        raise ValueError(
            f"No step-2 omission group in {project_file.name} satisfies the saved sensitivity tolerance."
        )
    return best_group


def filtered_gri30_species() -> list[ct.Species]:
    """Return the nitrogen-free, neutral, ground-state GRI30 species list."""

    gas = ct.Solution("gri30.yaml")
    return [
        species
        for species in gas.species()
        if species.name not in MANDATORY_EXCLUSIONS
        and abs(float(getattr(species, "charge", 0.0))) < 1e-12
        and not contains_nitrogen(species)
    ]


def reduced_mechanism_from_omissions(
    omitted_species_ids: list[str],
) -> tuple[list[ct.Species], list[ct.Reaction], dict[str, str]]:
    """Return kept species/reactions after removing the requested COMSOL ids."""

    gas = ct.Solution("gri30.yaml")
    base_species = filtered_gri30_species()
    species_ids = species_identifier_map(base_species)
    valid_ids = set(species_ids.values())
    unknown_ids = sorted(set(omitted_species_ids) - valid_ids)
    if unknown_ids:
        raise ValueError(f"Unknown species ids in omission list: {unknown_ids}")

    kept_species = [species for species in base_species if species_ids[species.name] not in omitted_species_ids]
    kept_names = {species.name for species in kept_species}
    kept_reactions = [
        reaction
        for reaction in gas.reactions()
        if set(reaction.reactants).issubset(kept_names) and set(reaction.products).issubset(kept_names)
    ]
    kept_species_ids = {species.name: species_ids[species.name] for species in kept_species}
    return kept_species, kept_reactions, kept_species_ids


def choose_plot_species(kept_species_names: list[str]) -> list[str]:
    """Return a compact, human-readable concentration plot selection."""

    kept_set = set(kept_species_names)
    chosen = [species for species in PREFERRED_PLOT_SPECIES if species in kept_set]
    if len(chosen) < min(10, len(kept_species_names)):
        for species in kept_species_names:
            if species not in chosen:
                chosen.append(species)
            if len(chosen) >= min(10, len(kept_species_names)):
                break
    return chosen


def _configure_plot(model, kept_species_names: list[str]) -> None:
    """Update the saved concentration plot to reference only kept species."""

    plot_species = choose_plot_species(kept_species_names)
    expressions = [f"re.c_{species_name}" for species_name in plot_species]
    global_plot = model / "plots/Concentration (re)/Global 1"
    global_plot.property("expr", expressions)
    global_plot.property("unit", ["mol/m^3"] * len(expressions))
    global_plot.property("descr", ["Concentration"] * len(expressions))
    global_plot.property("legends", plot_species)
    global_plot.property("autolegends", plot_species)
    global_plot.property("linecount", len(expressions))


def _encode_comsol_string_list(values: list[str]) -> str:
    """Return COMSOL's compact list encoding for string vectors."""

    encoded_items = ",".join(f"'{value}'" for value in values)
    return f"1|{len(values)},{encoded_items}"


def _patch_saved_plot(model_path: Path, kept_species_names: list[str]) -> None:
    """Patch the saved plot expressions directly in ``dmodel.xml``.

    COMSOL eagerly evaluates plot expressions when they are changed through the
    live API, which is fragile right after a mechanism rebuild. Patching the
    archive avoids that mid-edit validation while still leaving a clear saved
    plot in the exported model.
    """

    plot_species = choose_plot_species(kept_species_names)
    expressions = [f"re.c_{species_name}" for species_name in plot_species]

    with zipfile.ZipFile(model_path, "r") as source_archive:
        dmodel = ET.fromstring(source_archive.read("dmodel.xml"))

        feature = dmodel.find(".//ResultFeature[@tag='glob1']")
        if feature is None:
            raise RuntimeError(f"Could not find saved concentration plot in {model_path.name}")

        properties = {
            prop.attrib.get("name"): prop
            for prop in feature.findall("propertyValue")
            if prop.attrib.get("name")
        }

        def set_list_property(name: str, values: list[str]) -> None:
            prop = properties.get(name)
            if prop is None:
                raise RuntimeError(f"Missing plot property {name!r} in {model_path.name}")
            prop.attrib["valueMatrix"] = _encode_comsol_string_list(values)
            prop.attrib.pop("value", None)

        def set_scalar_property(name: str, value: str) -> None:
            prop = properties.get(name)
            if prop is None:
                raise RuntimeError(f"Missing plot property {name!r} in {model_path.name}")
            prop.attrib["value"] = value
            prop.attrib.pop("valueMatrix", None)

        set_list_property("p:expr", expressions)
        set_list_property("p:unit", ["mol/m^3"] * len(expressions))
        set_list_property("p:descr", ["Concentration"] * len(expressions))
        set_list_property("p:legends", plot_species)
        set_list_property("p:autolegends", plot_species)
        set_scalar_property("p:linecount", str(len(expressions)))

        updated_dmodel = ET.tostring(dmodel, encoding="utf-8", xml_declaration=True)
        temp_model = model_path.with_name(f"{model_path.stem}.tmp{model_path.suffix}")
        with zipfile.ZipFile(temp_model, "w") as destination_archive:
            for info in source_archive.infolist():
                payload = updated_dmodel if info.filename == "dmodel.xml" else source_archive.read(info.filename)
                destination_archive.writestr(info, payload)

    temp_model.replace(model_path)


def rebuild_reduced_model(
    template_model: Path,
    output_model: Path,
    kept_species: list[ct.Species],
    kept_reactions: list[ct.Reaction],
    kept_species_ids: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Rebuild the COMSOL RE mechanism with only the kept entries."""

    if output_model.exists():
        output_model.unlink()
    shutil.copy2(template_model, output_model)

    kept_species_names = [kept_species_ids[species.name] for species in kept_species]
    client = mph.start(cores=1)
    model = client.load(str(output_model))

    try:
        reaction_engineering = model / "physics/Reaction Engineering"

        for child in list(reaction_engineering.children()):
            if child.type() in {"SpeciesChem", "ReactionChem"}:
                child.remove()

        for sequence_number, species in enumerate(kept_species, start=1):
            cp_expr, h_expr, s_expr, t_low, t_mid, t_high = thermo_expressions(species)
            species_id = kept_species_ids[species.name]
            node = reaction_engineering.create("SpeciesChem")
            node.property("specLabel", species_id)
            node.property("chemicalFormula", species_id.split("__", 1)[0])
            node.property("sSequenceNo", sequence_number)
            node.property("M", f"{fmt(species.molecular_weight)}[g/mol]")
            node.property("Cp", cp_expr)
            node.property("h", h_expr)
            node.property("s", s_expr)
            node.property("Tlo", f"{fmt(t_low)}[K]")
            node.property("Tmid", f"{fmt(t_mid)}[K]")
            node.property("Thi", f"{fmt(max(t_high, THERMO_EXTENSION_TMAX_K))}[K]")
            node.property("z", "0")

            transport_data = species.input_data.get("transport", {})
            if "diameter" in transport_data:
                node.property("sigma", f"{fmt(float(transport_data['diameter']))}[angstrom]")
            if "well-depth" in transport_data:
                node.property("epsilonkb", f"{fmt(float(transport_data['well-depth']))}[K]")
            if getattr(species, "transport", None) is not None:
                node.property("mu", f"{fmt(float(species.transport.dipole))}[C*m]")

            node.retag(species_id)
            node.rename(f"Species: {species_id}")

        initial_values = model / "physics/Reaction Engineering/Initial Values 1"
        initial_vector = default_initial_value_expressions(kept_species_names)
        initial_values.property("VolumetricSpecies", kept_species_names)
        initial_values.property("initialValue", initial_vector)
        initial_values.property("F0", initial_vector)
        initial_values.property("T0", "T_g")

        reaction_equations: list[str] = []
        for sequence_number, reaction in enumerate(kept_reactions, start=1):
            node = reaction_engineering.create("ReactionChem")
            equation = format_reaction_equation(reaction, kept_species_ids)
            node.property("kf", forward_rate_expression(reaction, kept_species_names, kept_species_ids))
            node.property("rSequenceNo", sequence_number)
            node.property("formula", equation)
            reaction_equations.append(equation)

        model.save()

        model.save()
    finally:
        try:
            client.remove(model)
        except Exception:
            pass

    _patch_saved_plot(output_model, kept_species_names)
    return kept_species_names, reaction_equations


def write_verdict(project_file: Path, kept_species: list[str], kept_reactions: list[str]) -> None:
    """Replace or append a clear verdict block in the project file."""

    verdict_lines = ["VERDICT", "SPECIES KEPT", *kept_species, "REACTIONS KEPT", *kept_reactions, "END VERDICT", ""]
    verdict_text = "\n".join(verdict_lines)
    raw_text = _read_text(project_file)

    if "VERDICT" in raw_text and "END VERDICT" in raw_text:
        start = raw_text.index("VERDICT")
        end = raw_text.index("END VERDICT") + len("END VERDICT")
        updated = raw_text[:start] + verdict_text + raw_text[end:]
    else:
        if not raw_text.endswith("\n"):
            raw_text += "\n"
        updated = raw_text + "\n" + verdict_text
    project_file.write_text(updated, encoding="utf-8")


def main() -> None:
    project_file = DEFAULT_PROJECT_FILE
    template_model = DEFAULT_TEMPLATE_MODEL
    output_model = DEFAULT_OUTPUT_MODEL

    omitted_species_ids = select_omitted_species(project_file)
    kept_species, kept_reactions, kept_species_ids = reduced_mechanism_from_omissions(omitted_species_ids)
    kept_species_names, reaction_equations = rebuild_reduced_model(
        template_model=template_model,
        output_model=output_model,
        kept_species=kept_species,
        kept_reactions=kept_reactions,
        kept_species_ids=kept_species_ids,
    )
    write_verdict(project_file, kept_species_names, reaction_equations)

    print(f"Project file: {project_file.name}")
    print(f"Omitted species ({len(omitted_species_ids)}): {omitted_species_ids}")
    print(f"Kept species ({len(kept_species_names)}): {kept_species_names}")
    print(f"Kept reactions: {len(reaction_equations)}")
    print(f"Saved reduced model: {output_model}")


if __name__ == "__main__":
    main()
