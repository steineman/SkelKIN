"""Build a neutral, ground-state GRI-Mech 3.0 COMSOL RE model for SkelKIN.

This script does four things:
1. download the official GRI-Mech 3.0 Chemkin files
2. convert them to YAML with Cantera for validation and easy parsing
3. filter out non-neutral / excited-state species (here: ``CH2(S)``)
4. rebuild a COMSOL Reaction Engineering mechanism on top of the existing
   0D template model so SkelKIN can run its step-1 / step-2 reduction workflow

The generated model keeps the original 0D study/plot/runtime structure from the
template while replacing only the Reaction Engineering species, reactions, and
initial-value species list.
"""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess
import sys
import urllib.request

import cantera as ct
import mph


ROOT = Path(__file__).resolve().parent
MECH_DIR = ROOT / "mechanisms" / "gri30"
TEMPLATE_MODEL = ROOT / "0D_260312.mph"
OUTPUT_MODEL = ROOT / "0D_260312_gri30_noN_formula.mph"
DOWNLOADED_YAML = MECH_DIR / "gri30_downloaded.yaml"

OFFICIAL_FILES = {
    "grimech30.dat": "http://combustion.berkeley.edu/gri-mech/version30/files30/grimech30.dat",
    "thermo30.dat": "http://combustion.berkeley.edu/gri-mech/version30/files30/thermo30.dat",
    "transport.dat": "http://combustion.berkeley.edu/gri-mech/version30/files30/transport.dat",
}

# Neutral, ground-state only. GRI 3.0 is already neutral apart from the
# optional ionized variant; the only species to remove here is the excited
# methylene state.
EXCLUDED_SPECIES = {"CH2(S)", "AR"}
THERMO_EXTENSION_TMAX_K = 6000.0
GAS_CONSTANT = 8.31446261815324
DEFAULT_XCH4 = 0.5


def fmt(value: float) -> str:
    """Return a compact float string that stays readable in COMSOL expressions."""

    return f"{float(value):.17g}"


def ensure_downloads() -> None:
    """Download the official GRI files if they are missing."""

    MECH_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in OFFICIAL_FILES.items():
        destination = MECH_DIR / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Using existing download: {destination.name}")
            continue

        print(f"Downloading {filename} from {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())


def ensure_yaml() -> Path:
    """Convert the downloaded Chemkin files to a validated Cantera YAML file."""

    kinetics = MECH_DIR / "grimech30.dat"
    thermo = MECH_DIR / "thermo30.dat"
    transport = MECH_DIR / "transport.dat"

    if DOWNLOADED_YAML.exists() and DOWNLOADED_YAML.stat().st_size > 0:
        print(f"Using existing YAML: {DOWNLOADED_YAML.name}")
        return DOWNLOADED_YAML

    command = [
        sys.executable,
        "-m",
        "cantera.ck2yaml",
        "--input",
        str(kinetics),
        "--thermo",
        str(thermo),
        "--transport",
        str(transport),
        "--output",
        str(DOWNLOADED_YAML),
    ]
    print("Converting Chemkin files to YAML with Cantera")
    subprocess.run(command, check=True)
    return DOWNLOADED_YAML


def canonical_formula(species: ct.Species) -> str:
    """Return a COMSOL-safe chemical formula from Cantera species composition."""

    composition = {
        element: float(amount)
        for element, amount in species.composition.items()
        if abs(float(amount)) > 0.0
    }

    if "C" in composition:
        ordered_elements = ["C"]
        if "H" in composition:
            ordered_elements.append("H")
        ordered_elements.extend(sorted(element for element in composition if element not in {"C", "H"}))
    else:
        ordered_elements = sorted(composition)

    pieces: list[str] = []
    for element in ordered_elements:
        amount = composition[element]
        if abs(amount - round(amount)) < 1e-12:
            count = int(round(amount))
            suffix = "" if count == 1 else str(count)
        else:
            suffix = fmt(amount)
        pieces.append(f"{element}{suffix}")

    return "".join(pieces)


def contains_nitrogen(species: ct.Species) -> bool:
    """Return whether one species contains nitrogen atoms."""

    return abs(float(species.composition.get("N", 0.0))) > 0.0


def species_identifier_map(species_list: list[ct.Species]) -> dict[str, str]:
    """Return the COMSOL species ids used in the rebuilt model.

    Most species are renamed directly to their canonical formula. A small
    number of remaining isomers share the same formula, so those receive a
    formula-first suffix to keep the COMSOL mechanism unambiguous.
    """

    formula_counts: dict[str, int] = {}
    for species in species_list:
        formula = canonical_formula(species)
        formula_counts[formula] = formula_counts.get(formula, 0) + 1

    mapping: dict[str, str] = {}
    for species in species_list:
        formula = canonical_formula(species)
        if formula_counts[formula] == 1:
            mapping[species.name] = formula
        else:
            mapping[species.name] = f"{formula}__{species.name}"
    return mapping


def _format_species_coefficient(amount: float, species_name: str) -> str:
    """Return one stoichiometric species token for a reaction string."""

    if abs(amount - 1.0) < 1e-12:
        return species_name
    if abs(amount - round(amount)) < 1e-12:
        return f"{int(round(amount))} {species_name}"
    return f"{fmt(amount)} {species_name}"


def format_reaction_equation(reaction: ct.Reaction, species_ids: dict[str, str]) -> str:
    """Build a COMSOL reaction equation from mapped species ids."""

    def format_side(side: dict[str, float]) -> str:
        return " + ".join(
            _format_species_coefficient(float(amount), species_ids[species_name])
            for species_name, amount in side.items()
        ) or "0"

    arrow = "<=>" if reaction.reversible else "=>"
    return f"{format_side(reaction.reactants)} {arrow} {format_side(reaction.products)}"


def default_initial_value_expressions(species_names: list[str]) -> list[str]:
    """Return the saved COMSOL initial state parameterized by ``xCH4``."""

    expressions: list[str] = []
    for species_name in species_names:
        if species_name == "CH4":
            expressions.append("xCH4*1[atm]/(R_const*T_g)")
        elif species_name == "CO2":
            expressions.append("(1-xCH4)*1[atm]/(R_const*T_g)")
        else:
            expressions.append("0")
    return expressions


def unit_string(order_power: int) -> str:
    """Return the COMSOL unit string for an Arrhenius pre-exponential factor."""

    if order_power <= 0:
        return "1/s"

    distance_power = 3 * order_power
    if distance_power == 1:
        distance_unit = "m"
    else:
        distance_unit = f"m^{distance_power}"

    if order_power == 1:
        amount_unit = "mol"
    else:
        amount_unit = f"mol^{order_power}"

    return f"{distance_unit}/({amount_unit}*s)"


def nasa_cp_value(coeffs: list[float], temperature: float) -> float:
    a1, a2, a3, a4, a5, _, _ = coeffs
    t = temperature
    return GAS_CONSTANT * (a1 + a2 * t + a3 * t * t + a4 * t**3 + a5 * t**4)


def nasa_h_value(coeffs: list[float], temperature: float) -> float:
    a1, a2, a3, a4, a5, a6, _ = coeffs
    t = temperature
    return GAS_CONSTANT * t * (
        a1
        + a2 * t / 2.0
        + a3 * t * t / 3.0
        + a4 * t**3 / 4.0
        + a5 * t**4 / 5.0
        + a6 / t
    )


def nasa_s_value(coeffs: list[float], temperature: float) -> float:
    a1, a2, a3, a4, a5, _, a7 = coeffs
    t = temperature
    return GAS_CONSTANT * (
        a1 * math.log(t)
        + a2 * t
        + a3 * t * t / 2.0
        + a4 * t**3 / 3.0
        + a5 * t**4 / 4.0
        + a7
    )


def nasa_cp_expression(coeffs: list[float], temperature_symbol: str) -> str:
    a1, a2, a3, a4, a5, _, _ = coeffs
    t = temperature_symbol
    return (
        "R_const*("
        f"({fmt(a1)})"
        f" + ({fmt(a2)})*({t})"
        f" + ({fmt(a3)})*({t})^2"
        f" + ({fmt(a4)})*({t})^3"
        f" + ({fmt(a5)})*({t})^4"
        ")"
    )


def nasa_h_expression(coeffs: list[float], temperature_symbol: str) -> str:
    a1, a2, a3, a4, a5, a6, _ = coeffs
    t = temperature_symbol
    return (
        f"R_const*({t})*("
        f"({fmt(a1)})"
        f" + ({fmt(a2)})*({t})/2"
        f" + ({fmt(a3)})*({t})^2/3"
        f" + ({fmt(a4)})*({t})^3/4"
        f" + ({fmt(a5)})*({t})^4/5"
        f" + ({fmt(a6)})/({t})"
        ")"
    )


def nasa_s_expression(coeffs: list[float], temperature_symbol: str) -> str:
    a1, a2, a3, a4, a5, _, a7 = coeffs
    t = temperature_symbol
    return (
        "R_const*("
        f"({fmt(a1)})*log({t})"
        f" + ({fmt(a2)})*({t})"
        f" + ({fmt(a3)})*({t})^2/2"
        f" + ({fmt(a4)})*({t})^3/3"
        f" + ({fmt(a5)})*({t})^4/4"
        f" + ({fmt(a7)})"
        ")"
    )


def thermo_expressions(species: ct.Species) -> tuple[str, str, str, float, float, float]:
    """Return COMSOL Cp/h/s expressions for one NASA7 species."""

    thermo = species.input_data["thermo"]
    temperature_ranges = list(thermo["temperature-ranges"])
    low_coeffs, high_coeffs = [list(map(float, coeffs)) for coeffs in thermo["data"]]
    t_low, t_mid, t_high = map(float, temperature_ranges)

    t = "(T_var/1[K])"
    cp_low = nasa_cp_expression(low_coeffs, t)
    cp_high = nasa_cp_expression(high_coeffs, t)
    h_low = nasa_h_expression(low_coeffs, t)
    h_high = nasa_h_expression(high_coeffs, t)
    s_low = nasa_s_expression(low_coeffs, t)
    s_high = nasa_s_expression(high_coeffs, t)

    cp_at_high = nasa_cp_value(high_coeffs, t_high)
    h_at_high = nasa_h_value(high_coeffs, t_high)
    s_at_high = nasa_s_value(high_coeffs, t_high)

    cp_expression = (
        f"if(({t})<({fmt(t_high)}), "
        f"if(({t})<({fmt(t_mid)}), {cp_low}, {cp_high}), "
        f"({fmt(cp_at_high)})[J/(mol*K)])"
    )
    h_expression = (
        f"if(({t})<({fmt(t_high)}), "
        f"if(({t})<({fmt(t_mid)}), {h_low}, {h_high}), "
        f"({fmt(h_at_high)})[J/mol] + ({fmt(cp_at_high)})[J/(mol*K)]*((({t})-({fmt(t_high)}))))"
    )
    s_expression = (
        f"if(({t})<({fmt(t_high)}), "
        f"if(({t})<({fmt(t_mid)}), {s_low}, {s_high}), "
        f"({fmt(s_at_high)})[J/(mol*K)] + ({fmt(cp_at_high)})[J/(mol*K)]*log((({t}))/({fmt(t_high)})))"
    )

    return cp_expression, h_expression, s_expression, t_low, t_mid, t_high


def stripped_equation(equation: str) -> str:
    """Remove Chemkin third-body markers from an equation for COMSOL."""

    stripped = equation.replace(" (+M)", "").replace(" + M", "")
    return " ".join(stripped.split())


def arrhenius_expression(
    pre_exponential_si: float,
    temperature_exponent: float,
    activation_energy_si: float,
    order_power: int,
) -> str:
    """Return a COMSOL Arrhenius expression in mol-based SI units."""

    factor = 1000.0 ** (-order_power)
    a_mol = float(pre_exponential_si) * factor
    ea_mol = float(activation_energy_si) / 1000.0
    t = "(T_g/1[K])"
    return (
        f"({fmt(a_mol)})[{unit_string(order_power)}]"
        f"*({t})^({fmt(temperature_exponent)})"
        f"*exp(-({fmt(ea_mol)})[J/mol]/(R_const*T_g))"
    )


def third_body_concentration_expression(
    species_names: list[str],
    species_ids: dict[str, str],
    efficiencies: dict[str, float] | None,
) -> str:
    """Return the effective third-body concentration expression in mol/m^3."""

    terms = [f"re.c_{species_name}" for species_name in species_names]
    expression = " + ".join(terms) if terms else "0"

    for species_name, efficiency in sorted((efficiencies or {}).items()):
        if species_name in EXCLUDED_SPECIES:
            continue
        if species_name not in species_ids:
            continue
        mapped_species_name = species_ids[species_name]
        if mapped_species_name not in species_names:
            continue
        delta = float(efficiency) - 1.0
        if abs(delta) < 1e-12:
            continue
        expression += f" + ({fmt(delta)})*re.c_{mapped_species_name}"

    return f"({expression})"


def forward_rate_expression(
    reaction: ct.Reaction,
    species_names: list[str],
    species_ids: dict[str, str],
) -> str:
    """Build one COMSOL forward-rate expression from a Cantera reaction."""

    base_reactant_order = int(round(sum(float(value) for value in reaction.reactants.values())))
    reaction_type = reaction.input_data.get("type", "")

    if reaction_type == "three-body":
        rate = reaction.rate
        base = arrhenius_expression(
            rate.pre_exponential_factor,
            rate.temperature_exponent,
            rate.activation_energy,
            order_power=base_reactant_order,
        )
        collider = third_body_concentration_expression(
            species_names,
            species_ids,
            reaction.input_data.get("efficiencies"),
        )
        return f"({base})*{collider}"

    if type(reaction.rate).__name__ == "LindemannRate":
        rate = reaction.rate
        kinf = arrhenius_expression(
            rate.high_rate.pre_exponential_factor,
            rate.high_rate.temperature_exponent,
            rate.high_rate.activation_energy,
            order_power=base_reactant_order - 1,
        )
        k0 = arrhenius_expression(
            rate.low_rate.pre_exponential_factor,
            rate.low_rate.temperature_exponent,
            rate.low_rate.activation_energy,
            order_power=base_reactant_order,
        )
        collider = third_body_concentration_expression(
            species_names,
            species_ids,
            reaction.input_data.get("efficiencies"),
        )
        pr = f"(({k0})*{collider}/({kinf}))"
        return f"({kinf})*({pr})/(1 + ({pr}))"

    if type(reaction.rate).__name__ == "TroeRate":
        rate = reaction.rate
        kinf = arrhenius_expression(
            rate.high_rate.pre_exponential_factor,
            rate.high_rate.temperature_exponent,
            rate.high_rate.activation_energy,
            order_power=base_reactant_order - 1,
        )
        k0 = arrhenius_expression(
            rate.low_rate.pre_exponential_factor,
            rate.low_rate.temperature_exponent,
            rate.low_rate.activation_energy,
            order_power=base_reactant_order,
        )
        collider = third_body_concentration_expression(
            species_names,
            species_ids,
            reaction.input_data.get("efficiencies"),
        )
        pr = f"(({k0})*{collider}/({kinf}))"
        pr_safe = f"max(({pr}), 1e-300)"
        t = "(T_g/1[K])"
        troe = reaction.input_data["Troe"]
        fcent_terms = [
            f"(1 - ({fmt(troe['A'])}))*exp(-({t})/({fmt(troe['T3'])}))",
            f"({fmt(troe['A'])})*exp(-({t})/({fmt(troe['T1'])}))",
        ]
        t2_value = float(troe.get("T2", 0.0))
        if t2_value > 0.0:
            fcent_terms.append(f"exp(-({fmt(t2_value)})/({t}))")
        fcent = "(" + " + ".join(fcent_terms) + ")"
        log_fcent = f"log10(max({fcent}, 1e-300))"
        c_term = f"(-0.4 - 0.67*({log_fcent}))"
        n_term = f"(0.75 - 1.27*({log_fcent}))"
        d_term = "0.14"
        log_pr = f"log10({pr_safe})"
        f1 = f"(({log_pr}) + ({c_term}))/(({n_term}) - ({d_term})*(({log_pr}) + ({c_term})))"
        f_term = f"10^(({log_fcent})/(1 + ({f1})^2))"
        return f"({kinf})*({pr})/(1 + ({pr}))*({f_term})"

    # Plain Arrhenius reactions.
    rate = reaction.rate
    return arrhenius_expression(
        rate.pre_exponential_factor,
        rate.temperature_exponent,
        rate.activation_energy,
        order_power=base_reactant_order - 1,
    )


def filtered_mechanism(yaml_path: Path) -> tuple[list[ct.Species], list[ct.Reaction]]:
    """Load the downloaded mechanism and keep only neutral ground-state species."""

    gas = ct.Solution(str(yaml_path))
    species = [
        item
        for item in gas.species()
        if item.name not in EXCLUDED_SPECIES
        and abs(float(getattr(item, "charge", 0.0))) < 1e-12
        and not contains_nitrogen(item)
    ]
    species_names = {item.name for item in species}
    reactions = [
        reaction
        for reaction in gas.reactions()
        if set(reaction.reactants).issubset(species_names)
        and set(reaction.products).issubset(species_names)
    ]
    return species, reactions


def rebuild_model(
    template_model: Path,
    output_model: Path,
    species: list[ct.Species],
    reactions: list[ct.Reaction],
) -> None:
    """Copy the template model and replace its RE mechanism in-place."""

    if output_model.exists():
        output_model.unlink()
    shutil.copy2(template_model, output_model)

    client = mph.start(cores=1)
    model = client.load(str(output_model))
    reaction_engineering = model / "physics/Reaction Engineering"
    species_ids = species_identifier_map(species)

    print("Removing the existing Reaction Engineering mechanism")
    for child in list(reaction_engineering.children()):
        if child.type() in {"SpeciesChem", "ReactionChem"}:
            child.remove()

    print(f"Adding {len(species)} species")
    species_names = [species_ids[item.name] for item in species]
    for sequence_number, species_obj in enumerate(species, start=1):
        cp_expr, h_expr, s_expr, t_low, t_mid, t_high = thermo_expressions(species_obj)
        species_id = species_ids[species_obj.name]
        node = reaction_engineering.create("SpeciesChem")
        node.property("specLabel", species_id)
        node.property("chemicalFormula", canonical_formula(species_obj))
        node.property("sSequenceNo", sequence_number)
        node.property("M", f"{fmt(species_obj.molecular_weight)}[g/mol]")
        node.property("Cp", cp_expr)
        node.property("h", h_expr)
        node.property("s", s_expr)
        node.property("Tlo", f"{fmt(t_low)}[K]")
        node.property("Tmid", f"{fmt(t_mid)}[K]")
        node.property("Thi", f"{fmt(max(t_high, THERMO_EXTENSION_TMAX_K))}[K]")
        node.property("z", "0")

        transport_data = species_obj.input_data.get("transport", {})
        if "diameter" in transport_data:
            node.property("sigma", f"{fmt(float(transport_data['diameter']))}[angstrom]")
        if "well-depth" in transport_data:
            node.property("epsilonkb", f"{fmt(float(transport_data['well-depth']))}[K]")
        if getattr(species_obj, "transport", None) is not None:
            node.property("mu", f"{fmt(float(species_obj.transport.dipole))}[C*m]")

        node.retag(species_id)
        node.rename(f"Species: {species_id}")

    print("Updating the Initial Values species vector")
    initial_values = model / "physics/Reaction Engineering/Initial Values 1"
    model.parameter("xCH4", DEFAULT_XCH4)
    initial_vector = default_initial_value_expressions(species_names)
    initial_values.property("VolumetricSpecies", species_names)
    initial_values.property("initialValue", initial_vector)
    initial_values.property("F0", initial_vector)
    initial_values.property("T0", "T_g")

    print(f"Adding {len(reactions)} reactions")
    for sequence_number, reaction in enumerate(reactions, start=1):
        node = reaction_engineering.create("ReactionChem")
        node.property("kf", forward_rate_expression(reaction, species_names, species_ids))
        node.property("rSequenceNo", sequence_number)
        node.property("formula", format_reaction_equation(reaction, species_ids))

    print(f"Saving {output_model.name}")
    model.save()


def validate_output(model_path: Path) -> None:
    """Check the saved COMSOL model through SkelKIN's own parser."""

    sys.path.insert(0, str(ROOT))
    import fromcomsolRE  # pylint: disable=import-outside-toplevel

    fromcomsolRE.load_comsol_model(str(model_path))
    species = fromcomsolRE.get_species()
    reactions = fromcomsolRE.get_reactions()

    print(
        f"Validated COMSOL model: {len(species)} species, "
        f"{len(reactions)} reactions"
    )
    if "CH2(S)" in species:
        raise RuntimeError("Filtered COMSOL model still contains CH2(S).")


def main() -> None:
    print("Preparing official GRI-Mech 3.0 files")
    ensure_downloads()
    yaml_path = ensure_yaml()
    species, reactions = filtered_mechanism(yaml_path)

    print(
        "Filtered mechanism keeps "
        f"{len(species)} species and {len(reactions)} reactions "
        "(neutral ground-state subset)"
    )

    rebuild_model(TEMPLATE_MODEL, OUTPUT_MODEL, species, reactions)
    validate_output(OUTPUT_MODEL)


if __name__ == "__main__":
    main()
