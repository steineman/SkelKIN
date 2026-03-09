"""COMSOL Reaction Engineering backend for SkelKIN.

This module mirrors the public API of ``fromchemKIN.py`` but uses a COMSOL
Reaction Engineering ``.mph`` model as the mechanism source.

High-level flow:
1. Read ``dmodel.xml`` from the ``.mph`` zip archive.
2. Parse model parameters, analytic functions, species thermochemistry, and
   reaction-rate expressions.
3. Build a reduced mechanism view by removing species and reactions on demand.
4. Integrate the resulting homogeneous kinetics problem for the supplied
   temperature and pressure history.

The solver is intentionally conservative: it uses stiff implicit methods,
integrates each profile segment independently across imposed jumps, and rescales
state variables to improve numerical conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import math
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import condition_objects as cond

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False


relative_tolerance = 1e-3
absolute_tolerance = 1e-5
solver_methods = ("BDF", "LSODA", "Radau")
scaled_state_upper_bound = 2.0

AVOGADRO = 6.02214076e23
GAS_CONSTANT = 8.31446261815324
ONE_ATM = 101325.0

UNIT_ENV = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "s": 1.0,
    "mol": 1.0,
    "K": 1.0,
    "Pa": 1.0,
    "atm": ONE_ATM,
    "J": 1.0,
    "kg": 1.0,
    "C": 1.0,
    "angstrom": 1e-10,
}

_MODULE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _MODULE_ROOT / "chemistry" / "260306" / "0D_260306b.mph"
_model_file = _DEFAULT_MODEL
_cached_model: "ComsolModel | None" = None


def _unit_factor(unit_expression: str) -> float:
    cleaned = unit_expression.replace("^", "**").replace(" ", "")
    return float(eval(cleaned, {"__builtins__": {}}, UNIT_ENV))


def _replace_units(expr: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"*({_unit_factor(match.group(1))})"

    return re.sub(r"\[([^\]]+)\]", repl, expr)


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unmatched parenthesis in expression: {text}")


def _split_top_level_arguments(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args


def _replace_if_expressions(expr: str) -> str:
    result: list[str] = []
    index = 0
    while True:
        marker = expr.find("if(", index)
        if marker == -1:
            result.append(expr[index:])
            break
        result.append(expr[index:marker])
        close_index = _find_matching_paren(expr, marker + 2)
        inner = expr[marker + 3:close_index]
        args = _split_top_level_arguments(inner)
        if len(args) != 3:
            raise ValueError(f"COMSOL if-expression should have 3 arguments: {expr}")
        condition = _replace_if_expressions(args[0])
        on_true = _replace_if_expressions(args[1])
        on_false = _replace_if_expressions(args[2])
        result.append(f"(({on_true}) if ({condition}) else ({on_false}))")
        index = close_index + 1
    return "".join(result)


def _to_python_expression(expr: str) -> str:
    cleaned = unescape(expr)
    cleaned = cleaned.replace("\\,", ",")
    cleaned = _replace_units(cleaned)
    cleaned = cleaned.replace("^", "**")
    cleaned = _replace_if_expressions(cleaned)
    return cleaned


def _decode_comsol_value(value: str) -> str | list[str]:
    strings = re.findall(r"'([^']*)'", value)
    if not strings:
        return value
    if len(strings) == 1:
        return unescape(strings[0])
    return [unescape(item) for item in strings]


def _property_raw_value(node: ET.Element) -> str:
    return node.attrib.get("value", node.attrib.get("valueMatrix", ""))


def _parse_side(side: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (piece.strip() for piece in side.split("+")):
        if not part:
            continue
        match = re.match(r"^([0-9]*\.?[0-9]+)\s*(.+)$", part)
        if match:
            coefficient = float(match.group(1))
            species = match.group(2).strip()
        else:
            coefficient = 1.0
            species = part
        out[species] = out.get(species, 0.0) + coefficient
    return out


def _parse_equation(formula: str) -> tuple[dict[str, float], dict[str, float]]:
    equation = formula.strip()
    if "<=>" in equation:
        left, right = equation.split("<=>", 1)
    elif "=>" in equation:
        left, right = equation.split("=>", 1)
    elif "=" in equation:
        left, right = equation.split("=", 1)
    else:
        raise ValueError(f"Unsupported reaction formula: {formula}")
    return _parse_side(left), _parse_side(right)


@dataclass
class CompiledExpression:
    """A COMSOL expression translated to Python and precompiled once."""

    raw: str
    python: str
    code: object

    @classmethod
    def from_raw(cls, raw: str) -> "CompiledExpression":
        python_expr = _to_python_expression(raw)
        return cls(raw=raw, python=python_expr, code=compile(python_expr, "<comsol-expr>", "eval"))

    def evaluate(self, environment: dict[str, object]) -> float:
        return float(eval(self.code, {"__builtins__": {}}, environment))


@dataclass
class AnalyticFunction:
    """Callable wrapper for COMSOL analytic functions."""

    name: str
    args: list[str]
    expression: CompiledExpression
    functions: dict[str, "AnalyticFunction | callable"]
    constants: dict[str, float]

    def __call__(self, *values: float) -> float:
        if len(values) != len(self.args):
            raise ValueError(f"{self.name} expects {len(self.args)} arguments, got {len(values)}")
        local_env = _base_math_environment()
        local_env.update(self.functions)
        local_env.update(self.constants)
        for arg_name, value in zip(self.args, values):
            local_env[arg_name] = value
        return self.expression.evaluate(local_env)


@dataclass
class SpeciesData:
    """Species-level thermodynamic expressions extracted from COMSOL."""

    name: str
    sequence_no: int
    h_expr: CompiledExpression
    s_expr: CompiledExpression


@dataclass
class ReactionData:
    """Reaction stoichiometry and forward-rate expression."""

    tag: str
    sequence_no: int
    equation: str
    reactants: dict[str, float]
    products: dict[str, float]
    kf_expr: CompiledExpression


@dataclass
class MechanismView:
    """Filtered species/reaction view used by the reduction workflow."""

    species: list[SpeciesData]
    reactions: list[ReactionData]


@dataclass
class ReactionSystemData:
    """Dense array representation of the mechanism used by the solver kernels."""

    stoich: np.ndarray
    reactant_orders: np.ndarray
    product_orders: np.ndarray


class ComsolModel:
    """Parsed COMSOL reaction mechanism cached in memory."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.parameters: dict[str, float] = {}
        self.functions: dict[str, AnalyticFunction | callable] = {}
        self.species: list[SpeciesData] = []
        self.reactions: list[ReactionData] = []
        self._load()

    def _load(self) -> None:
        """Populate the model from ``dmodel.xml`` inside the ``.mph`` archive."""

        with zipfile.ZipFile(self.model_path) as archive:
            dmodel = archive.read("dmodel.xml")
        root = ET.fromstring(dmodel)

        self.parameters = self._load_parameters(root)
        self.functions = self._load_functions(root)
        self.species = self._load_species(root)
        self.reactions = self._load_reactions(root)

    def reduced(self, omitted_species: list[str] | None = None, omitted_reactions: list[str] | None = None) -> MechanismView:
        """Return a mechanism view after removing the requested entries."""

        omitted_species_set = set(omitted_species or [])
        omitted_reactions_set = set(omitted_reactions or [])
        kept_species = [sp for sp in self.species if sp.name not in omitted_species_set]
        kept_species_names = {sp.name for sp in kept_species}
        kept_reactions = []
        for reaction in self.reactions:
            if reaction.equation in omitted_reactions_set:
                continue
            if not set(reaction.reactants).issubset(kept_species_names):
                continue
            if not set(reaction.products).issubset(kept_species_names):
                continue
            kept_reactions.append(reaction)
        return MechanismView(species=kept_species, reactions=kept_reactions)

    def _load_parameters(self, root: ET.Element) -> dict[str, float]:
        params: dict[str, float] = {
            "R_const": GAS_CONSTANT,
            "R_GAS": GAS_CONSTANT,
            "N_A_const": AVOGADRO,
        }
        for node in root.findall("./ModelParam/expressions"):
            name = node.attrib.get("name")
            expr = node.attrib.get("expr")
            if not name or expr is None:
                continue
            compiled = CompiledExpression.from_raw(expr)
            env = _base_math_environment()
            env.update(params)
            params[name] = compiled.evaluate(env)
        return params

    def _load_functions(self, root: ET.Element) -> dict[str, AnalyticFunction | callable]:
        functions: dict[str, AnalyticFunction | callable] = {}

        for feature in root.findall(".//FunctionFeature[@op='Analytic']"):
            props = {prop.attrib.get("name"): _property_raw_value(prop) for prop in feature.findall("propertyValue")}
            func_name = props.get("p:funcname")
            expr = props.get("p:expr")
            raw_args = props.get("p:args")
            if not func_name or expr is None:
                continue
            decoded_args = _decode_comsol_value(raw_args or "")
            if isinstance(decoded_args, str):
                args = [decoded_args] if decoded_args else []
            else:
                args = decoded_args
            functions[func_name] = AnalyticFunction(
                name=func_name,
                args=args,
                expression=CompiledExpression.from_raw(expr),
                functions=functions,
                constants=self.parameters,
            )

        return functions

    def _load_species(self, root: ET.Element) -> list[SpeciesData]:
        species: list[SpeciesData] = []
        for feature in root.findall(".//Physics[@tag='re']//PhysicsFeature[@op='SpeciesChem']"):
            params = self._feature_params(feature)
            name = str(params.get("specLabel", feature.attrib.get("tag", "")))
            sequence_no = int(float(str(params.get("sSequenceNo", "0"))))
            h_expr = CompiledExpression.from_raw(str(params["h"]))
            s_expr = CompiledExpression.from_raw(str(params["s"]))
            species.append(SpeciesData(name=name, sequence_no=sequence_no, h_expr=h_expr, s_expr=s_expr))
        species.sort(key=lambda item: item.sequence_no)
        return species

    def _load_reactions(self, root: ET.Element) -> list[ReactionData]:
        reactions: list[ReactionData] = []
        for feature in root.findall(".//Physics[@tag='re']//PhysicsFeature[@op='ReactionChem']"):
            params = self._feature_params(feature)
            equation = str(params["formula"]).replace("  ", " ").strip()
            reactants, products = _parse_equation(equation)
            reactions.append(
                ReactionData(
                    tag=feature.attrib.get("tag", ""),
                    sequence_no=int(float(str(params.get("rSequenceNo", "0")))),
                    equation=equation,
                    reactants=reactants,
                    products=products,
                    kf_expr=CompiledExpression.from_raw(str(params["kf"])),
                )
            )
        reactions.sort(key=lambda item: item.sequence_no)
        return reactions

    @staticmethod
    def _feature_params(feature: ET.Element) -> dict[str, str | list[str]]:
        params: dict[str, str | list[str]] = {}
        for param in feature.findall("param"):
            name = param.attrib.get("param")
            value = param.attrib.get("value")
            if name is None or value is None:
                continue
            params[name] = _decode_comsol_value(value)
        return params


def _base_math_environment() -> dict[str, object]:
    """Math namespace allowed inside translated COMSOL expressions."""

    return {
        "abs": abs,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "max": max,
        "min": min,
        "pow": pow,
        "sqrt": math.sqrt,
    }


def _resolve_model_path(path_like: str | Path | None) -> Path:
    """Resolve a model path relative to the caller's current working directory."""

    if path_like is None:
        return _DEFAULT_MODEL
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_model_to_yaml(kinetic: str | Path, thermo: str | Path | None = None):
    """Compatibility wrapper that stores the COMSOL model path."""

    del thermo
    global _model_file, _cached_model
    _model_file = _resolve_model_path(kinetic)
    _cached_model = None
    _get_model()


def load_model(model_file: str | Path):
    """Compatibility alias matching the ChemKIN backend interface."""

    load_model_to_yaml(model_file)


def _get_model() -> ComsolModel:
    """Load and cache the currently selected COMSOL model."""

    global _cached_model
    if _cached_model is None:
        if not _model_file.exists():
            raise FileNotFoundError(f"COMSOL model not found: {_model_file}")
        _cached_model = ComsolModel(_model_file)
    return _cached_model


def get_species():
    """Return the species names in model order."""

    model = _get_model()
    return [species.name for species in model.species]


def get_reactions():
    """Return reaction equations in model order."""

    model = _get_model()
    return [reaction.equation for reaction in model.reactions]


def _condition_pressure(condition: cond.ThermalCondition, time_value: float) -> float:
    """Return the condition pressure at a given time."""

    pressure_interp = _make_profile_interpolator(condition.pressure_profile)
    if pressure_interp is None:
        return ONE_ATM
    return float(pressure_interp(time_value))


def _make_profile_interpolator(profile: tuple[list[float], list[float]] | tuple[list, list] | None):
    """Build a 1D interpolator for a piecewise-linear profile.

    ``None`` is returned when the profile is absent, which is how pressure is
    treated for conditions that do not specify it explicitly.
    """

    if not profile or profile == ([], []):
        return None
    return interp1d(
        profile[0],
        profile[1],
        bounds_error=False,
        fill_value="extrapolate",
    )


def _normalise_mixture(mixture: dict[str, float], allowed_species: set[str]) -> dict[str, float]:
    """Filter unknown species and renormalise positive mixture entries."""

    filtered = {name: float(value) for name, value in mixture.items() if name in allowed_species and value > 0.0}
    total = sum(filtered.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in filtered.items()}


def _species_gibbs(species: SpeciesData, temperature: float, functions: dict[str, AnalyticFunction | callable]) -> float:
    """Evaluate species Gibbs free energy from the COMSOL enthalpy/entropy fits."""

    env = _base_math_environment()
    env.update(functions)
    env.update({"R_const": GAS_CONSTANT, "R_GAS": GAS_CONSTANT, "N_A_const": AVOGADRO, "T_var": temperature, "T_g": temperature})
    enthalpy = species.h_expr.evaluate(env)
    entropy = species.s_expr.evaluate(env)
    return enthalpy - temperature * entropy


def _build_environment(
    model: ComsolModel,
    temperature: float,
    pressure: float,
    time_value: float,
) -> dict[str, object]:
    """Construct the expression-evaluation environment for one time/state point."""

    env = _base_math_environment()
    env.update(model.functions)
    env.update(model.parameters)
    env.update(
        {
            "R_const": GAS_CONSTANT,
            "R_GAS": GAS_CONSTANT,
            "N_A_const": AVOGADRO,
            "T_var": temperature,
            "T_g": temperature,
            "t": time_value,
            "c_g": pressure / (GAS_CONSTANT * temperature),
        }
    )
    return env


def _safe_equilibrium_constant(delta_g: float, temperature: float) -> float:
    """Evaluate ``exp(-delta_g / RT)`` with overflow protection."""

    exponent = -delta_g / (GAS_CONSTANT * temperature)
    if exponent > 700.0:
        return math.exp(700.0)
    if exponent < -700.0:
        return math.exp(-700.0)
    return math.exp(exponent)


def _power_product(concentrations: np.ndarray, orders: dict[int, float]) -> float:
    """Return ``prod(c_i ** order_i)`` for one reaction side."""

    value = 1.0
    for idx, order in orders.items():
        value *= concentrations[idx] ** order
    return value


def _species_gibbs_map(
    species_list: list[SpeciesData],
    temperature: float,
    functions: dict[str, AnalyticFunction | callable],
) -> dict[str, float]:
    """Evaluate Gibbs energy for every active species once at a given temperature."""

    return {
        species.name: _species_gibbs(species, temperature, functions)
        for species in species_list
    }


def _species_gibbs_array(
    species_list: list[SpeciesData],
    temperature: float,
    functions: dict[str, AnalyticFunction | callable],
) -> np.ndarray:
    """Evaluate Gibbs energy for every active species in species order."""

    return np.array(
        [_species_gibbs(species, temperature, functions) for species in species_list],
        dtype=np.float64,
    )


def _forward_rate_constants(
    reactions: list[ReactionData],
    environment: dict[str, object],
) -> np.ndarray:
    """Evaluate forward rate constants for every reaction in order."""

    return np.array(
        [reaction.kf_expr.evaluate(environment) for reaction in reactions],
        dtype=np.float64,
    )


def _partial_power_product(concentrations: np.ndarray, orders: dict[int, float], target_idx: int) -> float:
    """Derivative of ``prod(c_i ** order_i)`` with respect to one species."""

    order = orders.get(target_idx)
    if order is None:
        return 0.0

    value = order
    for idx, exponent in orders.items():
        if idx == target_idx:
            reduced = exponent - 1.0
            if reduced > 0.0:
                value *= concentrations[idx] ** reduced
        else:
            value *= concentrations[idx] ** exponent
    return value


def _rate_products_impl(
    concentrations: np.ndarray,
    reactant_orders: np.ndarray,
    product_orders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate forward and reverse concentration products for every reaction."""

    n_species, n_reactions = reactant_orders.shape
    forward_terms = np.ones(n_reactions, dtype=np.float64)
    reverse_terms = np.ones(n_reactions, dtype=np.float64)

    for rxn_index in range(n_reactions):
        for species_index in range(n_species):
            reactant_order = reactant_orders[species_index, rxn_index]
            if reactant_order != 0.0:
                forward_terms[rxn_index] *= concentrations[species_index] ** reactant_order

            product_order = product_orders[species_index, rxn_index]
            if product_order != 0.0:
                reverse_terms[rxn_index] *= concentrations[species_index] ** product_order

    return forward_terms, reverse_terms


def _partial_products_impl(
    concentrations: np.ndarray,
    orders: np.ndarray,
    target_species: int,
) -> np.ndarray:
    """Evaluate partial derivatives of reaction-side products for one species."""

    n_species, n_reactions = orders.shape
    partials = np.zeros(n_reactions, dtype=np.float64)

    for rxn_index in range(n_reactions):
        order = orders[target_species, rxn_index]
        if order == 0.0:
            continue

        value = order
        for species_index in range(n_species):
            exponent = orders[species_index, rxn_index]
            if exponent == 0.0:
                continue
            if species_index == target_species:
                reduced = exponent - 1.0
                if reduced > 0.0:
                    value *= concentrations[species_index] ** reduced
            else:
                value *= concentrations[species_index] ** exponent
        partials[rxn_index] = value

    return partials


def _assemble_rates_impl(
    concentrations: np.ndarray,
    stoich: np.ndarray,
    reactant_orders: np.ndarray,
    product_orders: np.ndarray,
    kf_values: np.ndarray,
    equilibrium_constants: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble reaction rates and species source terms."""

    n_species, n_reactions = reactant_orders.shape
    forward_terms = np.ones(n_reactions, dtype=np.float64)
    reverse_terms = np.ones(n_reactions, dtype=np.float64)
    for rxn_index in range(n_reactions):
        for species_index in range(n_species):
            reactant_order = reactant_orders[species_index, rxn_index]
            if reactant_order != 0.0:
                forward_terms[rxn_index] *= concentrations[species_index] ** reactant_order

            product_order = product_orders[species_index, rxn_index]
            if product_order != 0.0:
                reverse_terms[rxn_index] *= concentrations[species_index] ** product_order

    reverse_rate_constants = kf_values / np.maximum(equilibrium_constants, 1e-300)
    rates = kf_values * forward_terms - reverse_rate_constants * reverse_terms
    return rates, stoich @ rates


def _assemble_jacobian_impl(
    concentrations: np.ndarray,
    stoich: np.ndarray,
    reactant_orders: np.ndarray,
    product_orders: np.ndarray,
    kf_values: np.ndarray,
    equilibrium_constants: np.ndarray,
) -> np.ndarray:
    """Assemble the dense species Jacobian."""

    n_species = concentrations.shape[0]
    reverse_rate_constants = kf_values / np.maximum(equilibrium_constants, 1e-300)
    jac = np.zeros((n_species, n_species), dtype=np.float64)

    for species_j in range(n_species):
        forward_partials = np.zeros(kf_values.shape[0], dtype=np.float64)
        reverse_partials = np.zeros(kf_values.shape[0], dtype=np.float64)

        for rxn_index in range(kf_values.shape[0]):
            reactant_order = reactant_orders[species_j, rxn_index]
            if reactant_order != 0.0:
                value = reactant_order
                for species_index in range(n_species):
                    exponent = reactant_orders[species_index, rxn_index]
                    if exponent == 0.0:
                        continue
                    if species_index == species_j:
                        reduced = exponent - 1.0
                        if reduced > 0.0:
                            value *= concentrations[species_index] ** reduced
                    else:
                        value *= concentrations[species_index] ** exponent
                forward_partials[rxn_index] = value

            product_order = product_orders[species_j, rxn_index]
            if product_order != 0.0:
                value = product_order
                for species_index in range(n_species):
                    exponent = product_orders[species_index, rxn_index]
                    if exponent == 0.0:
                        continue
                    if species_index == species_j:
                        reduced = exponent - 1.0
                        if reduced > 0.0:
                            value *= concentrations[species_index] ** reduced
                    else:
                        value *= concentrations[species_index] ** exponent
                reverse_partials[rxn_index] = value

        jac[:, species_j] = stoich @ (kf_values * forward_partials - reverse_rate_constants * reverse_partials)

    return jac


if NUMBA_AVAILABLE:
    _rate_products = njit(cache=True)(_rate_products_impl)
    _partial_products = njit(cache=True)(_partial_products_impl)
    _assemble_rates = njit(cache=True)(_assemble_rates_impl)
    _assemble_jacobian = njit(cache=True)(_assemble_jacobian_impl)
else:
    _rate_products = _rate_products_impl
    _partial_products = _partial_products_impl
    _assemble_rates = _assemble_rates_impl
    _assemble_jacobian = _assemble_jacobian_impl


def _initial_state(
    condition: cond.ThermalCondition,
    species_index: dict[str, int],
    species_set: set[str],
) -> tuple[np.ndarray, float]:
    """Build the initial concentration vector and a stable scaling reference."""

    initial_temperature = float(condition.temperature_profile[1][0])
    initial_pressure = _condition_pressure(condition, condition.temperature_profile[0][0])
    mixture = _normalise_mixture(condition.species, species_set)
    initial_total_concentration = initial_pressure / (GAS_CONSTANT * initial_temperature)

    pressure_values = (
        list(condition.pressure_profile[1])
        if condition.pressure_profile and condition.pressure_profile != ([], [])
        else [ONE_ATM]
    )
    temperature_values = [value for value in condition.temperature_profile[1] if value > 0.0]
    if not temperature_values:
        raise ValueError("Temperature profile must contain at least one positive temperature.")

    max_profile_concentration = max(
        pressure / (GAS_CONSTANT * temperature)
        for pressure in pressure_values
        for temperature in temperature_values
    )
    concentration_scale = max(initial_total_concentration, max_profile_concentration, 1.0)

    initial_state = np.zeros(len(species_index), dtype=float)
    for name, fraction in mixture.items():
        initial_state[species_index[name]] = fraction * initial_total_concentration

    return initial_state, concentration_scale


def _build_reaction_system(
    mechanism: MechanismView,
    species_index: dict[str, int],
) -> ReactionSystemData:
    """Precompute dense stoichiometry and concentration-order arrays."""

    stoich = np.zeros((len(species_index), len(mechanism.reactions)), dtype=np.float64)
    reactant_orders = np.zeros_like(stoich)
    product_orders = np.zeros_like(stoich)

    for rxn_index, reaction in enumerate(mechanism.reactions):
        for name, coefficient in reaction.reactants.items():
            idx = species_index[name]
            stoich[idx, rxn_index] -= coefficient
            reactant_orders[idx, rxn_index] = coefficient
        for name, coefficient in reaction.products.items():
            idx = species_index[name]
            stoich[idx, rxn_index] += coefficient
            product_orders[idx, rxn_index] = coefficient

    return ReactionSystemData(
        stoich=stoich,
        reactant_orders=reactant_orders,
        product_orders=product_orders,
    )


def _profile_breakpoints(condition: cond.ThermalCondition, t_end: float) -> list[float]:
    """Return the sorted set of segment boundaries used for piecewise integration."""

    breakpoints = {0.0, t_end}
    breakpoints.update(float(value) for value in condition.temperature_profile[0])
    if condition.pressure_profile and condition.pressure_profile != ([], []):
        breakpoints.update(float(value) for value in condition.pressure_profile[0])
    return sorted(time_value for time_value in breakpoints if 0.0 <= time_value <= t_end)


def _simulate_mechanism(condition: cond.ThermalCondition, mechanism: MechanismView):
    """Integrate one thermal condition for the supplied mechanism view."""

    if not mechanism.species:
        warnings.warn("Reduced mechanism has no species.")
        return [0.0], {}

    model = _get_model()
    species_names = [species.name for species in mechanism.species]
    species_set = set(species_names)
    species_index = {name: index for index, name in enumerate(species_names)}

    initial_state, concentration_scale = _initial_state(condition, species_index, species_set)
    scaled_initial_state = initial_state / concentration_scale
    temperature_interp = _make_profile_interpolator(condition.temperature_profile)
    pressure_interp = _make_profile_interpolator(condition.pressure_profile)
    reaction_system = _build_reaction_system(mechanism, species_index)

    def rhs(time_value: float, concentrations: np.ndarray) -> np.ndarray:
        clipped = np.maximum(concentrations, 0.0)
        temperature = float(temperature_interp(time_value))
        pressure = float(pressure_interp(time_value)) if pressure_interp is not None else ONE_ATM
        env = _build_environment(model, temperature, pressure, time_value)
        gibbs = _species_gibbs_array(mechanism.species, temperature, model.functions)
        kf_values = _forward_rate_constants(mechanism.reactions, env)
        delta_g = reaction_system.stoich.T @ gibbs
        equilibrium_constants = np.exp(np.clip(-delta_g / (GAS_CONSTANT * temperature), -700.0, 700.0))
        _, deriv = _assemble_rates(
            clipped,
            reaction_system.stoich,
            reaction_system.reactant_orders,
            reaction_system.product_orders,
            kf_values,
            equilibrium_constants,
        )
        if not np.all(np.isfinite(deriv)):
            raise FloatingPointError("Non-finite derivative encountered in COMSOL-derived mechanism.")
        return deriv

    def rhs_scaled(time_value: float, scaled_concentrations: np.ndarray) -> np.ndarray:
        bounded_state = np.clip(scaled_concentrations, 0.0, scaled_state_upper_bound)
        concentrations = bounded_state * concentration_scale
        return rhs(time_value, concentrations) / concentration_scale

    def jacobian(time_value: float, concentrations: np.ndarray) -> np.ndarray:
        clipped = np.maximum(concentrations, 0.0)
        temperature = float(temperature_interp(time_value))
        pressure = float(pressure_interp(time_value)) if pressure_interp is not None else ONE_ATM
        env = _build_environment(model, temperature, pressure, time_value)
        gibbs = _species_gibbs_array(mechanism.species, temperature, model.functions)
        kf_values = _forward_rate_constants(mechanism.reactions, env)
        delta_g = reaction_system.stoich.T @ gibbs
        equilibrium_constants = np.exp(np.clip(-delta_g / (GAS_CONSTANT * temperature), -700.0, 700.0))
        jac = _assemble_jacobian(
            clipped,
            reaction_system.stoich,
            reaction_system.reactant_orders,
            reaction_system.product_orders,
            kf_values,
            equilibrium_constants,
        )

        if not np.all(np.isfinite(jac)):
            raise FloatingPointError("Non-finite Jacobian encountered in COMSOL-derived mechanism.")
        return jac

    def jacobian_scaled(time_value: float, scaled_concentrations: np.ndarray) -> np.ndarray:
        bounded_state = np.clip(scaled_concentrations, 0.0, scaled_state_upper_bound)
        concentrations = bounded_state * concentration_scale
        return jacobian(time_value, concentrations)

    t_end = float(condition.temperature_profile[0][-1])
    if t_end <= 0.0:
        t_values = np.array([0.0], dtype=float)
        solution = initial_state[:, np.newaxis]
    else:
        t_values = np.linspace(0.0, t_end, max(1001, len(condition.temperature_profile[0]) * 10))
        segment_times = _profile_breakpoints(condition, t_end)

        time_blocks: list[np.ndarray] = []
        state_blocks: list[np.ndarray] = []
        current_state = scaled_initial_state

        for segment_index, (segment_start, segment_stop) in enumerate(zip(segment_times[:-1], segment_times[1:])):
            if segment_stop <= segment_start:
                continue

            segment_duration = segment_stop - segment_start
            segment_mask = (t_values >= segment_start) & (t_values <= segment_stop)
            segment_eval = t_values[segment_mask]

            # Always advance exactly to the breakpoint so the next segment starts
            # from a state consistent with the imposed temperature/pressure jump.
            if segment_eval.size == 0 or segment_eval[-1] < segment_stop:
                segment_eval = np.append(segment_eval, segment_stop)
            if segment_eval[0] > segment_start:
                segment_eval = np.insert(segment_eval, 0, segment_start)

            solved = None
            last_error: RuntimeError | None = None
            max_step = 0.005 if segment_duration <= 1e-2 else 0.05
            atol_vector = np.maximum(absolute_tolerance / concentration_scale, 1e-12 * np.maximum(current_state, 1.0))
            segment_eval_local = (segment_eval - segment_start) / segment_duration

            def rhs_local(local_time: float, scaled_concentrations: np.ndarray) -> np.ndarray:
                physical_time = segment_start + local_time * segment_duration
                return segment_duration * rhs_scaled(physical_time, scaled_concentrations)

            def jacobian_local(local_time: float, scaled_concentrations: np.ndarray) -> np.ndarray:
                physical_time = segment_start + local_time * segment_duration
                return segment_duration * jacobian_scaled(physical_time, scaled_concentrations)

            for method in solver_methods:
                solve_kwargs = {
                    "method": method,
                    "t_eval": segment_eval_local,
                    "rtol": relative_tolerance,
                    "atol": atol_vector,
                    "max_step": max_step,
                }
                if method != "LSODA":
                    solve_kwargs["jac"] = jacobian_local

                candidate = solve_ivp(
                    rhs_local,
                    (0.0, 1.0),
                    current_state,
                    **solve_kwargs,
                )
                if candidate.success:
                    solved = candidate
                    break
                last_error = RuntimeError(f"{method} failed: {candidate.message}")

            if solved is None:
                if last_error is None:
                    raise RuntimeError("No solver methods configured.")
                raise last_error

            current_state = np.maximum(solved.y[:, -1], 0.0)
            segment_solution = np.maximum(solved.y, 0.0) * concentration_scale
            segment_output_times = segment_start + solved.t * segment_duration

            if segment_index > 0 and segment_output_times.size > 0:
                segment_output_times = segment_output_times[1:]
                segment_solution = segment_solution[:, 1:]

            if segment_output_times.size > 0:
                time_blocks.append(segment_output_times)
                state_blocks.append(segment_solution)

        if not time_blocks:
            t_values = np.array([0.0], dtype=float)
            solution = initial_state[:, np.newaxis]
        else:
            t_values = np.concatenate(time_blocks)
            solution = np.concatenate(state_blocks, axis=1)

    total_concentration = np.sum(solution, axis=0)
    total_concentration[total_concentration <= 0.0] = 1.0
    mole_fractions = solution / total_concentration

    result_dict = {
        species.name: list(mole_fractions[index, :])
        for index, species in enumerate(mechanism.species)
    }
    return list(t_values), result_dict


def run_standard_model(condition: cond.ThermalCondition):
    """Run the full COMSOL-derived mechanism for one condition."""

    mechanism = _get_model().reduced()
    return _simulate_mechanism(condition, mechanism)


def run_reduced_species_model(condition: cond.ThermalCondition, omitted_species: list[str]):
    """Run a species-reduced mechanism while keeping all remaining reactions."""

    mechanism = _get_model().reduced(omitted_species=omitted_species)
    return _simulate_mechanism(condition, mechanism)


def run_reduced_reactions_model(condition: cond.ThermalCondition, omitted_species: list[str], omitted_reactions: list[str]):
    """Run a mechanism reduced by both species and explicit reaction removal."""

    mechanism = _get_model().reduced(omitted_species=omitted_species, omitted_reactions=omitted_reactions)
    return _simulate_mechanism(condition, mechanism)
