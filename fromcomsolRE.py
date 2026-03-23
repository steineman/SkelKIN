"""COMSOL Reaction Engineering backend for SkelKIN.

This module mirrors the public API of ``fromchemKIN.py`` while keeping COMSOL
as the actual transient solver for the thermal workflow.

What this file is responsible for:
1. Parse the ``.mph`` archive so SkelKIN can inspect the RE species/reactions
   and build reduced mechanism views in Python.
2. Start and manage live COMSOL sessions through ``mph``.
3. Create a Python-owned temporary COMSOL study/solver/dataset runtime for each
   worker process instead of depending on a study saved in the model file.
4. Map each condition-file temperature profile to COMSOL's ``T_var(t)``
   interpolation function.
5. Run the full or reduced mechanism in COMSOL, retrying with more conservative
   solver settings when a condition is hard.
6. Extract concentration histories back into the SkelKIN comparison format.

There is also a manual export helper in this file that can write a reduced
``.mph`` copy with species/reactions disabled, but the main step-2 workflow no
longer triggers that automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import condition_objects as cond

try:
    import mph
    import jpype
    from jpype import JArray, JString
    from jpype.types import JInt
    MPH_AVAILABLE = True
except ImportError:
    mph = None
    jpype = None
    JArray = None
    JString = None
    JInt = None
    MPH_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False


def preferred_comsol_session_mode() -> str:
    """Return the most reliable MPh session mode for this COMSOL workflow."""

    return "client-server"


relative_tolerance = 1e-3
absolute_tolerance = 1e-5
solver_methods = ("BDF", "LSODA")
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
_DEFAULT_MODEL = _MODULE_ROOT / "SkelKIN" / "0D_260312.mph"
_model_file = _DEFAULT_MODEL
_cached_model: "ComsolModel | None" = None
_cached_client = None
_cached_comsol_model = None
_cached_feature_nodes_model = None
_cached_feature_nodes: dict[str, dict[str, object]] | None = None
_cached_applied_mechanism: tuple[frozenset[str], frozenset[str]] | None = None
_cached_runtime_model = None
_cached_runtime = None
_comsol_cores = 1
_comsol_session_mode = preferred_comsol_session_mode()

# Names and tuning knobs for the temporary COMSOL runtime that Python owns.
# These are the settings the worker-local runtime uses when it creates a fresh
# transient study/solver sequence around a condition profile.
_COMSOL_STUDY_NAME = "Parametric"
_COMSOL_DATASET_NAME = "Parametric//Solution 1"
_COMSOL_TEMPERATURE_FUNCTION_LABEL = "Temp"
_COMSOL_TEMPERATURE_FUNCTION_TAG = "step1"
_COMSOL_SWEEP_PATH = "studies/Parametric/All"
_COMSOL_PHYSICS_PATH = "physics/Reaction Engineering"
_COMSOL_EVALUATIONS_PATH = "evaluations"
_COMSOL_CONCENTRATION_PLOT_PATH = "plots/Concentration (re)"
_COMSOL_TEMP_STUDY_LABEL = "SkelKIN Study"
_COMSOL_TEMP_DATASET_LABEL = "SkelKIN Dataset"
_COMSOL_TEMPERATURE_INTERPOLATION = "piecewisecubic"
_COMSOL_INITIAL_TIMESTEP = 1e-10
_COMSOL_MAXSTEP_PLATEAU_DIVISOR = 50.0
_COMSOL_NONLINEAR_MINSTEP = 1e-5
_COMSOL_NONLINEAR_MAXITER = 80
_COMSOL_PIVOT_PERTURB = 1e-7
_COMSOL_RETRY_MAXSTEP_DIVISOR = 200.0
_COMSOL_RETRY_MINSTEP = 1e-7
_COMSOL_RETRY_MAXITER = 150
_COMSOL_RETRY_PIVOT_PERTURB = 1e-5
_COMSOL_TRANSITION_MULTIPLIERS = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
_COMSOL_SEGMENTATION_EXTRA_POINTS = 3
_COMSOL_START_RETRIES = 3
_COMSOL_START_RETRY_DELAY = 2.0
# Add fixed COMSOL parameters here if they should be pushed from Python into the model.
_COMSOL_MODEL_PARAMETERS: dict[str, str | float | int] = {}


def _prepare_comsol_runtime() -> None:
    """Prepare COMSOL runtime state before starting an ``mph`` session.

    In practice this is mostly Windows/parallel-worker hygiene:
    - set the requested ``mph`` session mode
    - pre-create COMSOL's Equinox lock directory when needed
    - ensure APPDATA exists for COMSOL/MPh startup
    """

    if not MPH_AVAILABLE:
        return

    mph.option("session", _comsol_session_mode)

    comsol_home = Path.home() / ".comsol"
    if comsol_home.exists():
        for version_dir in comsol_home.iterdir():
            if not version_dir.is_dir():
                continue
            if not version_dir.name.startswith("v"):
                continue
            manager_dir = version_dir / "configuration" / "comsol" / "org.eclipse.osgi" / ".manager"
            manager_dir.mkdir(parents=True, exist_ok=True)

    # Some COMSOL/MPh code paths expect APPDATA to exist and be writable.
    os.environ.setdefault("APPDATA", str(Path.home() / "AppData" / "Roaming"))


def _start_mph_client():
    """Start one ``mph`` client, retrying transient COMSOL startup failures.

    The retry loop matters for the reduction workflow because many workers may
    start COMSOL at nearly the same time.
    """

    existing_client = getattr(getattr(mph, "session", None), "client", None)
    if existing_client is not None:
        return existing_client

    last_error: Exception | None = None
    for attempt in range(1, _COMSOL_START_RETRIES + 1):
        try:
            _prepare_comsol_runtime()
            return mph.start(cores=_comsol_cores)
        except Exception as exc:
            last_error = exc
            existing_client = getattr(getattr(mph, "session", None), "client", None)
            if existing_client is not None:
                return existing_client
            if jpype is not None and jpype.isJVMStarted():
                break
            if attempt == _COMSOL_START_RETRIES:
                break
            time.sleep(_COMSOL_START_RETRY_DELAY)
    assert last_error is not None
    raise last_error


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


@dataclass
class ComsolRuntime:
    """Persistent worker-local COMSOL study, solver, and dataset."""

    study: object
    step: object
    solution: object
    dataset: object


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


def _reset_cached_runtime(*, remove_nodes: bool = False) -> None:
    """Drop the cached worker-local COMSOL runtime, optionally removing its nodes."""

    global _cached_runtime_model, _cached_runtime

    runtime_model = _cached_runtime_model
    runtime = _cached_runtime

    if remove_nodes and runtime_model is not None and runtime is not None:
        for container_path, node in (
            ("datasets", runtime.dataset),
            ("solutions", runtime.solution),
            ("studies", runtime.study),
        ):
            try:
                tag = node.tag()
            except Exception:
                tag = None
            _remove_comsol_node(runtime_model / container_path, tag)

    _cached_runtime_model = None
    _cached_runtime = None


def load_model_to_yaml(kinetic: str | Path, thermo: str | Path | None = None):
    """Compatibility wrapper that stores the COMSOL model path."""

    del thermo
    global _model_file, _cached_model, _cached_client, _cached_comsol_model, _cached_applied_mechanism, _cached_feature_nodes_model, _cached_feature_nodes
    _model_file = _resolve_model_path(kinetic)
    _cached_model = None
    _cached_applied_mechanism = None
    _cached_feature_nodes_model = None
    _cached_feature_nodes = None
    _reset_cached_runtime()
    if _cached_client is not None:
        try:
            _cached_client.remove(_cached_comsol_model)
        except Exception:
            pass
    _cached_client = None
    _cached_comsol_model = None
    _get_model()


def load_comsol_model(model_file: str | Path):
    """Load a COMSOL ``.mph`` model for parsing and live COMSOL solves."""

    load_model_to_yaml(model_file)


def set_comsol_cores(cores: int = 1):
    """Set the number of CPU cores used by each COMSOL/mph session."""

    global _comsol_cores, _cached_client, _cached_comsol_model, _cached_applied_mechanism, _cached_feature_nodes_model, _cached_feature_nodes
    cores = max(1, int(cores))
    if cores != _comsol_cores:
        _comsol_cores = cores
        _cached_applied_mechanism = None
        _cached_feature_nodes_model = None
        _cached_feature_nodes = None
        _reset_cached_runtime()
        if _cached_client is not None:
            try:
                _cached_client.remove(_cached_comsol_model)
            except Exception:
                pass
            _cached_client = None
            _cached_comsol_model = None


def set_comsol_session_mode(mode: str | None = None):
    """Set the MPh session mode used for future COMSOL starts."""

    global _comsol_session_mode, _cached_client, _cached_comsol_model, _cached_applied_mechanism, _cached_feature_nodes_model, _cached_feature_nodes
    if mode is None:
        mode = preferred_comsol_session_mode()
    if mode not in {"client-server", "stand-alone"}:
        raise ValueError("COMSOL session mode must be 'client-server' or 'stand-alone'.")
    if mode != _comsol_session_mode:
        _comsol_session_mode = mode
        _cached_applied_mechanism = None
        _cached_feature_nodes_model = None
        _cached_feature_nodes = None
        _reset_cached_runtime()
        if _cached_client is not None:
            try:
                _cached_client.remove(_cached_comsol_model)
            except Exception:
                pass
            _cached_client = None
            _cached_comsol_model = None


def load_model(model_file: str | Path):
    """Compatibility alias matching the ChemKIN backend interface."""

    load_comsol_model(model_file)


def _get_model() -> ComsolModel:
    """Load and cache the currently selected COMSOL model."""

    global _cached_model
    if _cached_model is None:
        if not _model_file.exists():
            raise FileNotFoundError(f"COMSOL model not found: {_model_file}")
        _cached_model = ComsolModel(_model_file)
    return _cached_model


def _get_comsol_model():
    """Load and cache the live COMSOL model used for actual solves."""

    if not MPH_AVAILABLE:
        raise RuntimeError("The mph library is required for COMSOL-backed solves.")

    global _cached_client, _cached_comsol_model, _cached_applied_mechanism, _cached_feature_nodes_model, _cached_feature_nodes
    if _cached_comsol_model is None:
        _cached_client = _start_mph_client()
        _cached_comsol_model = _cached_client.load(str(_model_file))
        _cached_applied_mechanism = None
        _cached_feature_nodes_model = None
        _cached_feature_nodes = None
        _reset_cached_runtime()
    return _cached_comsol_model


def _get_feature_nodes(model) -> dict[str, dict[str, object]]:
    """Map COMSOL feature tags to toggleable species and reaction nodes."""

    global _cached_feature_nodes_model, _cached_feature_nodes
    if _cached_feature_nodes_model is model and _cached_feature_nodes is not None:
        return _cached_feature_nodes

    physics = model / _COMSOL_PHYSICS_PATH
    species_nodes: dict[str, object] = {}
    reaction_nodes: dict[str, object] = {}

    for child in physics.children():
        if child.type() == "SpeciesChem":
            species_nodes[child.tag()] = child
        elif child.type() == "ReactionChem":
            reaction_nodes[child.tag()] = child

    _cached_feature_nodes_model = model
    _cached_feature_nodes = {
        "species": species_nodes,
        "reactions": reaction_nodes,
    }
    return _cached_feature_nodes


def _collect_feature_nodes(model) -> dict[str, dict[str, object]]:
    """Collect species/reaction feature nodes without touching global caches."""

    physics = model / _COMSOL_PHYSICS_PATH
    species_nodes: dict[str, object] = {}
    reaction_nodes: dict[str, object] = {}

    for child in physics.children():
        if child.type() == "SpeciesChem":
            species_nodes[child.tag()] = child
        elif child.type() == "ReactionChem":
            reaction_nodes[child.tag()] = child

    return {
        "species": species_nodes,
        "reactions": reaction_nodes,
    }


def get_species():
    """Return the species names in model order."""

    model = _get_model()
    return [species.name for species in model.species]


def get_reactions():
    """Return reaction equations in model order."""

    model = _get_model()
    return [reaction.equation for reaction in model.reactions]


def _export_reduced_comsol_model_impl(
    model_file: str | Path,
    omitted_species: list[str],
    output_file: str | Path,
    *,
    session_mode: str,
    cores: int,
) -> Path:
    """Save a reduced COMSOL copy inside one isolated Python process.

    This helper is intentionally isolated because ``mph`` only allows one
    client per Python process. It is meant for manual/explicit export, not for
    the automatic step-2 flow.
    """

    global _comsol_session_mode, _comsol_cores

    model_path = Path(model_file).resolve()
    output_path = Path(output_file).resolve()
    omitted_species = list(dict.fromkeys(omitted_species))

    _comsol_session_mode = session_mode
    _comsol_cores = max(1, int(cores))

    mechanism = ComsolModel(model_path).reduced(omitted_species=omitted_species)
    kept_species = {species.name for species in mechanism.species}
    kept_reactions = {reaction.tag for reaction in mechanism.reactions}

    client = _start_mph_client()
    model = client.load(str(model_path))
    try:
        feature_nodes = _collect_feature_nodes(model)
        for tag, node in feature_nodes["species"].items():
            node.toggle("enable" if tag in kept_species else "disable")
        for tag, node in feature_nodes["reactions"].items():
            node.toggle("enable" if tag in kept_reactions else "disable")
        model.save(str(output_path))
    finally:
        try:
            client.remove(model)
        except Exception:
            pass

    return output_path


def export_reduced_comsol_model(
    omitted_species: list[str],
    output_file: str | Path | None = None,
) -> Path:
    """Save a copy of the COMSOL model with omitted species disabled.

    Reactions removed by those omitted species are also disabled so the saved
    Reaction Engineering model remains internally consistent.

    The main driver currently leaves this as a manual step and only prints the
    recommended species to remove. This function remains available for explicit
    export workflows and debugging.
    """

    omitted_species = list(dict.fromkeys(omitted_species))

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = _model_file.with_name(f"{_model_file.stem}_SkelKIN_{timestamp}{_model_file.suffix}")

    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module_dir = Path(__file__).resolve().parent
    script = (
        "import json, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import fromcomsolRE; "
        "fromcomsolRE._export_reduced_comsol_model_impl("
        "sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], "
        "session_mode=sys.argv[5], cores=int(sys.argv[6]))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(module_dir),
            str(_model_file),
            json.dumps(omitted_species),
            str(output_path),
            _comsol_session_mode,
            str(_comsol_cores),
        ],
        cwd=str(module_dir),
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Unknown export failure."
        raise RuntimeError(detail) from None

    if not output_path.exists():
        raise RuntimeError(f"COMSOL export reported success but no file was written: {output_path}") from None

    return output_path


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
        species.name: mole_fractions[index, :].astype(float).tolist()
        for index, species in enumerate(mechanism.species)
    }
    return list(t_values), result_dict


def _condition_temperature_profile(condition: cond.ThermalCondition) -> tuple[list[float], list[float]]:
    """Return a validated temperature profile for COMSOL interpolation."""

    times, temperatures = condition.temperature_profile
    if not times or not temperatures or len(times) != len(temperatures):
        raise ValueError("Condition temperature profile is empty or malformed.")

    cleaned_times: list[float] = []
    cleaned_temperatures: list[float] = []

    for raw_time, raw_temperature in zip(times, temperatures):
        time_value = float(raw_time)
        temperature_value = float(raw_temperature)
        if cleaned_times and time_value <= cleaned_times[-1]:
            raise ValueError("Condition temperature times must be strictly increasing for cubic interpolation.")
        cleaned_times.append(time_value)
        cleaned_temperatures.append(temperature_value)

    return cleaned_times, cleaned_temperatures


def _condition_initial_temperature(condition: cond.ThermalCondition) -> float:
    """Return the starting gas temperature for the COMSOL RE initial condition."""

    _, temperatures = _condition_temperature_profile(condition)
    return float(temperatures[0])


def _condition_smallest_time_step(condition: cond.ThermalCondition) -> float:
    """Return the smallest positive time increment in the temperature profile."""

    times, _ = _condition_temperature_profile(condition)
    if len(times) < 2:
        return _COMSOL_INITIAL_TIMESTEP
    min_step = min(times[index + 1] - times[index] for index in range(len(times) - 1))
    return max(float(min_step), _COMSOL_INITIAL_TIMESTEP)


def _condition_plateau_time_step(condition: cond.ThermalCondition) -> float:
    """Return a representative non-transition timescale for the condition."""

    times, _ = _condition_temperature_profile(condition)
    if len(times) < 2:
        return _COMSOL_INITIAL_TIMESTEP

    deltas = [times[index + 1] - times[index] for index in range(len(times) - 1)]
    smallest = min(deltas)
    threshold = max(_COMSOL_INITIAL_TIMESTEP * 1e3, smallest * 100.0)
    significant = [delta for delta in deltas if delta > threshold]
    if significant:
        return min(significant)
    return max(float(smallest), _COMSOL_INITIAL_TIMESTEP)


def _condition_solver_times(condition: cond.ThermalCondition) -> list[float]:
    """Return the explicit COMSOL output times for one condition.

    The condition files define the forcing profile. COMSOL should therefore
    solve/output on that same timeline rather than reusing the 1-second study
    range stored in the original `.mph`.
    """

    times, temperatures = _condition_temperature_profile(condition)
    solver_times = [0.0] if times[0] > 0.0 else []
    solver_times.extend(times)

    for index in range(len(times) - 1):
        start_time = float(times[index])
        end_time = float(times[index + 1])
        start_temperature = float(temperatures[index])
        end_temperature = float(temperatures[index + 1])

        if math.isclose(start_temperature, end_temperature, rel_tol=0.0, abs_tol=0.0):
            continue

        transition_width = end_time - start_time
        next_time = float(times[index + 2]) if index + 2 < len(times) else end_time
        for multiplier in _COMSOL_TRANSITION_MULTIPLIERS:
            candidate = end_time + transition_width * multiplier
            if candidate < next_time:
                solver_times.append(candidate)

    return sorted(set(solver_times))


def _condition_tlist_string(condition: cond.ThermalCondition) -> str:
    """Serialize the condition times for COMSOL's `tlist` property."""

    return " ".join(str(time_value) for time_value in _condition_solver_times(condition))


def _solver_times_to_string(times: list[float]) -> str:
    """Serialize an explicit time list for COMSOL."""

    return " ".join(str(time_value) for time_value in times)


def _condition_smallest_solver_step(condition: cond.ThermalCondition) -> float:
    """Return the smallest strictly positive time increment in the solver grid."""

    solver_times = _condition_solver_times(condition)
    positive_steps = [
        solver_times[index + 1] - solver_times[index]
        for index in range(len(solver_times) - 1)
        if solver_times[index + 1] - solver_times[index] > 0.0
    ]
    if not positive_steps:
        return _COMSOL_INITIAL_TIMESTEP
    return min(positive_steps)


def _condition_xch4(condition: cond.ThermalCondition) -> float:
    """Return the CH4 feed fraction used by the COMSOL model."""

    return float(condition.species.get("CH4", 0.0))


def _condition_model_parameters(condition: cond.ThermalCondition) -> dict[str, str | float | int]:
    """Return the Python-owned COMSOL parameter set for one condition."""

    parameters = dict(_COMSOL_MODEL_PARAMETERS)
    parameters["xCH4"] = _condition_xch4(condition)
    parameters["T_g"] = _condition_initial_temperature(condition)
    return parameters


def _apply_comsol_model_parameters(model, parameters: dict[str, str | float | int]) -> None:
    """Write Python-owned parameter values into the COMSOL model."""

    for name, value in parameters.items():
        model.parameter(name, str(value))


def _ensure_comsol_temperature_function(model):
    """Ensure the COMSOL model has a cubic interpolation function named ``Temp``."""

    functions = (model / "functions").java

    try:
        existing = next(
            child for child in (model / "functions").children()
            if child.tag() == _COMSOL_TEMPERATURE_FUNCTION_TAG
        )
    except StopIteration:
        existing = None

    if existing is not None and existing.type() != "Interpolation":
        functions.remove(_COMSOL_TEMPERATURE_FUNCTION_TAG)
        existing = None

    if existing is None:
        feature = functions.create(_COMSOL_TEMPERATURE_FUNCTION_TAG, "Interpolation")
        feature.label(_COMSOL_TEMPERATURE_FUNCTION_LABEL)
    else:
        feature = existing.java
        feature.label(_COMSOL_TEMPERATURE_FUNCTION_LABEL)

    return feature


def _configure_comsol_temperature_function(model, condition: cond.ThermalCondition) -> None:
    """Map the condition-file temperature profile to COMSOL's ``T_var(t)``.

    The condition file is treated as the source of truth. We rewrite the COMSOL
    interpolation table for every condition instead of relying on a pre-saved
    study-specific temperature function in the model.
    """

    times, temperatures = _condition_temperature_profile(condition)
    function = _ensure_comsol_temperature_function(model)
    function.set("funcname", "T_var")
    # Global cubic splines overshoot badly on flat/jump/flat profiles.
    # COMSOL's piecewise-cubic interpolation keeps the forcing local while
    # still using cubic polynomials between points.
    function.set("interp", _COMSOL_TEMPERATURE_INTERPOLATION)
    function.set("extrap", "const")
    function.set("argunit", ["s"])
    function.set("fununit", "K")
    function.set(
        "table",
        [[str(time_value), str(temperature_value)] for time_value, temperature_value in zip(times, temperatures)],
    )


def _configure_comsol_time_solver(solution, condition: cond.ThermalCondition) -> None:
    """Apply conservative Python-owned solver settings to one temporary COMSOL solution."""

    solver = solution / "Time-Dependent Solver 1"
    fully_coupled = solver / "Fully Coupled 1"
    direct = solver / "Direct"
    plateau_step = _condition_plateau_time_step(condition)

    solver.property("initialstepbdfactive", True)
    solver.property("initialstepbdf", _COMSOL_INITIAL_TIMESTEP)
    solver.property(
        "maxstepbdf",
        min(0.02, max(plateau_step / _COMSOL_MAXSTEP_PLATEAU_DIVISOR, _COMSOL_INITIAL_TIMESTEP)),
    )
    solver.property("plot", "off")
    solver.property("tlist", _condition_tlist_string(condition))
    fully_coupled.property("maxiter", _COMSOL_NONLINEAR_MAXITER)
    fully_coupled.property("minstep", _COMSOL_NONLINEAR_MINSTEP)
    fully_coupled.property("jtech", "onevery")
    direct.property("pivotperturb", _COMSOL_PIVOT_PERTURB)


def _create_comsol_runtime(model, condition: cond.ThermalCondition) -> ComsolRuntime:
    """Create the persistent worker-local study, solver, and dataset.

    Each worker keeps these nodes alive and reuses them across many candidates,
    which is much cheaper and generally more robust than recreating the entire
    COMSOL runtime for every step-1 / step-2 simulation.
    """

    existing_solution_tags = {solution.tag() for solution in (model / "solutions").children()}
    study = (model / "studies").create(name=_COMSOL_TEMP_STUDY_LABEL)
    step = study.create("Transient", name="Run")

    step.property("tlist", _condition_tlist_string(condition))

    study.java.createAutoSequences("all")

    created_solutions = [
        solution for solution in (model / "solutions").children()
        if solution.tag() not in existing_solution_tags
    ]
    if len(created_solutions) != 1:
        raise RuntimeError("Failed to identify the temporary COMSOL solution sequence.")

    solution = created_solutions[0]
    _configure_comsol_time_solver(solution, condition)
    dataset = (model / "datasets").create("Solution", name=_COMSOL_TEMP_DATASET_LABEL)
    dataset.property("solution", solution.tag())
    return ComsolRuntime(study=study, step=step, solution=solution, dataset=dataset)


def _runtime_is_valid(runtime: ComsolRuntime | None) -> bool:
    """Return whether the cached runtime still points to live COMSOL nodes."""

    if runtime is None:
        return False

    try:
        runtime.study.tag()
        runtime.step.tag()
        runtime.solution.tag()
        runtime.dataset.tag()
    except Exception:
        return False
    return True


def _configure_comsol_runtime(runtime: ComsolRuntime, condition: cond.ThermalCondition) -> None:
    """Update the persistent runtime nodes for one condition."""

    runtime.step.property("tlist", _condition_tlist_string(condition))
    runtime.dataset.property("solution", runtime.solution.tag())
    _configure_comsol_time_solver(runtime.solution, condition)


def _set_runtime_tlist(runtime: ComsolRuntime, times: list[float]) -> None:
    """Update the temporary study and solver with one explicit segment time list."""

    tlist = _solver_times_to_string(times)
    runtime.step.property("tlist", tlist)
    time_solver = runtime.solution / "Time-Dependent Solver 1"
    time_solver.property("tlist", tlist)


def _condition_transition_end_times(condition: cond.ThermalCondition) -> list[float]:
    """Return the times at which a temperature transition interval ends."""

    times, temperatures = _condition_temperature_profile(condition)
    transition_ends: list[float] = []
    for index in range(len(times) - 1):
        if not math.isclose(float(temperatures[index]), float(temperatures[index + 1]), rel_tol=0.0, abs_tol=0.0):
            transition_ends.append(float(times[index + 1]))
    return transition_ends


def _condition_transition_priority_boundaries(condition: cond.ThermalCondition) -> list[float]:
    """Return continuation boundaries ordered from latest/slowest fallback first."""

    times, temperatures = _condition_temperature_profile(condition)
    solver_times = _condition_solver_times(condition)
    boundaries: list[float] = []

    for index in range(len(times) - 2, -1, -1):
        start_time = float(times[index])
        end_time = float(times[index + 1])
        start_temperature = float(temperatures[index])
        end_temperature = float(temperatures[index + 1])
        if math.isclose(start_temperature, end_temperature, rel_tol=0.0, abs_tol=0.0):
            continue

        next_time = float(times[index + 2]) if index + 2 < len(times) else float("inf")
        per_transition = [end_time]
        post_transition = [
            time_value
            for time_value in solver_times
            if end_time < time_value < next_time
        ]
        per_transition.extend(post_transition[:_COMSOL_SEGMENTATION_EXTRA_POINTS])

        for boundary in per_transition:
            if any(math.isclose(boundary, existing, rel_tol=0.0, abs_tol=1e-12) for existing in boundaries):
                continue
            boundaries.append(boundary)

    return boundaries


def _split_solver_times_at_boundaries(
    solver_times: list[float],
    boundary_times: list[float],
) -> list[list[float]]:
    """Split one explicit time grid at the requested boundary times."""

    if len(solver_times) <= 1 or not boundary_times:
        return [solver_times]

    segments: list[list[float]] = []
    current_segment: list[float] = []
    for time_value in solver_times:
        current_segment.append(time_value)
        if any(math.isclose(time_value, boundary, rel_tol=0.0, abs_tol=1e-12) for boundary in boundary_times):
            segments.append(current_segment)
            current_segment = []

    if current_segment:
        segments.append(current_segment)

    filtered = [segment for segment in segments if segment]
    if len(filtered) <= 1:
        return filtered

    merged: list[list[float]] = []
    index = 0
    while index < len(filtered):
        segment = list(filtered[index])
        while len(segment) < 2 and index + 1 < len(filtered):
            index += 1
            segment.extend(filtered[index])
        if len(segment) < 2 and merged:
            merged[-1].extend(segment)
        else:
            merged.append(segment)
        index += 1
    return merged


def _condition_solver_segmentations(condition: cond.ThermalCondition) -> list[list[list[float]]]:
    """Return progressively more segmented solve plans for one condition.

    The first plan is a monolithic run. Later plans add continuation breaks
    around the temperature-profile transitions. This lets COMSOL continue from
    already converged early segments instead of forcing a single transient solve
    through the whole profile when only one narrow jump is problematic.
    """

    solver_times = _condition_solver_times(condition)
    plans: list[list[list[float]]] = [[solver_times]]
    priority_boundaries = _condition_transition_priority_boundaries(condition)
    for count in range(1, len(priority_boundaries) + 1):
        boundaries = priority_boundaries[:count]
        plan = _split_solver_times_at_boundaries(solver_times, boundaries)
        if plan != plans[-1]:
            plans.append(plan)
    return plans


def _ensure_comsol_runtime(
    model,
    condition: cond.ThermalCondition,
    *,
    recreate: bool = False,
) -> ComsolRuntime:
    """Return the worker-local runtime, creating or rebuilding it as needed."""

    global _cached_runtime_model, _cached_runtime

    if recreate and _cached_runtime_model is model:
        _reset_cached_runtime(remove_nodes=True)

    if _cached_runtime_model is not model or not _runtime_is_valid(_cached_runtime):
        _cached_runtime_model = model
        _cached_runtime = _create_comsol_runtime(model, condition)
    else:
        _configure_comsol_runtime(_cached_runtime, condition)

    return _cached_runtime


def _clear_comsol_runtime_solution(runtime: ComsolRuntime) -> None:
    """Drop previously computed solution data while keeping the solver sequence."""

    try:
        if not runtime.solution.java.isEmpty():
            runtime.solution.java.clearSolutionData()
    except Exception:
        try:
            runtime.solution.java.clearSolution()
        except Exception:
            pass


def _merge_comsol_segment_results(
    accumulated_times: list[float],
    accumulated_results: dict[str, list[float]],
    segment_times: list[float],
    segment_results: dict[str, list[float]],
) -> tuple[list[float], dict[str, list[float]]]:
    """Append one segment result, skipping any overlap already present."""

    if not accumulated_times:
        return list(segment_times), {name: list(values) for name, values in segment_results.items()}

    overlap_tolerance = 1e-12
    last_time = accumulated_times[-1]
    start_index = 0
    while start_index < len(segment_times) and segment_times[start_index] <= last_time + overlap_tolerance:
        start_index += 1

    if start_index >= len(segment_times):
        return accumulated_times, accumulated_results

    accumulated_times.extend(segment_times[start_index:])
    for species_name, values in segment_results.items():
        accumulated_results[species_name].extend(values[start_index:])
    return accumulated_times, accumulated_results


def _run_comsol_runtime_segments(
    model,
    runtime: ComsolRuntime,
    mechanism: MechanismView,
    segments: list[list[float]],
) -> tuple[list[float], dict[str, list[float]]]:
    """Solve one condition by continuing across explicit time segments.

    The stitched result is returned as one continuous history so the rest of
    SkelKIN can compare it exactly like a single-run simulation.
    """

    if not segments:
        raise RuntimeError("Condition does not define any solver times.")

    stitched_times: list[float] = []
    stitched_results: dict[str, list[float]] = {}

    _set_runtime_tlist(runtime, segments[0])
    runtime.study.java.run()
    segment_times, segment_results = _evaluate_current_comsol_solution(model, mechanism, dataset=runtime.dataset)
    stitched_times, stitched_results = _merge_comsol_segment_results(
        stitched_times,
        stitched_results,
        segment_times,
        segment_results,
    )

    for segment in segments[1:]:
        _set_runtime_tlist(runtime, segment)
        runtime.solution.java.continueRun()
        segment_times, segment_results = _evaluate_current_comsol_solution(model, mechanism, dataset=runtime.dataset)
        stitched_times, stitched_results = _merge_comsol_segment_results(
            stitched_times,
            stitched_results,
            segment_times,
            segment_results,
        )

    return stitched_times, stitched_results


def _configure_comsol_result_plots(model) -> None:
    """Disable automatic COMSOL plotting and keep plot definitions consistent."""

    try:
        (model / "plots").java.setOnlyPlotWhenRequested(True)
    except Exception:
        pass

    try:
        plot_group = model / _COMSOL_CONCENTRATION_PLOT_PATH
        global_plot = plot_group / "Global 1"
    except Exception:
        return

    try:
        plot_group.java.active(False)
    except Exception:
        pass

    plot_group.property("progressactive", False)
    plot_group.property("showlegends", "off")
    global_plot.property("xdata", "expr")
    global_plot.property("xdataexpr", "t")
    global_plot.property("xdatadescractive", True)
    global_plot.property("xdatadescr", "Time")
    plot_group.property("xlabelactive", True)
    plot_group.property("xlabel", "Time [s]")


def _configure_comsol_retry_solver(solution, condition: cond.ThermalCondition) -> None:
    """Apply a harsher solver configuration for retrying hard cases."""

    time_solver = solution / "Time-Dependent Solver 1"
    fully_coupled = time_solver / "Fully Coupled 1"
    direct = time_solver / "Direct"
    plateau_step = _condition_plateau_time_step(condition)
    smallest_solver_step = _condition_smallest_solver_step(condition)

    time_solver.property(
        "maxstepbdf",
        min(
            0.001,
            max(
                _COMSOL_INITIAL_TIMESTEP,
                min(
                    plateau_step / _COMSOL_RETRY_MAXSTEP_DIVISOR,
                    smallest_solver_step * 10.0,
                ),
            ),
        ),
    )
    time_solver.property("initialstepbdfactive", True)
    time_solver.property("initialstepbdf", _COMSOL_INITIAL_TIMESTEP)
    fully_coupled.property("maxiter", _COMSOL_RETRY_MAXITER)
    fully_coupled.property("minstep", _COMSOL_RETRY_MINSTEP)
    fully_coupled.property("jtech", "onevery")
    direct.property("pivotperturb", _COMSOL_RETRY_PIVOT_PERTURB)


def _collect_comsol_problem_messages(model) -> list[str]:
    """Collect COMSOL problem messages, if available."""

    problem_messages: list[str] = []
    try:
        for problem in model.problems():
            message = str(problem.get("message", "")).strip()
            if message:
                problem_messages.append(message)
    except Exception:
        pass
    return problem_messages


def _remove_comsol_node(container, tag: str | None) -> None:
    """Remove one COMSOL node if it still exists."""

    if not tag:
        return
    try:
        container.java.remove(tag)
    except Exception:
        pass


def _apply_mechanism_to_comsol(model, mechanism: MechanismView) -> None:
    """Enable only the species and reactions present in the reduced mechanism.

    This is the mechanism toggle that step-1 / step-2 workers use internally
    before solving reduced candidates in COMSOL.
    """

    global _cached_applied_mechanism
    feature_nodes = _get_feature_nodes(model)
    kept_species = frozenset(species.name for species in mechanism.species)
    kept_reactions = frozenset(reaction.tag for reaction in mechanism.reactions)

    if _cached_applied_mechanism is None:
        previous_species = frozenset(feature_nodes["species"].keys())
        previous_reactions = frozenset(feature_nodes["reactions"].keys())
    else:
        previous_species, previous_reactions = _cached_applied_mechanism

    for tag in previous_species - kept_species:
        feature_nodes["species"][tag].toggle("disable")
    for tag in previous_reactions - kept_reactions:
        feature_nodes["reactions"][tag].toggle("disable")

    for tag in kept_species - previous_species:
        feature_nodes["species"][tag].toggle("enable")
    for tag in kept_reactions - previous_reactions:
        feature_nodes["reactions"][tag].toggle("enable")

    _cached_applied_mechanism = (kept_species, kept_reactions)


def _get_comsol_dataset(model):
    """Return the solved COMSOL dataset node used for result extraction."""

    return model / "datasets" / _COMSOL_DATASET_NAME


def _dataset_has_solution(model, dataset) -> bool:
    """Return whether the dataset already points to a computed solution."""

    if "solution" in dataset.properties():
        tag = dataset.property("solution")
    elif "data" in dataset.properties():
        tag = dataset.property("data")
    else:
        return False

    for solution in model / "solutions":
        if solution.tag() == tag:
            return not solution.java.isEmpty()
    return False


def _evaluate_comsol_globals(
    model,
    expressions: list[str],
    *,
    dataset,
    outer_index: int | None = None,
) -> np.ndarray:
    """Evaluate global expressions directly through COMSOL's EvalGlobal feature.

    Using ``mph.Model.evaluate()`` is fragile here because it falls back to a
    generic ``Eval`` feature when ``EvalGlobal`` raises, and that fallback can
    request domain context in a 0D/global RE model. This helper stays on the
    global-evaluation path and returns one value trace per expression.
    """

    evaluation = (model / _COMSOL_EVALUATIONS_PATH).create("EvalGlobal")
    try:
        evaluation.property("expr", expressions)
        dataset_tag = dataset.tag() if hasattr(dataset, "tag") else dataset
        evaluation.property("data", dataset_tag)
        if outer_index is not None and JArray is not None and JInt is not None:
            evaluation.java.set("outersolnum", JArray(JInt)([int(outer_index)]))
        elif outer_index is not None:
            evaluation.property("outersolnum", [int(outer_index)])

        java = evaluation.java
        results = np.asarray(java.computeResult(), dtype=float)
        data = results[0]
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data.T
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"COMSOL global evaluation failed: {detail}") from None
    finally:
        evaluation.remove()


def _evaluate_comsol_solution(
    model,
    mechanism: MechanismView,
    outer_index: int,
) -> tuple[list[float], dict[str, list[float]]]:
    """Read one outer parametric solution and convert concentrations to mole fractions."""

    dataset = _get_comsol_dataset(model)
    expressions = [f"re.c_{species.name}" for species in mechanism.species]
    time_values = _evaluate_comsol_globals(model, ["t"], dataset=dataset, outer_index=outer_index)[0]
    concentration_arrays = _evaluate_comsol_globals(model, expressions, dataset=dataset, outer_index=outer_index)
    total_concentration = np.sum(concentration_arrays, axis=0)
    total_concentration[total_concentration <= 0.0] = 1.0
    mole_fractions = concentration_arrays / total_concentration

    result_dict = {
        species.name: list(mole_fractions[index, :])
        for index, species in enumerate(mechanism.species)
    }
    return list(time_values), result_dict


def _evaluate_current_comsol_solution(
    model,
    mechanism: MechanismView,
    dataset=None,
) -> tuple[list[float], dict[str, list[float]]]:
    """Read the currently selected COMSOL solution without outer selection.

    COMSOL returns concentrations; SkelKIN compares mole fractions, so this
    helper normalizes the result before handing it back to ``main.py``.
    """

    if dataset is None:
        dataset = _get_comsol_dataset(model)
    expressions = [f"re.c_{species.name}" for species in mechanism.species]
    time_values = _evaluate_comsol_globals(model, ["t"], dataset=dataset)[0]
    concentration_arrays = _evaluate_comsol_globals(model, expressions, dataset=dataset)
    total_concentration = np.sum(concentration_arrays, axis=0)
    total_concentration[total_concentration <= 0.0] = 1.0
    mole_fractions = concentration_arrays / total_concentration

    result_dict = {
        species.name: list(mole_fractions[index, :])
        for index, species in enumerate(mechanism.species)
    }
    return list(time_values), result_dict


def _condition_signature(condition: cond.ThermalCondition) -> tuple[float, float, float]:
    """Return the embedded-study signature for one standard COMSOL condition."""

    times, temperatures = _condition_temperature_profile(condition)
    start_temperature = temperatures[0]
    end_temperature = temperatures[-1]
    return (_condition_xch4(condition), float(start_temperature), float(end_temperature))


def _parse_study_values(value: str) -> list[float]:
    """Parse one COMSOL parametric list into floats."""

    return [float(item) for item in str(value).split()]


def _saved_study_signatures(model) -> list[tuple[float, float, float]]:
    """Read the saved COMSOL standard sweep from the model."""

    study = (model / _COMSOL_SWEEP_PATH).java
    parameter_names = list(study.getStringArray("pname"))
    parameter_lists = {
        name: _parse_study_values(value)
        for name, value in zip(parameter_names, list(study.getStringArray("plistarr")))
    }
    if not {"xCH4", "Tstart", "Tend"}.issubset(parameter_lists):
        raise RuntimeError("Embedded COMSOL study is missing xCH4/Tstart/Tend.")
    return list(zip(parameter_lists["xCH4"], parameter_lists["Tstart"], parameter_lists["Tend"]))


def _can_use_embedded_standard_study(model, conditions: list[cond.ThermalCondition]) -> bool:
    """Return whether the requested conditions are covered by the embedded study."""

    saved = set(_saved_study_signatures(model))
    return all(_condition_signature(condition) in saved for condition in conditions)


def _open_fresh_comsol_model():
    """Open a fresh COMSOL session/model pair for untouched-study execution."""

    if not MPH_AVAILABLE:
        raise RuntimeError("The mph library is required for COMSOL-backed solves.")

    client = _start_mph_client()
    model = client.load(str(_model_file))
    return client, model


def run_embedded_standard_models(conditions: list[cond.ThermalCondition]):
    """Run the saved COMSOL study unchanged and extract the requested subset."""

    mechanism = _get_model().reduced()
    client, model = _open_fresh_comsol_model()
    try:
        if not _can_use_embedded_standard_study(model, conditions):
            raise RuntimeError("Requested conditions are not covered by the embedded COMSOL study.")

        signature_to_outer = {
            signature: outer_index
            for outer_index, signature in enumerate(_saved_study_signatures(model), start=1)
        }
    finally:
        try:
            client.remove(model)
        except Exception:
            pass

    requested = []
    for condition in conditions:
        signature = _condition_signature(condition)
        outer_index = signature_to_outer[signature]
        outer_client, outer_model = _open_fresh_comsol_model()
        try:
            _configure_comsol_result_plots(outer_model)
            dataset = _get_comsol_dataset(outer_model)
            if not _dataset_has_solution(outer_model, dataset):
                outer_model.solve(_COMSOL_STUDY_NAME)
            requested.append(_evaluate_comsol_solution(outer_model, mechanism, outer_index))
        finally:
            try:
                outer_client.remove(outer_model)
            except Exception:
                pass

    time_values = [times for times, _ in requested]
    result_dicts = [result_dict for _, result_dict in requested]
    return time_values, result_dicts


def _run_mechanism_model_with_history(
    conditions: list[cond.ThermalCondition],
    mechanism: MechanismView,
) -> tuple[list[float], dict[str, list[float]]]:
    """Run several conditions sequentially and return the final one."""

    time_values, result_dicts = _run_comsol_mechanism(conditions, mechanism)
    return time_values[-1], result_dicts[-1]


def run_standard_model_with_history(conditions: list[cond.ThermalCondition]):
    """Run a GUI-order continuation prefix for the full mechanism."""

    mechanism = _get_model().reduced()
    return _run_mechanism_model_with_history(conditions, mechanism)


def _standard_condition_order(conditions: list[cond.ThermalCondition]) -> list[tuple[int, cond.ThermalCondition]]:
    """Return the COMSOL GUI sweep order for the supplied conditions."""

    return sorted(
        list(enumerate(conditions)),
        key=lambda item: (
            0 if item[1].get_temperature()[1][0] < item[1].get_temperature()[1][-1] else 1,
            _condition_xch4(item[1]),
            item[0],
        ),
    )


def _run_mechanism_models_via_continuation(
    conditions: list[cond.ThermalCondition],
    mechanism: MechanismView,
) -> tuple[list[list[float]], list[dict[str, list[float]]]]:
    """Run all conditions independently in sequence on one COMSOL session."""

    return _run_comsol_mechanism(conditions, mechanism)


def run_standard_models_via_continuation(conditions: list[cond.ThermalCondition]):
    """Run all standard conditions by replaying the COMSOL continuation order."""

    mechanism = _get_model().reduced()
    return _run_mechanism_models_via_continuation(conditions, mechanism)


def _run_comsol_mechanism(
    conditions: list[cond.ThermalCondition],
    mechanism: MechanismView,
) -> tuple[list[list[float]], list[dict[str, list[float]]]]:
    """Run all supplied conditions with a persistent worker-local COMSOL runtime.

    This is the central COMSOL execution path used by the standard model and by
    the reduced step-1 / step-2 candidates.
    """

    model = _get_comsol_model()
    time_values: list[list[float]] = []
    result_dicts: list[dict[str, list[float]]] = []

    for condition in conditions:
        _apply_mechanism_to_comsol(model, mechanism)
        _configure_comsol_temperature_function(model, condition)
        _configure_comsol_result_plots(model)
        _apply_comsol_model_parameters(model, _condition_model_parameters(condition))
        solve_plans = _condition_solver_segmentations(condition)
        time_vals: list[float] | None = None
        result_dict: dict[str, list[float]] | None = None

        try:
            runtime = _ensure_comsol_runtime(model, condition)
            _clear_comsol_runtime_solution(runtime)
            time_vals, result_dict = _run_comsol_runtime_segments(model, runtime, mechanism, solve_plans[0])
        except Exception:
            retry_exc: Exception | None = None
            for solve_plan in solve_plans[1:] or solve_plans:
                _apply_mechanism_to_comsol(model, mechanism)
                _configure_comsol_temperature_function(model, condition)
                _configure_comsol_result_plots(model)
                _apply_comsol_model_parameters(model, _condition_model_parameters(condition))
                runtime = _ensure_comsol_runtime(model, condition, recreate=True)
                _configure_comsol_retry_solver(runtime.solution, condition)
                _clear_comsol_runtime_solution(runtime)
                try:
                    time_vals, result_dict = _run_comsol_runtime_segments(model, runtime, mechanism, solve_plan)
                    retry_exc = None
                    break
                except Exception as exc:
                    retry_exc = exc

            if retry_exc is not None:
                problem_messages = _collect_comsol_problem_messages(model)
                detail = f"{type(retry_exc).__name__}: {retry_exc}"
                if problem_messages:
                    detail = detail + " | COMSOL problems: " + " || ".join(problem_messages)
                raise RuntimeError(f"COMSOL solve failed after retry: {detail}") from None

        assert time_vals is not None and result_dict is not None
        time_values.append(time_vals)
        result_dicts.append(result_dict)

    return time_values, result_dicts


def run_standard_models(conditions: list[cond.ThermalCondition]):
    """Run the full COMSOL mechanism for all supplied conditions in one sweep."""

    if len(conditions) > 1:
        return run_standard_models_via_continuation(conditions)

    mechanism = _get_model().reduced()
    return _run_comsol_mechanism(conditions, mechanism)


def run_reduced_species_models(conditions: list[cond.ThermalCondition], omitted_species: list[str]):
    """Run a species-reduced COMSOL sweep for all supplied conditions.

    Step 1 passes a single omitted species. Step 2 passes grouped omissions.
    """

    mechanism = _get_model().reduced(omitted_species=omitted_species)
    if len(conditions) > 1:
        return _run_mechanism_models_via_continuation(conditions, mechanism)
    return _run_comsol_mechanism(conditions, mechanism)


def run_reduced_reactions_models(
    conditions: list[cond.ThermalCondition],
    omitted_species: list[str],
    omitted_reactions: list[str],
):
    """Run a COMSOL sweep reduced by both species and explicit reactions."""

    mechanism = _get_model().reduced(omitted_species=omitted_species, omitted_reactions=omitted_reactions)
    if len(conditions) > 1:
        return _run_mechanism_models_via_continuation(conditions, mechanism)
    return _run_comsol_mechanism(conditions, mechanism)


def run_standard_model(condition: cond.ThermalCondition):
    """Run the full COMSOL-derived mechanism for one condition."""

    times, results = run_standard_models([condition])
    return times[0], results[0]


def run_reduced_species_model(condition: cond.ThermalCondition, omitted_species: list[str]):
    """Run a species-reduced mechanism while keeping all remaining reactions."""

    times, results = run_reduced_species_models([condition], omitted_species)
    return times[0], results[0]


def run_reduced_reactions_model(condition: cond.ThermalCondition, omitted_species: list[str], omitted_reactions: list[str]):
    """Run a mechanism reduced by both species and explicit reaction removal."""

    times, results = run_reduced_reactions_models([condition], omitted_species, omitted_reactions)
    return times[0], results[0]
