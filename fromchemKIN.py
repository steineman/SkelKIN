# Opens files, reads structures, runs models, tosses species, rebuilds reduced files all in ChemKIN (using Cantera)

import warnings
import cantera as ct
from scipy.interpolate import interp1d
import numpy as np
import subprocess
import os
import condition_objects as cond

yaml_file = "model_file.yaml"

relative_tolerance = 1e-4
absolute_tolerance = 1e-6

def load_model_to_yaml(kinetic, thermo):
    cmd = [
        "ck2yaml",
        "--input", kinetic,
        "--thermo", thermo,
        "--output", yaml_file,
    ]

    subprocess.run(cmd, check=True)


def get_species():
    if not os.path.exists(yaml_file):
        warnings.warn("No YAML file.")
    gas = ct.Solution(yaml_file)
    return gas.species_names


def get_reactions():
    if not os.path.exists(yaml_file):
        warnings.warn("No YAML file.")
    gas = ct.Solution(yaml_file)
    return gas.reaction_equations()


def run_standard_model(condition : cond.ThermalCondition):
    if not os.path.exists(yaml_file):
        warnings.warn("No YAML file. moron.")

    gas = ct.Solution(yaml_file)
    gas.TPX = condition.temperature_profile[1][0], ct.one_atm, condition.species
    if condition.pressure_profile != ([], []):
        gas.TPX = condition.temperature_profile[1][0], condition.temperature_profile[1][0], condition.species

    r = ct.IdealGasReactor(gas, energy='off')
    sim = ct.ReactorNet([r])

    t_end = condition.temperature_profile[0][-1]

    # Stepsize determined via SUNDIALS, with min 1000 steps
    sim.max_time_step = t_end / 1000
    sim.rtol = relative_tolerance
    sim.atol = absolute_tolerance

    time = []
    X_history = []
    T_interp = interp1d(condition.temperature_profile[0], condition.temperature_profile[1], bounds_error=False, fill_value='extrapolate')
    if condition.pressure_profile != ([], []):
        p_interp = interp1d(condition.pressure_profile[0], condition.pressure_profile[1], bounds_error=False, fill_value='extrapolate')
    else:
        p_interp = None

    while sim.time < t_end:
        if condition.pressure_profile != ([], []):
            r.thermo.TP = T_interp(sim.time), p_interp(sim.time)
        else:
            r.thermo.TP = T_interp(sim.time), gas.P

        r.syncState()
        sim.step()

        time.append(sim.time)
        X_history.append(r.thermo.X)

    X_history = np.array(X_history)

    return_dict = {}

    species_list = gas.species_names
    for sp in species_list:
        idx = species_list.index(sp)
        return_dict[sp] = list(X_history[:, idx])

    return time, return_dict



def run_reduced_species_model(condition : cond.ThermalCondition, omitted_species : list[str]):
    if not os.path.exists(yaml_file):
        warnings.warn("No YAML file. moron.")

    gas_dummy = ct.Solution(yaml_file)

    # Actual reduction part
    all_species_names = gas_dummy.species_names
    all_species = ct.Species.list_from_file(yaml_file)
    all_reactions = ct.Reaction.list_from_file(yaml_file, gas_dummy)

    temp_species = list(set(all_species_names) - set(omitted_species))

    # keep requested species
    selected_species = [s for s in all_species if s.name in temp_species]

    # Keep requested reactions
    selected_reactions = []

    for rxn in all_reactions:
        reactant_names = rxn.reactants.keys()
        product_names = rxn.products.keys()

        if all(sp in temp_species for sp in reactant_names) and \
                all(sp in temp_species for sp in product_names):
            selected_reactions.append(rxn)

    # create reduced gas object
    gas = ct.Solution(
        thermo='ideal-gas',
        kinetics='gas',
        species=selected_species,
        reactions=selected_reactions
    )

    gas.TPX = condition.temperature_profile[1][0], ct.one_atm, condition.species
    if condition.pressure_profile != ([], []):
        gas.TPX = condition.temperature_profile[1][0], condition.temperature_profile[1][0], condition.species

    r = ct.IdealGasReactor(gas, energy='off')
    sim = ct.ReactorNet([r])

    t_end = condition.temperature_profile[0][-1]

    # Stepsize determined via SUNDIALS, with min 1000 steps
    sim.max_time_step = t_end / 1000
    sim.rtol = relative_tolerance
    sim.atol = absolute_tolerance

    time = []
    X_history = []
    T_interp = interp1d(condition.temperature_profile[0], condition.temperature_profile[1], bounds_error=False, fill_value='extrapolate')
    if condition.pressure_profile != ([], []):
        p_interp = interp1d(condition.pressure_profile[0], condition.pressure_profile[1], bounds_error=False, fill_value='extrapolate')
    else:
        p_interp = None

    while sim.time < t_end:
        if condition.pressure_profile != ([], []):
            r.thermo.TP = T_interp(sim.time), p_interp(sim.time)
        else:
            r.thermo.TP = T_interp(sim.time), gas.P

        r.syncState()
        sim.step()

        time.append(sim.time)
        X_history.append(r.thermo.X)

    X_history = np.array(X_history)

    return_dict = {}

    species_list = gas.species_names
    for sp in species_list:
        idx = species_list.index(sp)
        return_dict[sp] = list(X_history[:, idx])

    return time, return_dict


def run_reduced_reactions_model(condition : cond.ThermalCondition, omitted_species : list[str], omitted_reactions : list[str]):
    if not os.path.exists(yaml_file):
        warnings.warn("No YAML file. moron.")

    gas_dummy = ct.Solution(yaml_file)

    # Actual reduction part
    all_species_names = gas_dummy.species_names
    all_species = ct.Species.list_from_file(yaml_file)
    all_reactions = ct.Reaction.list_from_file(yaml_file, gas_dummy)

    temp_species = list(set(all_species_names) - set(omitted_species))

    # keep requested species
    selected_species = [s for s in all_species if s.name in temp_species]

    # Keep requested reactions
    selected_reactions = []

    for rxn in all_reactions:
        reactant_names = rxn.reactants.keys()
        product_names = rxn.products.keys()

        if all(sp in temp_species for sp in reactant_names) and \
                all(sp in temp_species for sp in product_names):
            if rxn.equation not in omitted_reactions:
                selected_reactions.append(rxn)

    # create reduced gas object
    gas = ct.Solution(
        thermo='ideal-gas',
        kinetics='gas',
        species=selected_species,
        reactions=selected_reactions
    )

    gas.TPX = condition.temperature_profile[1][0], ct.one_atm, condition.species
    if condition.pressure_profile != ([], []):
        gas.TPX = condition.temperature_profile[1][0], condition.temperature_profile[1][0], condition.species

    r = ct.IdealGasReactor(gas, energy='off')
    sim = ct.ReactorNet([r])

    t_end = condition.temperature_profile[0][-1]

    # Stepsize determined via SUNDIALS, with min 1000 steps
    sim.max_time_step = t_end / 1000
    sim.rtol = relative_tolerance
    sim.atol = absolute_tolerance

    time = []
    X_history = []
    T_interp = interp1d(condition.temperature_profile[0], condition.temperature_profile[1], bounds_error=False, fill_value='extrapolate')
    if condition.pressure_profile != ([], []):
        p_interp = interp1d(condition.pressure_profile[0], condition.pressure_profile[1], bounds_error=False, fill_value='extrapolate')
    else:
        p_interp = None

    while sim.time < t_end:
        if condition.pressure_profile != ([], []):
            r.thermo.TP = T_interp(sim.time), p_interp(sim.time)
        else:
            r.thermo.TP = T_interp(sim.time), gas.P

        r.syncState()
        sim.step()

        time.append(sim.time)
        X_history.append(r.thermo.X)

    X_history = np.array(X_history)

    return_dict = {}

    species_list = gas.species_names
    for sp in species_list:
        idx = species_list.index(sp)
        return_dict[sp] = list(X_history[:, idx])

    return time, return_dict
