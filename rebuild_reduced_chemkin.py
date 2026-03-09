import subprocess
import cantera as ct

chemkin_kinet = "CRECK_2003_TOT_HT_SOOT_pyro_COMSOL.CKI.txt"
chemkin_thermo = "CRECK_2003_TOT_HT_SOOT_therm_pyro_COMSOL.CKT.txt"

omitted = ["PC4H9", "NC5H12", "IC4H9", "C6H5CH2C6H5", "NC7H16", "C6H5C2H4C6H5", "CYC6H11", "C10H15", "C6H5C4H9", "FC10H10", "IC16H33", "IC12H25", "IC8H18", "NC12H26", "RDECALIN", "MCYC6", "RODECA", "DECALIN", "NC10H22", "NC10H19", "RMCYC6", "IC12H26", "NC10H20", "C10H16", "IC16H34", "NC16H34", "NC12H25", "NC10H21", "ODECAL", "NC16H33", "C6H5C4H7_3", "DCYC5", "CYC6H10", "NC6H12", "STILB", "RCYC6H9", "CYC6H12", "IC8H16", "RBBENZ", "NC7H14", "CYC5H8", "NEOC5H12", "NEOC5H11", "RC6H5C4H8", "C18H14", "C3H5S", "RC9H11", "LC5H8", "AR", "NC7H13", "DIALLYL", "TMBENZ", "IC8H17", "TETRALIN", "C6H5C2H4", "SC4H9", "B2M2", "B1M3", "NC5H10", "TC4H9", "B1M2", "IC4H10", "DIMEPTD", "NC5H11", "NC5H9_3", "NC7H15", "RXYLENE", "NPBENZ", "RC6H9A", "C6H5CHCH3", "XYLENE", "C8H2", "NC4H10", "C3H5T", "C5H7", "MCPTD", "C5H2", "C6H3", "C6H4C2H3", "CYC6H8", "IC4H8", "CH3C6H4", "C6H5C2H5", "C4H8_2", "C4H8_1", "LC6H6", "IC4H7", "NC3H7", "C10H6CH3", "IC3H7", "C6H4", "C5H3", "C6H5C2H2", "BIPHENYL", "C6H5C2H3", "C4H7_14", "FLUORENE", "C4H7_13", "RTETRALIN", "CYC5H4", "C10H7CH3", "C3H5A", "BENZYNE", "C3H6", "FULVENE", "C12H9", "C14H10", "C7H8", "CH2xS", "C5H6", "C6H2", "C10H10", "C7H5", "INDENYL", "C10H7CH2", "C4H5", "INDENE"]

temp_yaml = "steins_trashcan/temp_yaml.yaml"

cmd = [
        "ck2yaml",
        "--input", chemkin_kinet,
        "--thermo", chemkin_thermo,
        "--output", temp_yaml,
    ]

subprocess.run(cmd, check=True)

gas_dummy = ct.Solution(temp_yaml)

all_species_names = gas_dummy.species_names
all_species = ct.Species.list_from_file(temp_yaml)
all_reactions = ct.Reaction.list_from_file(temp_yaml, gas_dummy)

temp_species = list(set(all_species_names) - set(omitted))

len_temp = len(temp_species)

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

gas.write_yaml("steins_trashcan/reduced.yaml")

cmd2 = [
        "yaml2ck",
        "--mechanism", f"steins_trashcan/reduced_creck_{len_temp}.ck",
        "--thermo", f"steins_trashcan/reduced_creck_thermo_{len_temp}.dat",
        "--overwrite",
        "steins_trashcan/reduced.yaml",
    ]

subprocess.run(cmd2, check=True)