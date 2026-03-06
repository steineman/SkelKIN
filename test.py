import warnings
import cantera as ct
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import condition_objects
import numpy as np
import subprocess
import fromchemKIN
import os
import condition_objects as cond

def run_sim(condition, omitted_species):
    # Run reduced simulation
    time_vals, result_dict = fromchemKIN.run_reduced_species_model(
        condition,
        [omitted_species] if isinstance(omitted_species, str) else omitted_species
    )
    return time_vals, result_dict


R111 = ["PC4H9", "NC5H12", "IC4H9", "C6H5CH2C6H5", "NC7H16", "C6H5C2H4C6H5", "CYC6H11", "C10H15", "C6H5C4H9", "FC10H10", "IC16H33", "IC12H25", "IC8H18", "NC12H26", "RDECALIN", "MCYC6", "RODECA", "DECALIN", "NC10H22", "NC10H19", "RMCYC6", "IC12H26", "NC10H20", "C10H16", "IC16H34", "NC16H34", "NC12H25", "NC10H21", "ODECAL", "NC16H33", "C6H5C4H7_3", "DCYC5", "CYC6H10", "NC6H12", "STILB", "RCYC6H9", "CYC6H12", "IC8H16", "RBBENZ", "NC7H14", "CYC5H8", "NEOC5H12", "NEOC5H11", "RC6H5C4H8", "C18H14", "C3H5S", "RC9H11", "LC5H8", "AR", "NC7H13", "DIALLYL", "TMBENZ", "IC8H17", "TETRALIN", "C6H5C2H4", "SC4H9", "B2M2", "B1M3", "NC5H10", "TC4H9", "B1M2", "IC4H10", "DIMEPTD", "NC5H11", "NC5H9_3", "NC7H15", "RXYLENE", "NPBENZ", "RC6H9A", "C6H5CHCH3", "XYLENE", "C8H2", "NC4H10", "C3H5T", "C5H7", "MCPTD", "C5H2", "C6H3", "C6H4C2H3", "CYC6H8", "IC4H8", "CH3C6H4", "C6H5C2H5", "C4H8_2", "C4H8_1", "LC6H6", "IC4H7", "NC3H7", "C", "C10H6CH3", "IC3H7", "C3H8", "C6H4", "C5H3", "C6H5C2H2", "BIPHENYL", "C6H5C2H3", "C4H7_14", "FLUORENE", "C4H7_13", "RTETRALIN", "CYC5H4", "C10H7CH3", "C3H5A", "BENZYNE", "C3H6", "FULVENE", "C12H9", "C14H10", "C7H8", "CH"]

R120 = ["PC4H9", "NC5H12", "IC4H9", "C6H5CH2C6H5", "NC7H16", "C6H5C2H4C6H5", "CYC6H11", "C10H15", "C6H5C4H9", "FC10H10", "IC16H33", "IC12H25", "IC8H18", "NC12H26", "RDECALIN", "MCYC6", "RODECA", "DECALIN", "NC10H22", "NC10H19", "RMCYC6", "IC12H26", "NC10H20", "C10H16", "IC16H34", "NC16H34", "NC12H25", "NC10H21", "ODECAL", "NC16H33", "C6H5C4H7_3", "DCYC5", "CYC6H10", "NC6H12", "STILB", "RCYC6H9", "CYC6H12", "IC8H16", "RBBENZ", "NC7H14", "CYC5H8", "NEOC5H12", "NEOC5H11", "RC6H5C4H8", "C18H14", "C3H5S", "RC9H11", "LC5H8", "AR", "NC7H13", "DIALLYL", "TMBENZ", "IC8H17", "TETRALIN", "C6H5C2H4", "SC4H9", "B2M2", "B1M3", "NC5H10", "TC4H9", "B1M2", "IC4H10", "DIMEPTD", "NC5H11", "NC5H9_3", "NC7H15", "RXYLENE", "NPBENZ", "RC6H9A", "C6H5CHCH3", "XYLENE", "C8H2", "NC4H10", "C3H5T", "C5H7", "MCPTD", "C5H2", "C6H3", "C6H4C2H3", "CYC6H8", "IC4H8", "CH3C6H4", "C6H5C2H5", "C4H8_2", "C4H8_1", "LC6H6", "IC4H7", "NC3H7", "C", "C10H6CH3", "IC3H7", "C3H8", "C6H4", "C5H3", "C6H5C2H2", "BIPHENYL", "C6H5C2H3", "C4H7_14", "FLUORENE", "C4H7_13", "RTETRALIN", "CYC5H4", "C10H7CH3", "C3H5A", "BENZYNE", "C3H6", "FULVENE", "C12H9", "C14H10", "C7H8", "CH", "CH2xS", "C5H6", "C6H2", "C10H10", "C7H5", "INDENYL", "C10H7CH2", "C4H5", "INDENE"]

input_folder = "CRECK_conditions/"

condition_numba = 2

# Stepsize determined via SUNDIALS

conditions_list = condition_objects.TestConditions([])

for filename in os.listdir(input_folder):
    new_condition = condition_objects.load_thermal_condition(input_folder + "/" + filename)
    conditions_list.add_condition(new_condition)

t, d = run_sim(conditions_list.conditions[condition_numba], ["PC4H9"])

print("one_done")

tr111, dr111 = run_sim(conditions_list.conditions[condition_numba], R111)

print("two_done")

tr120, dr120 = run_sim(conditions_list.conditions[condition_numba], R120)

print("three_done")

# ==========================================================
# 5. Plot mole fractions linear
# ==========================================================

plt.figure(figsize=(8, 6))

for sp in d:
    if max(d[sp]) > 1e-4:
        plt.plot(t, d[sp], label=sp)

for sp in dr111:
    if max(dr111[sp]) > 1e-4:
        plt.plot(tr111, dr111[sp], label=sp, linestyle="--")

for sp in dr120:
    if max(dr120[sp]) > 1e-4:
        plt.plot(tr120, dr120[sp], label=sp, linestyle=":")

plt.xlabel("Time [s]")
plt.ylabel("Mole Fraction")
plt.title(f"Condition {condition_numba}")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# 5. Plot mole fractions log
# ==========================================================

plt.figure(figsize=(8, 6))

for sp in d:
    if max(d[sp]) > 1e-4:
        plt.plot(t, d[sp], label=sp)

for sp in dr111:
    if max(dr111[sp]) > 1e-4:
        plt.plot(tr111, dr111[sp], label=sp, linestyle="--")

for sp in dr120:
    if max(dr120[sp]) > 1e-4:
        plt.plot(tr120, dr120[sp], label=sp, linestyle=":")

plt.xlabel("Time [s]")
plt.ylabel("Mole Fraction")
plt.title(f"Condition {condition_numba}")
plt.yscale("log")
plt.ylim([1e-8, 1])
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# 5. Plot mole fractions linear
# ==========================================================

plt.figure(figsize=(8,6))

for sp in d:
    if max(d[sp]) > 1e-4:
        plt.plot(t, d[sp], label=sp, )

for sp in dr111:
    if max(dr111[sp]) > 1e-4:
        plt.plot(tr111, dr111[sp], label=sp, linestyle="--")

for sp in dr120:
    if max(dr120[sp]) > 1e-4:
        plt.plot(tr120, dr120[sp], label=sp, linestyle=":")

plt.xlabel("Time [s]")
plt.ylabel("Mole Fraction")
plt.xscale("log")
plt.title(f"Condition {condition_numba}")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# 5. Plot mole fractions log
# ==========================================================

plt.figure(figsize=(8, 6))

for sp in d:
    if max(d[sp]) > 1e-4:
        plt.plot(t, d[sp], label=sp)

for sp in dr111:
    if max(dr111[sp]) > 1e-4:
        plt.plot(tr111, dr111[sp], label=sp, linestyle="--")

for sp in dr120:
    if max(dr120[sp]) > 1e-4:
        plt.plot(tr120, dr120[sp], label=sp, linestyle=":")

plt.xlabel("Time [s]")
plt.ylabel("Mole Fraction")
plt.xscale("log")
plt.title(f"Condition {condition_numba}")
plt.yscale("log")
plt.ylim([1e-8, 1])
plt.legend()
plt.tight_layout()
plt.show()