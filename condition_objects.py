class ThermalCondition:
    def __init__(self, species: dict[str, float], temperature_profile: tuple[list[float], list[float]],
                 pressure_profile: tuple[list[float], list[float]] = ()):
        """
        species: {"O2": 0.21, "N2": 0.79, ...} float = molar fraction
        temperature_profile: (time, temperature)
        """
        self.species = species
        self.temperature_profile = temperature_profile
        self.pressure_profile = pressure_profile

    def get_species(self):
        return self.species

    def get_molar_fraction(self, name: str) -> float:
        return self.species.get(name)

    def add_species(self, name: str, molar_fraction: float):
        self.species[name] = molar_fraction

    def set_temperature(self, time: list[float], temperature: list[float]):
        self.temperature_profile = (time, temperature)

    def set_pressure(self, time: list[float], pressure: list[float]):
        self.pressure_profile = (time, pressure)

    def get_temperature(self):
        return self.temperature_profile

    def get_pressure(self):
        return self.pressure_profile


class PlasmaCondition:
    def __init__(self, species: dict[str, float], temperature_profile: tuple[list[float], list[float]],
                 power_density: tuple[list[float], list[float]], pressure_profile: tuple[list[float], list[float]] = ()):
        """
        species: {"O2": 0.21, "N2": 0.79, ...} float = molar fraction
        temperature_profile: (time, temperature)
        power_density: (time, power_density)
        """
        self.species = species
        self.temperature_profile = temperature_profile
        self.power_density = power_density
        self.pressure_profile = pressure_profile

    def get_species(self):
        return self.species

    def get_molar_fraction(self, name: str) -> float:
        return self.species.get(name)

    def add_species(self, name: str, molar_fraction: float):
        self.species[name] = molar_fraction

    def set_temperature(self, time: list[float], temperature: list[float]):
        self.temperature_profile = (time, temperature)

    def set_power_density(self, time: list[float], power_density: list[float]):
        self.power_density = (time, power_density)

    def set_pressure(self, time: list[float], pressure: list[float]):
        self.pressure_profile = (time, pressure)

    def get_temperature(self):
        return self.temperature_profile

    def get_power_density(self):
        return self.power_density

    def get_pressure(self):
        return self.pressure_profile


class TestConditions:
    def __init__(self, conditions: list[ThermalCondition | PlasmaCondition]):
        """
        conditions: list[ThermalCondition | PlasmaCondition]
        """
        self.conditions = conditions

    def get_condition(self, idx: int) -> ThermalCondition | PlasmaCondition:
        return self.conditions[idx]

    def add_condition(self, condition: ThermalCondition | PlasmaCondition):
        self.conditions.append(condition)

    def get_conditions(self) -> list[ThermalCondition | PlasmaCondition]:
        return self.conditions


class ItemError:
    def __init__(self, item: str | tuple[str], value: float, max_value: float, weight: float = 1):
        """
        item: "CO2" | "CO2 + O <=> CO + O2" | tuple["CO2", "CH4"] | tuple["CO2 + O <=> CO + O2", "..."]
        value: float
        max_value: float
        weight: float
        """
        self.item = item
        self.value = value
        self.max_value = max_value
        self.weight = weight

    def get_item(self) -> str | tuple[str]:
        return self.item

    def get_value(self) -> float:
        return self.value

    def set_item(self, item: str | tuple[str]):
        self.item = item

    def reset_value(self, value: float):
        self.value = value

    def reset_max_value(self, max_value: float):
        self.max_value = max_value

    def confirm_max_value(self, new_max_value: float):
        self.max_value = max(new_max_value, self.max_value)

    def add_to_value(self, added_value: float, added_weight: float = 1):
        new_value = (self.value * self.weight + added_value * added_weight) / (self.weight + added_weight)
        self.max_value = max(added_value, self.max_value)
        self.weight = self.weight + added_weight
        self.value = new_value


class ItemErrorList:
    def __init__(self, items: list[ItemError]):
        """
        items: list[ItemError]
        """
        self.items = items

    def get_items(self) -> list[ItemError]:
        return self.items

    def add_to_list(self, item: ItemError):
        self.items.append(item)

    def reset_items(self):
        self.items = []

    def sort_items(self):
        items_sorted = sorted(self.items, key=lambda item: item.get_value(), reverse=True)
        self.items = items_sorted


def load_thermal_condition(file_name: str) -> ThermalCondition:
    with open(file_name, "r") as f:
        lines = f.readlines()
    species = {}
    temp_T_time = []
    temp_T_temp = []
    temp_p_time = []
    temp_p_p = []
    currently_reading = "NONE"
    for line in lines:
        if line.startswith("INITIAL MOLAR FRACTION"):
            currently_reading = "MOLAR_FRACTION"
            continue
        elif line.startswith("TEMPERATURE"):
            currently_reading = "TEMPERATURE"
            continue
        elif line.startswith("PRESSURE"):
            currently_reading = "PRESSURE"
            continue
        elif line.startswith("END"):
            currently_reading = "NONE"

        # read stuff
        if currently_reading == "MOLAR_FRACTION":
            species[line.split()[0]] = float(line.split()[1])

        if currently_reading == "TEMPERATURE":
            temp_T_time.append(float(line.split()[0]))
            temp_T_temp.append(float(line.split()[1]))

        if currently_reading == "PRESSURE":
            temp_p_time.append(float(line.split()[0]))
            temp_p_p.append(float(line.split()[1]))

    return ThermalCondition(species, (temp_T_time, temp_T_temp), (temp_p_time, temp_p_p))


def load_plasma_condition(file_name: str) -> PlasmaCondition:
    with open(file_name, "r") as f:
        lines = f.readlines()
    species = {}
    temp_T_time = []
    temp_T_temp = []
    temp_Pd_time = []
    temp_Pd_Pd = []
    temp_p_time = []
    temp_p_p = []
    currently_reading = "NONE"
    for line in lines:
        if line.startswith("INITIAL MOLAR FRACTION"):
            currently_reading = "MOLAR_FRACTION"
            continue
        elif line.startswith("TEMPERATURE"):
            currently_reading = "TEMPERATURE"
            continue
        elif line.startswith("PRESSURE"):
            currently_reading = "PRESSURE"
            continue
        elif line.startswith("POWER DENSITY"):
            currently_reading = "POWER_DENSITY"
            continue
        elif line.startswith("END"):
            currently_reading = "NONE"

        # read stuff
        if currently_reading == "MOLAR_FRACTION":
            species[line.split()[0]] = float(line.split()[1])

        if currently_reading == "TEMPERATURE":
            temp_T_time.append(float(line.split()[0]))
            temp_T_temp.append(float(line.split()[1]))

        if currently_reading == "POWER DENSITY":
            temp_Pd_time.append(float(line.split()[0]))
            temp_Pd_Pd.append(float(line.split()[1]))

        if currently_reading == "PRESSURE":
            temp_p_time.append(float(line.split()[0]))
            temp_p_p.append(float(line.split()[1]))

    return PlasmaCondition(species, (temp_T_time, temp_T_temp), (temp_Pd_time, temp_Pd_Pd), (temp_p_time, temp_p_p))