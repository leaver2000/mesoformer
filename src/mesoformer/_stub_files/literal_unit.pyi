import typing

LiteralUnit = typing.Literal[
    "ppm",
    "stattesla",
    "tesla",
    "enzyme_unit",
    "langley",
    "point",
    "joule",
    "radian",
    "abhenry",
    "eulers_number",
    "apothecary_ounce",
    "metric_horsepower",
    "octave",
    "kip_per_square_inch",
    "troy_pound",
    "planck_current",
    "entropy_unit",
    "square_league",
    "sievert",
    "candela",
    "conventional_ampere_90",
    "molar_gas_constant",
    "US_international_ampere",
    "baud",
    "day",
    "foot_per_second",
    "gram",
    "sidereal_year",
    "inch_Hg_60F",
    "unit_pole",
    "mole",
    "hour",
    "denier",
    "square_yard",
    "pixel",
    "meter_per_second",
    "molar",
    "force_kilogram",
    "eon",
    "atomic_unit_of_intensity",
    "bits_per_pixel",
    "millennium",
    "number_meter",
    "townsend",
    "conventional_henry_90",
    "atomic_unit_of_current",
    "furlong",
    "liter",
    "chain",
    "teaspoon",
    "imperial_fluid_scruple",
    "x_unit_Mo",
    "hundredweight",
    "dry_barrel",
    "stere",
    "atomic_mass_constant",
    "board_foot",
    "debye",
    "aberdeen",
    "neutron_mass",
    "lux",
    "force_long_ton",
    "abampere",
    "darcy",
    "tex_point",
    "rads",
    "tex_didot",
    "bag",
    "mean_international_ohm",
    "atomic_unit_of_electric_field",
    "avogadro_number",
    "mile",
    "inch",
    "newtonian_constant_of_gravitation",
    "abvolt",
    "electrical_horsepower",
    "degree",
    "curie",
    "water",
    "wien_frequency_displacement_law_constant",
    "cubic_yard",
    "bushel",
    "technical_atmosphere",
    "slug",
    "boltzmann_constant",
    "hartree",
    "fluid_ounce",
    "dry_pint",
    "classical_electron_radius",
    "gill",
    "franklin",
    "dry_quart",
    "imperial_gill",
    "gregorian_year",
    "first_radiation_constant",
    "wien_x",
    "count",
    "mil",
    "decibelmilliwatt",
    "von_klitzing_constant",
    "gauss",
    "decibel",
    "micron",
    "decade",
    "pica",
    "lumen",
    "pixels_per_centimeter",
    "biot_turn",
    "statweber",
    "common_year",
    "biot",
    "conventional_von_klitzing_constant",
    "wien_wavelength_displacement_law_constant",
    "gallon",
    "maxwell",
    "second",
    "atomic_unit_of_temperature",
    "scruple",
    "atomic_unit_of_force",
    "month",
    "K_alpha_Mo_d_220",
    "light_year",
    "grain",
    "US_international_volt",
    "cicero",
    "mean_international_ampere",
    "survey_foot",
    "tex",
    "watt",
    "square_inch",
    "imperial_fluid_ounce",
    "ohm",
    "ampere_turn",
    "planck_length",
    "statvolt",
    "particle",
    "becquerel",
    "conventional_farad_90",
    "grade",
    "bohr",
    "slinch",
    "horsepower",
    "roentgen",
    "josephson_constant",
    "imperial_peck",
    "hertz",
    "lambda",
    "international_british_thermal_unit",
    "pound",
    "K_alpha_W_d_220",
    "tansec",
    "poundal",
    "sidereal_month",
    "tex_cicero",
    "coulomb_constant",
    "planck_time",
    "kip",
    "conventional_josephson_constant",
    "foot",
    "dirac_constant",
    "K_alpha_Cu_d_220",
    "statohm",
    "planck_temperature",
    "sound_pressure_level",
    "erg",
    "dalton",
    "pixels_per_inch",
    "torr",
    "watt_hour",
    "faraday_constant",
    "therm",
    "square_rod",
    "RKM",
    "barrel",
    "nuclear_magneton",
    "revolutions_per_minute",
    "fifteen_degree_calorie",
    "standard_gravity",
    "degree_Celsius",
    "angstrom",
    "poise",
    "force_gram",
    "fifth",
    "apothecary_dram",
    "tex_pica",
    "shake",
    "stilb",
    "avogadro_constant",
    "mean_international_volt",
    "speed_of_light",
    "peak_sun_hour",
    "square_foot",
    "bit",
    "tablespoon",
    "acre_foot",
    "elementary_charge",
    "revolutions_per_second",
    "electron_g_factor",
    "beer_barrel",
    "pound_force_per_square_inch",
    "hectare",
    "force_pound",
    "imperial_pint",
    "siemens",
    "standard_atmosphere",
    "second_radiation_constant",
    "stefan_boltzmann_constant",
    "US_therm",
    "mile_per_hour",
    "statmho",
    "conventional_watt_90",
    "ln10",
    "meter",
    "shot",
    "international_calorie",
    "cubic_inch",
    "cubic_centimeter",
    "pi",
    "ounce",
    "sverdrup",
    "ampere_hour",
    "pint",
    "nautical_mile",
    "imperial_barrel",
    "css_pixel",
    "kilometer_per_hour",
    "force_ton",
    "thermochemical_british_thermal_unit",
    "conventional_volt_90",
    "neper",
    "oil_barrel",
    "bar",
    "stone",
    "inch_Hg",
    "cubic_foot",
    "square_mile",
    "katal",
    "gamma",
    "mercury_60F",
    "newton",
    "decibelmicrowatt",
    "farad",
    "quadrillion_Btu",
    "rydberg",
    "quart",
    "stokes",
    "rhe",
    "imperial_gallon",
    "centimeter_Hg",
    "gray",
    "stathenry",
    "lattice_spacing_of_Si",
    "are",
    "hand",
    "hogshead",
    "volt",
    "abohm",
    "parsec",
    "US_hundredweight",
    "mercury",
    "centimeter_H2O",
    "league",
    "counts_per_second",
    "volt_ampere",
    "imperial_cup",
    "year",
    "wien_u",
    "arcsecond",
    "carat",
    "UK_hundredweight",
    "buckingham",
    "reyn",
    "electron_volt",
    "fluid_dram",
    "didot",
    "statampere",
    "imperial_bushel",
    "turn",
    "UK_force_ton",
    "US_force_ton",
    "unified_atomic_mass_unit",
    "fine_structure_constant",
    "quarter",
    "henry",
    "reciprocal_centimeter",
    "tropical_month",
    "british_thermal_unit",
    "atomic_unit_of_time",
    "vacuum_permeability",
    "circular_mil",
    "oersted",
    "fortnight",
    "survey_mile",
    "rydberg_constant",
    "impedance_of_free_space",
    "lambert",
    "calorie",
    "inch_H2O_60F",
    "water_60F",
    "abcoulomb",
    "long_ton",
    "water_39F",
    "apothecary_pound",
    "imperial_fluid_drachm",
    "statfarad",
    "synodic_month",
    "kelvin",
    "pascal",
    "conventional_coulomb_90",
    "kilometer_per_second",
    "foot_H2O",
    "week",
    "cup",
    "dyne",
    "scaled_point",
    "faraday",
    "milliarcsecond",
    "bohr_magneton",
    "zeta",
    "link",
    "dtex",
    "x_unit_Cu",
    "rem",
    "atmosphere_liter",
    "number_english",
    "century",
    "imperial_quart",
    "proton_mass",
    "UK_ton",
    "dram",
    "steradian",
    "absiemens",
    "magnetic_flux_quantum",
    "knot",
    "standard_liter_per_minute",
    "metric_ton",
    "force_metric_ton",
    "dry_gallon",
    "sidereal_day",
    "degree_Rankine",
    "force_ounce",
    "conventional_ohm_90",
    "cables_length",
    "conductance_quantum",
    "inch_H2O_39F",
    "barye",
    "degree_Fahrenheit",
    "imperial_minim",
    "clausius",
    "abfarad",
    "minim",
    "nit",
    "fermi",
    "electron_mass",
    "US_international_ohm",
    "thomson_cross_section",
    "millimeter_Hg",
    "peck",
    "fathom",
    "byte",
    "square_survey_mile",
    "rod",
    "planck_mass",
    "refrigeration_ton",
    "barn",
    "pennyweight",
    "square_degree",
    "weber",
    "leap_year",
    "svedberg",
    "minute",
    "boiler_horsepower",
    "gilbert",
    "angstrom_star",
    "astronomical_unit",
    "thou",
    "long_hundredweight",
    "galileo",
    "yard",
    "ton",
    "tonne_of_oil_equivalent",
    "acre",
    "coulomb",
    "tropical_year",
    "ampere",
    "ton_TNT",
    "vacuum_permittivity",
    "foot_pound",
    "planck_constant",
    "jute",
    "US_ton",
    "cooling_tower_ton",
    "arcminute",
    "gamma_mass",
    "rutherford",
    "degree_Reaumur",
    "troy_ounce",
    "percent",
    "foot",
    "cubic_inch",
    "cubic_yard",
    "thou",
    "square_yard",
    "square_foot",
    "square_mile",
    "cubic_foot",
    "mile",
    "square_inch",
    "yard",
    "circular_mil",
    "hand",
    "inch",
    "square_league",
    "square_rod",
    "link",
    "survey_foot",
    "furlong",
    "cables_length",
    "fathom",
    "square_survey_mile",
    "rod",
    "chain",
    "league",
    "acre",
    "acre_foot",
    "survey_mile",
    "dry_gallon",
    "board_foot",
    "dry_quart",
    "dry_barrel",
    "peck",
    "dry_pint",
    "bushel",
    "quart",
    "minim",
    "pint",
    "gallon",
    "fluid_ounce",
    "fluid_dram",
    "gill",
    "fifth",
    "hogshead",
    "oil_barrel",
    "beer_barrel",
    "shot",
    "barrel",
    "teaspoon",
    "tablespoon",
    "cup",
    "slinch",
    "quarter",
    "force_ounce",
    "stone",
    "long_hundredweight",
    "slug",
    "force_long_ton",
    "ton",
    "dram",
    "pound",
    "ounce",
    "bag",
    "long_ton",
    "poundal",
    "kip",
    "force_pound",
    "force_ton",
    "hundredweight",
    "UK_force_ton",
    "slinch",
    "quarter",
    "force_ounce",
    "stone",
    "long_hundredweight",
    "slug",
    "force_long_ton",
    "ton",
    "UK_ton",
    "dram",
    "pound",
    "ounce",
    "bag",
    "long_ton",
    "poundal",
    "UK_hundredweight",
    "kip",
    "force_pound",
    "force_ton",
    "hundredweight",
    "slinch",
    "US_force_ton",
    "quarter",
    "force_ounce",
    "stone",
    "long_hundredweight",
    "slug",
    "force_long_ton",
    "US_hundredweight",
    "ton",
    "dram",
    "pound",
    "ounce",
    "bag",
    "long_ton",
    "US_ton",
    "poundal",
    "kip",
    "force_pound",
    "force_ton",
    "hundredweight",
    "troy_ounce",
    "troy_pound",
    "pennyweight",
    "apothecary_ounce",
    "apothecary_pound",
    "scruple",
    "apothecary_dram",
    "imperial_quart",
    "imperial_bushel",
    "imperial_gill",
    "imperial_cup",
    "imperial_fluid_drachm",
    "imperial_minim",
    "imperial_gallon",
    "imperial_barrel",
    "imperial_peck",
    "imperial_fluid_ounce",
    "imperial_fluid_scruple",
    "imperial_pint",
    "pica",
    "tex_point",
    "scaled_point",
    "tex_didot",
    "pixels_per_centimeter",
    "cicero",
    "point",
    "tex_cicero",
    "pixel",
    "pixels_per_inch",
    "bits_per_pixel",
    "css_pixel",
    "tex_pica",
    "didot",
    "jute",
    "denier",
    "dtex",
    "aberdeen",
    "tex",
    "RKM",
    "number_english",
    "number_meter",
    "statampere",
    "franklin",
    "statfarad",
    "statohm",
    "statmho",
    "maxwell",
    "gauss",
    "oersted",
    "statvolt",
    "statampere",
    "stattesla",
    "franklin",
    "statfarad",
    "statohm",
    "statmho",
    "statweber",
    "stathenry",
    "maxwell",
    "gauss",
    "oersted",
    "statvolt",
    "ppm",
    "tesla",
    "enzyme_unit",
    "langley",
    "joule",
    "radian",
    "abhenry",
    "eulers_number",
    "metric_horsepower",
    "octave",
    "kip_per_square_inch",
    "planck_current",
    "entropy_unit",
    "sievert",
    "candela",
    "conventional_ampere_90",
    "molar_gas_constant",
    "US_international_ampere",
    "baud",
    "day",
    "foot_per_second",
    "gram",
    "sidereal_year",
    "inch_Hg_60F",
    "unit_pole",
    "mole",
    "hour",
    "meter_per_second",
    "molar",
    "force_kilogram",
    "eon",
    "atomic_unit_of_intensity",
    "millennium",
    "townsend",
    "conventional_henry_90",
    "atomic_unit_of_current",
    "liter",
    "x_unit_Mo",
    "stere",
    "atomic_mass_constant",
    "debye",
    "neutron_mass",
    "lux",
    "abampere",
    "darcy",
    "rads",
    "mean_international_ohm",
    "atomic_unit_of_electric_field",
    "avogadro_number",
    "newtonian_constant_of_gravitation",
    "abvolt",
    "electrical_horsepower",
    "degree",
    "curie",
    "water",
    "wien_frequency_displacement_law_constant",
    "technical_atmosphere",
    "boltzmann_constant",
    "hartree",
    "classical_electron_radius",
    "gregorian_year",
    "first_radiation_constant",
    "wien_x",
    "count",
    "mil",
    "decibelmilliwatt",
    "von_klitzing_constant",
    "decibel",
    "micron",
    "decade",
    "lumen",
    "biot_turn",
    "common_year",
    "biot",
    "conventional_von_klitzing_constant",
    "wien_wavelength_displacement_law_constant",
    "second",
    "atomic_unit_of_temperature",
    "atomic_unit_of_force",
    "month",
    "K_alpha_Mo_d_220",
    "light_year",
    "grain",
    "US_international_volt",
    "mean_international_ampere",
    "watt",
    "ohm",
    "ampere_turn",
    "planck_length",
    "grade",
    "particle",
    "becquerel",
    "conventional_farad_90",
    "bohr",
    "horsepower",
    "roentgen",
    "josephson_constant",
    "hertz",
    "lambda",
    "international_british_thermal_unit",
    "K_alpha_W_d_220",
    "tansec",
    "sidereal_month",
    "coulomb_constant",
    "planck_time",
    "conventional_josephson_constant",
    "dirac_constant",
    "K_alpha_Cu_d_220",
    "planck_temperature",
    "sound_pressure_level",
    "erg",
    "dalton",
    "torr",
    "watt_hour",
    "faraday_constant",
    "therm",
    "nuclear_magneton",
    "revolutions_per_minute",
    "fifteen_degree_calorie",
    "standard_gravity",
    "degree_Celsius",
    "angstrom",
    "poise",
    "force_gram",
    "shake",
    "stilb",
    "avogadro_constant",
    "mean_international_volt",
    "speed_of_light",
    "peak_sun_hour",
    "bit",
    "elementary_charge",
    "revolutions_per_second",
    "electron_g_factor",
    "pound_force_per_square_inch",
    "hectare",
    "siemens",
    "standard_atmosphere",
    "second_radiation_constant",
    "stefan_boltzmann_constant",
    "US_therm",
    "mile_per_hour",
    "conventional_watt_90",
    "ln10",
    "meter",
    "international_calorie",
    "cubic_centimeter",
    "pi",
    "sverdrup",
    "ampere_hour",
    "nautical_mile",
    "kilometer_per_hour",
    "thermochemical_british_thermal_unit",
    "conventional_volt_90",
    "neper",
    "bar",
    "inch_Hg",
    "katal",
    "gamma",
    "mercury_60F",
    "newton",
    "decibelmicrowatt",
    "farad",
    "quadrillion_Btu",
    "rydberg",
    "stokes",
    "rhe",
    "centimeter_Hg",
    "gray",
    "lattice_spacing_of_Si",
    "are",
    "volt",
    "abohm",
    "parsec",
    "mercury",
    "centimeter_H2O",
    "counts_per_second",
    "volt_ampere",
    "year",
    "wien_u",
    "arcsecond",
    "carat",
    "reyn",
    "buckingham",
    "electron_volt",
    "turn",
    "fine_structure_constant",
    "unified_atomic_mass_unit",
    "henry",
    "reciprocal_centimeter",
    "tropical_month",
    "british_thermal_unit",
    "atomic_unit_of_time",
    "vacuum_permeability",
    "fortnight",
    "rydberg_constant",
    "impedance_of_free_space",
    "lambert",
    "calorie",
    "inch_H2O_60F",
    "water_60F",
    "abcoulomb",
    "water_39F",
    "synodic_month",
    "kelvin",
    "pascal",
    "conventional_coulomb_90",
    "kilometer_per_second",
    "foot_H2O",
    "week",
    "dyne",
    "faraday",
    "milliarcsecond",
    "bohr_magneton",
    "zeta",
    "x_unit_Cu",
    "rem",
    "atmosphere_liter",
    "century",
    "proton_mass",
    "steradian",
    "absiemens",
    "magnetic_flux_quantum",
    "knot",
    "standard_liter_per_minute",
    "metric_ton",
    "force_metric_ton",
    "sidereal_day",
    "degree_Rankine",
    "conventional_ohm_90",
    "conductance_quantum",
    "inch_H2O_39F",
    "barye",
    "degree_Fahrenheit",
    "clausius",
    "abfarad",
    "nit",
    "fermi",
    "electron_mass",
    "US_international_ohm",
    "thomson_cross_section",
    "millimeter_Hg",
    "byte",
    "planck_mass",
    "refrigeration_ton",
    "barn",
    "square_degree",
    "weber",
    "leap_year",
    "svedberg",
    "minute",
    "boiler_horsepower",
    "gilbert",
    "angstrom_star",
    "astronomical_unit",
    "galileo",
    "tonne_of_oil_equivalent",
    "tropical_year",
    "coulomb",
    "ampere",
    "ton_TNT",
    "vacuum_permittivity",
    "foot_pound",
    "planck_constant",
    "cooling_tower_ton",
    "arcminute",
    "gamma_mass",
    "rutherford",
    "degree_Reaumur",
    "percent",
]