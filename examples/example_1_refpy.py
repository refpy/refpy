'''
Example script
'''
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import refpy

def example1():
    """
    Example workflow for pipeline design calculations using refpy.

    This function demonstrates:
    - Importing design data from an Excel file.
    - Calculating pipe properties (corroded wall thickness, total outer diameter).
    - Performing DNV limit state calculations (burst pressure).
    - Calculating pipe-soil interaction (vertical bearing capacity arrays).
    - Performing lateral buckling calculations (friction factor distribution).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the input data and calculated results for each step.
    """

    # Import design data
    df = pd.read_excel(
        'example_1_2.xlsx', sheet_name='Example1', header=None).iloc[1:].transpose()
    df.columns = df.iloc[0].to_numpy()
    df = df[1:].reset_index(drop=True)

    # Example of pipe properties calculation
    pipe = refpy.Pipe(
        outer_diameter = df['Outer Diameter'],
        wall_thickness = df['Wall Thickness'],
        corrosion_allowance = df['Corrosion Allowance'],
    )
    df['Corroded Wall Thickness'] = pipe.wall_thickness_corroded()
    df['Total Outer Diameter'] = pipe.total_outer_diameter()

    # Example of DNV limit state calculation
    dnv = refpy.DNVLimitStates(
        outer_diameter = df['Outer Diameter'],
        corroded_wall_thickness = df['Corroded Wall Thickness'],
        material = df['Material'],
        smys = df['SMYS'],
        smts = df['SMTS'],
        temperature = df['Temperature'],
        material_strength_factor = df['Material Strength Factor']
    )
    df['Burst Pressure'] = dnv.burst_pressure()

    # Example of PSI calculation
    psi = refpy.PSI(
        total_outer_diameter = df['Total Outer Diameter'],
        surface_roughness = df['Surface Roughness'],
        density = df['Density'],
        surcharge_on_seabed = df['Surcharge on Seabed'],
        burial_depth = df['Burial Depth']
    )
    depth_arrays, vertical_bearing_capacity_arrays = psi.downward_undrained_model1()
    df['Depth Arrays'] = list(depth_arrays)
    df['Vertical Bearing Capacity Arrays'] = list(vertical_bearing_capacity_arrays)

    # Example of lateral buckling calculation
    lb = refpy.LBDistributions(
        friction_factor_le = df['Lateral Friction Factor - LE'],
        friction_factor_be = df['Lateral Friction Factor - BE'],
        friction_factor_he = df['Lateral Friction Factor - HE'],
        friction_factor_fit_type = df['Lateral Friction Factor - Fit Type'],
    )
    (
        df['Lateral Friction Factor - Mean'],
        df['Lateral Friction Factor - STD'],
        df['Lateral Friction Factor - Location'],
        df['Lateral Friction Factor - Scale'],
        df['Lateral Friction Factor - LE Fit'],
        df['Lateral Friction Factor - BE Fit'],
        df['Lateral Friction Factor - HE Fit'],
        df['Lateral Friction Factor - RMSE']
    ) = lb.friction_distribution()

    return df

###
### Example 1
###
dfe1 = example1()
dfe1 = dfe1.transpose()
new_cols = [f'Sensitivity{i}' for i in range(1, dfe1.shape[1]+1)]
dfe1.columns = new_cols[:dfe1.shape[1]]
dfe1.to_csv('example_1_output.csv')