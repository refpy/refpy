'''
Example script
'''
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        'example_refpy.xlsx', sheet_name='Example1', header=None).iloc[1:].transpose()
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
    psi = refpy.PipeSoilInteraction(
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
    lb = refpy.LateralBuckling(
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

def example2_data():
    """
    Example workflow for OOS pipeline survey and route data processing using refpy.

    This function demonstrates:
    - Importing route and survey data from Excel files.
    - Initializing the OOSAnonymisation class with route and survey data.
    - Running the pipeline section processing workflow.
    - Adding section and curve radius results to the survey DataFrame.
    - Creating a DataFrame with group section and modified coordinates.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame containing the survey data with modified coordinates and curve radius results.
    df2 : pandas.DataFrame
        DataFrame containing group section and modified coordinates for each survey point.
    """

    # Import design data
    dfr = pd.read_excel('example_refpy.xlsx', sheet_name='Example2-Route')
    dfs = pd.read_excel('example_refpy.xlsx', sheet_name='Example2-Survey')
    fft_cutoff_wavelength = 20.0  # FFT cutoff wavelength in meters
    gaussian_bandwidth = 4.0  # Bandwidth for Gaussian smoothing

    # Initialise OOS anonymisation class
    oos = refpy.OOSAnonymisation(
        route_development=dfr['Development'],
        survey_type=dfs['Survey Type'],
        route_pipeline_group=dfr['Pipeline Group'],
        route_pipeline=dfr['Pipeline'],
        route_kp_from=dfr['KP From'],
        route_kp_to=dfr['KP To'],
        route_pipeline_section_type=dfr['Pipeline Section - Type'],
        route_curve_radius=dfr['Route Curve Radius'],
        survey_development=dfs['Development'],
        survey_pipeline_group=dfs['Pipeline Group'],
        survey_pipeline=dfs['Pipeline'],
        survey_kp=dfs['KP'],
        survey_easting=dfs['Easting'],
        survey_northing=dfs['Northing']
    )

    # Full survey (not anonymised)
    oos.process(anonymise=False)
    df1 = pd.DataFrame({
        'Development': oos.survey_development,
        'Survey Type': oos.survey_type,
        'Pipeline Group': oos.survey_pipeline_group,
        'Pipeline': oos.survey_pipeline,
        'Section Type': oos.survey_section_type,
        'Section No': oos.survey_section_no,
        'KP Mod': oos.survey_section_kp_mod,
        'Easting Mod': oos.survey_section_easting_mod,
        'Northing Mod': oos.survey_section_northing_mod,
        'Features': oos.survey_feature,
        'Design Route Curve Radius': oos.survey_design_route_curve_radius,
        'Actual Route Curve Radius': oos.survey_actual_route_curve_radius
    })

    # Anonymised survey
    oos.process(anonymise=True)
    df2 = pd.DataFrame({
        'Development': oos.survey_development,
        'Survey Type': oos.survey_type,
        'Pipeline Group': oos.survey_pipeline_group,
        'Pipeline': oos.survey_pipeline,
        'Section Type': oos.survey_section_type,
        'Section No': oos.survey_section_no,
        'KP Mod': oos.survey_section_kp_mod,
        'Easting Mod': oos.survey_section_easting_mod,
        'Northing Mod': oos.survey_section_northing_mod,
        'Features': oos.survey_feature,
        'Design Route Curve Radius': oos.survey_design_route_curve_radius,
        'Actual Route Curve Radius': oos.survey_actual_route_curve_radius,
        'Group Section Type': oos.survey_group_section_type,
        'Group Section No': oos.survey_group_section_no,
        'Group KP Mod': oos.survey_group_section_kp_mod,
        'Group Easting Mod': oos.survey_group_section_easting_mod,
        'Group Northing Mod': oos.survey_group_section_northing_mod
    })

    # Initialise OOS smoothing class
    oos_smooth = refpy.OOSSmoother(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        group_section_number = df2['Group Section No'],
        x = df2['Group Easting Mod'],
        y = df2['Group Northing Mod'],
        fft_cutoff_wl = fft_cutoff_wavelength,
        gaussian_bandwidth = gaussian_bandwidth
    )
    #
    oos_smooth.gaussian_filter()
    df2['Group Gaussian Smooth Northing Mod'] = oos_smooth.y_smooth_gaussian
    #
    df2['FFT Cutoff Wavelength'] = fft_cutoff_wavelength
    oos_smooth.fft_filter()
    df2['Group FFT Smooth Northing Mod'] = oos_smooth.y_smooth
    df2['Group FFT Smooth Frequencies - Raw'] = oos_smooth.freqs_raw
    df2['Group FFT Smooth Spectrum - Raw'] = oos_smooth.fft_raw
    df2['Group FFT Smooth Frequencies - Filtered'] = oos_smooth.freqs
    df2['Group FFT Smooth Spectrum - Filtered'] = oos_smooth.fft
    #
    oos_smooth.fft_welch()
    df3 = pd.DataFrame({
        'Development': oos_smooth.psd_development,
        'Survey Type': oos_smooth.psd_survey_type,
        'Pipeline Group': oos_smooth.psd_pipeline_group,
        'Section Type': oos_smooth.psd_group_section_type,
        'Frequencies': oos_smooth.psd_freqs,
        'PSD': oos_smooth.psd_vals,
    })

    return df1, df2, df3

def example2_plot1(df):
    """
    Plot Easting Mod vs Northing Mod for all pipelines and section numbers.
    Creates two subplots: one for Section Type = 'Straight', one for Section Type = 'Curve'.
    Each (Pipeline, Section No) is shown as a separate line in each subplot.
    For the Curve subplot, annotates each line with the first value of
    'Design Route Curve Radius' and 'Actual Route Curve Radius'.
    Also adds scatter markers for survey features (PWMD, LBMD, ILT) with feature names in the legend.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_types = ['Straight', 'Curve']
    feature_types = ['PWMD', 'LBMD', 'ILT']
    feature_markers = itertools.cycle(['o', 's', '^'])
    feature_colors = itertools.cycle(['black', 'black', 'black'])
    for idx, (ax, sec_type) in enumerate(zip(axes, section_types)):
        grouped = df[df['Section Type'] == sec_type].groupby(['Pipeline', 'Section No'])
        for (pipeline, section_no), group in grouped:
            ax.plot(
                group['Easting Mod'],
                group['Northing Mod'],
                label=f"Pipeline {pipeline}, Section {section_no}"
            )
            if sec_type == 'Curve' and not group.empty:
                # Annotate at the first point
                x0 = group['Easting Mod'].mean()
                y0 = group['Northing Mod'].mean()
                design_radius = group['Design Route Curve Radius'].iloc[0]
                actual_radius = group['Actual Route Curve Radius'].iloc[0]
                ax.annotate(
                    f"Design: {design_radius:.2f}\nActual: {actual_radius:.2f}",
                    xy=(x0, y0),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.3)
                )
        for feature, marker, color in zip(feature_types, feature_markers, feature_colors):
            feature_mask = (df['Features'] == feature) & (df['Section Type'] == sec_type)
            if feature_mask.any():
                ax.scatter(
                    df.loc[feature_mask, 'Easting Mod'],
                    df.loc[feature_mask, 'Northing Mod'],
                    marker=marker,
                    color=color,
                    s=40,
                    label=feature
                )
    for ax, sec_type in zip(axes, section_types):
        ax.set_xlabel('Easting Mod (m)')
        ax.set_ylabel('Northing Mod (m)')
        ax.set_title(f'Section Type: {sec_type}')
        ax.legend(fontsize='small')
        ax.grid()

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except Exception:
        try:
            mng.window.showMaximized()  # For Qt backend
        except Exception:
            pass  # If backend does not support maximising
    plt.tight_layout()
    plt.show()

def example2_plot2(df):
    """
    Plot Easting Mod vs Northing Mod for all pipeline groups and section types.
    Creates two subplots: one for Section Type = 'Straight', one for Section Type = 'Curve'.
    Each Pipeline Group is shown as a separate line in each subplot.
    For the Curve subplot, annotates each line with the first value of
    'Design Route Curve Radius' and 'Actual Route Curve Radius'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_types = ['Straight', 'Curve']
    index = 0
    for (ax, sec_type) in zip(axes, section_types):
        # Filter for section type
        df_sec = df[df['Section Type'] == sec_type]
        # Group by Pipeline Group and Group Section No
        grouped = df_sec.groupby(['Pipeline Group', 'Group Section No'])
        for (pipeline_group, group_section_no), group in grouped:
            ax.plot(
                group['Group Easting Mod'],
                group['Group FFT Smooth Northing Mod'],
                label = "FFT Smoothing" if index == 0 else None,
                color = 'black'
            )
            ax.plot(
                group['Group Easting Mod'],
                group['Group Gaussian Smooth Northing Mod'],
                label = "Gaussian Smoothing" if index == 0 else None,
                color = 'red'
            )
            ax.plot(
                group['Group Easting Mod'],
                group['Group Northing Mod'],
                label=f"Group {pipeline_group}, Section {group_section_no}",
                linewidth = 0.5,
                color = f'C{index}'
            )
            # Optionally annotate for curves
            if sec_type == 'Curve' and not group.empty:
                x0 = group['Group Easting Mod'].mean()
                y0 = group['Group Northing Mod'].mean()
                design_radius = group['Design Route Curve Radius'].iloc[0]
                actual_radius = group['Actual Route Curve Radius'].iloc[0]
                ax.annotate(
                    f"Design: {design_radius:.2f}\nActual: {actual_radius:.2f}",
                    xy=(x0, y0),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.3)
                )
            index += 1
    for ax, sec_type in zip(axes, section_types):
        ax.set_xlabel('Easting Mod (m)')
        ax.set_ylabel('Northing Mod (m)')
        ax.set_title(f'Section Type: {sec_type}')
        ax.legend(fontsize='small')
        ax.grid()

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except Exception:
        try:
            mng.window.showMaximized()  # For Qt backend
        except Exception:
            pass  # If backend does not support maximising
    plt.tight_layout()
    plt.show()

def example2_plot3(df1, df2):

    cutoff_wavelength = df1['FFT Cutoff Wavelength'].unique()[0]

    # Find all unique group combinations present in df1
    group_cols = ['Development', 'Survey Type', 'Pipeline Group', 'Section Type']
    unique_groups = df1[group_cols].drop_duplicates().itertuples(index=False, name=None)

    fig, a1 = plt.subplots(1, 1, figsize=(14, 6))

    for idx, group in enumerate(unique_groups):

        mask1 = (
            (df1['Development'] == group[0]) &
            (df1['Survey Type'] == group[1]) &
            (df1['Pipeline Group'] == group[2]) &
            (df1['Section Type'] == group[3])
        )
        df1_group = df1[mask1]

        mask2 = (
            (df2['Development'] == group[0]) &
            (df2['Survey Type'] == group[1]) &
            (df2['Pipeline Group'] == group[2]) &
            (df2['Section Type'] == group[3])
        )
        df2_group = df2[mask2]
        print(df2_group)
        a1.plot(
            1/df1_group['Group FFT Smooth Frequencies - Raw'],
            np.abs(df1_group['Group FFT Smooth Spectrum - Raw']),
            label=f"{group[2]} - {group[3]} - Original",
            linewidth = 0.5,
            color=f'C{idx}'
        )
        a1.plot(
            1/df1_group['Group FFT Smooth Frequencies - Filtered'],
            np.abs(df1_group['Group FFT Smooth Spectrum - Filtered']),
            label=f"{group[2]} - {group[3]} - Filtered - WL > {int(cutoff_wavelength)}m",
            linestyle='--',
            color=f'C{idx}'
        )
        a1.axvline(
            cutoff_wavelength,
            linestyle='--',
            color=f'C{idx}'
        )
        a1.plot(
            1/df2_group.iloc[0]['Frequencies'],
            df2_group.iloc[0]['PSD'],
            label=f"{group[2]} - {group[3]} - Welch PSD",
            linestyle=':',
            color=f'C{idx}'
        )

    a1.set_xlabel("Wavelength (m)")
    a1.set_ylabel("Power Spectral Density")
    a1.grid()
    a1.legend()
    a1.set_xscale('log')
    a1.set_yscale('log')

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except Exception:
        try:
            mng.window.showMaximized()  # For Qt backend
        except Exception:
            pass  # If backend does not support maximising
    plt.tight_layout()
    plt.show()


###
### Example 1
###
dfe1 = example1()
print(dfe1.head())

###
### Example 2
###
dfe2_1, dfe2_2, dfe2_3 = example2_data()
# example2_plot1(dfe2_1)
# example2_plot2(dfe2_2)
example2_plot3(dfe2_2, dfe2_3)
