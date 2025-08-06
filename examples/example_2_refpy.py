'''
Example script
'''
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refpy

def example2_data():
    """
    Example workflow for OOS pipeline survey and route data processing using refpy.

    This function demonstrates:
    - Importing route and survey data from Excel files.
    - Initializing and running the OOSAnonymisation class with route and survey data.
    - Initializing and running the OOSDespiker class with post-processed survey data.
    - Initializing and running the FFTSmoother class with post-processed survey data.
    - Initializing and running the GaussianSmoother class with post-processed survey data.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame containing the survey data with modified coordinates and curve radius results.
    df2 : pandas.DataFrame
        DataFrame containing group section and modified coordinates for each survey point.
    """
    # Import design data
    dfr = pd.read_excel(FILENAME, sheet_name='Example2-Route')
    dfs = pd.read_excel(FILENAME, sheet_name='Example2-Survey')
    dfs = dfs.loc[
        dfs['Development'].isin(DEVELOPMENT) &
        dfs['Survey Type'].isin(SURVEY_TYPE) &
        dfs['Pipeline Group'].isin(PIPELINE_GROUP) &
        dfs['Pipeline'].isin(PIPELINE)
    ]
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
        'Development': oos.get_survey_development(),
        'Survey Type': oos.get_survey_type(),
        'Pipeline Group': oos.get_survey_pipeline_group(),
        'Pipeline': oos.get_survey_pipeline(),
        'Section Type': oos.get_survey_section_type(),
        'Section No': oos.get_survey_section_no(),
        'KP Mod': oos.get_survey_section_kp_mod(),
        'Easting Mod': oos.get_survey_section_easting_mod(),
        'Northing Mod': oos.get_survey_section_northing_mod(),
        'Features': oos.get_survey_feature(),
        'Design Route Curve Radius': oos.get_survey_design_route_curve_radius(),
        'Actual Route Curve Radius': oos.get_survey_actual_route_curve_radius()
    })
    # Initialise and run the despiking class
    despike = refpy.OOSDespiker(
        development = df1['Development'],
        survey_type = df1['Survey Type'],
        pipeline_group = df1['Pipeline Group'],
        group_section_type = df1['Section Type'],
        y = df1['Northing Mod'],
        window = 100,
        sigma = 1.5
    )
    df1['Northing Mod - Despike'] = despike.get_y_despike()
    # Anonymised survey
    oos.process(anonymise=True)
    df2 = pd.DataFrame({
        'Development': oos.get_survey_development(),
        'Survey Type': oos.get_survey_type(),
        'Pipeline Group': oos.get_survey_pipeline_group(),
        'Pipeline': oos.get_survey_pipeline(),
        'Section Type': oos.get_survey_section_type(),
        'Section No': oos.get_survey_section_no(),
        'KP Mod': oos.get_survey_section_kp_mod(),
        'Easting Mod': oos.get_survey_section_easting_mod(),
        'Northing Mod': oos.get_survey_section_northing_mod(),
        'Features': oos.get_survey_feature(),
        'Design Route Curve Radius': oos.get_survey_design_route_curve_radius(),
        'Actual Route Curve Radius': oos.get_survey_actual_route_curve_radius(),
        'Group Section Type': oos.get_survey_group_section_type(),
        'Group Section No': oos.get_survey_group_section_no(),
        'Group KP Mod': oos.get_survey_group_section_kp_mod(),
        'Group Easting Mod': oos.get_survey_group_section_easting_mod(),
        'Group Northing Mod': oos.get_survey_group_section_northing_mod()
    })
    # Initialise and run the despiking class
    despike = refpy.OOSDespiker(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        y = df2['Group Northing Mod'],
        window = 11,
        sigma = 3.0
    )
    # Run the despiking process
    df2['Group Northing Mod - Despike'] = despike.get_y_despike()
    # Initialise the OOS curvature class
    curvature = refpy.OOSCurvature(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        x = df2['Group Easting Mod'],
        y = df2['Group Northing Mod - Despike']
    )
    # Run the curvature process
    df2['Group Arc Length'] = curvature.get_arc_length()
    df2['Group Angle'] = curvature.get_angle()
    df2['Group Curvature'] = curvature.get_curvature()
    # Initialise and run FFT smoothing class - Coordinates
    fft_smooth = refpy.FFTSmoother(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        x = df2['Group Easting Mod'],
        y = df2['Group Northing Mod - Despike'],
        cutoff = FFT_CUTOFF_WAVELENGTH
    )
    df2['Group FFT Smooth Coordinate - Northing Mod'] = fft_smooth.get_y_smooth()
    df2['Group FFT Smooth Coordinate - Frequencies - Raw'] = fft_smooth.get_freqs_raw()
    df2['Group FFT Smooth Coordinate - Spectrum - Raw'] = fft_smooth.get_fft_raw()
    df2['Group FFT Smooth Coordinate - Frequencies - Filtered'] = fft_smooth.get_freqs()
    df2['Group FFT Smooth Coordinate - Spectrum - Filtered'] = fft_smooth.get_fft()
    df2['Group FFT Smooth Coordinate - Cutoff'] = FFT_CUTOFF_WAVELENGTH
    df3 = pd.DataFrame({
        'Development': fft_smooth.get_psd_development(),
        'Survey Type': fft_smooth.get_psd_survey_type(),
        'Pipeline Group': fft_smooth.get_psd_pipeline_group(),
        'Section Type': fft_smooth.get_psd_group_section_type(),
        'Coordinate - Frequencies': fft_smooth.get_psd_freqs(),
        'Coordinate - PSD': fft_smooth.get_psd_vals(),
    })
    # Initialise FFT smoothing class - Curvatures
    fft_smooth = refpy.FFTSmoother(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        x = df2['Group Arc Length'],
        y = df2['Group Curvature'],
        cutoff = FFT_CUTOFF_WAVELENGTH_CURVATURE
    )
    df2['Group FFT Smooth Curvature - Curvature'] = fft_smooth.get_y_smooth()
    df2['Group FFT Smooth Curvature - Frequencies - Raw'] = fft_smooth.get_freqs_raw()
    df2['Group FFT Smooth Curvature - Spectrum - Raw'] = fft_smooth.get_fft_raw()
    df2['Group FFT Smooth Curvature - Frequencies - Filtered'] = fft_smooth.get_freqs()
    df2['Group FFT Smooth Curvature - Spectrum - Filtered'] = fft_smooth.get_fft()
    df2['Group FFT Smooth Curvature - Cutoff'] = FFT_CUTOFF_WAVELENGTH_CURVATURE
    df3['Curvature - Frequencies'] = fft_smooth.get_psd_freqs()
    df3['Curvature - PSD'] = fft_smooth.get_psd_vals()
    df2['Group FFT Smooth Curvature - Easting Mod'] = fft_smooth.get_x_recon()
    df2['Group FFT Smooth Curvature - Northing Mod'] = fft_smooth.get_y_recon()
    # Initialise Gaussian smoothing class
    gaussian_smooth = refpy.GaussianSmoother(
        development = df2['Development'],
        survey_type = df2['Survey Type'],
        pipeline_group = df2['Pipeline Group'],
        group_section_type = df2['Group Section Type'],
        x = df2['Group Easting Mod'],
        y = df2['Group Northing Mod - Despike'],
        bandwidth = GAUSSIAN_BANDWIDTH
    )
    df2['Group Gaussian Smooth Coordinate - Northing Mod'] = gaussian_smooth.get_y_smooth()
    df2['Group Gaussian Smooth Coordinate - Bandwidth'] = GAUSSIAN_BANDWIDTH
    return df1, df2, df3

def example2_plot1(df):
    """
    Plot Easting Mod vs Northing Mod for all pipelines and section numbers.
    Creates two subplots: one for Section Type = 'Straight', one for Section Type = 'Curve'.
    Each (Pipeline, Section No) is shown as a separate line in each subplot.
    For the Curve subplot, annotates each line with the first value of
    'Design Route Curve Radius' and 'Actual Route Curve Radius'.
    Also adds scatter markers for survey features (PWMD, LBMD, ILT) with feature names in the
    legend.
    """

    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_types = ['Straight', 'Curve']
    feature_types = ['PWMD', 'LBMD', 'ILT']
    feature_markers = itertools.cycle(['o', 's', '^'])
    feature_colors = itertools.cycle(['black', 'black', 'black'])
    for _, (ax, sec_type) in enumerate(zip(axes, section_types)):
        grouped = (
            df[df['Section Type'] == sec_type]
            .groupby(['Development', 'Survey Type', 'Pipeline Group', 'Pipeline', 'Section No'])
        )
        i = 0
        for (development, survey_type, pipeline_group, pipeline, section_no), group in grouped:
            # Add scatter plot for NaN despiked points
            nan_mask = group['Northing Mod - Despike'].isna()
            if nan_mask.any():
                ax.scatter(
                    group.loc[nan_mask, 'Easting Mod'],
                    group.loc[nan_mask, 'Northing Mod'],
                    color='red',
                    marker='x',
                    s=20,
                    label='Despiked Survey Points' if i == 0 else None
                )
            ax.plot(
                group['Easting Mod'],
                group['Northing Mod'],
                label=(
                    f"{development} - {survey_type} - {pipeline_group}"
                    f" - {pipeline} - S{int(section_no)}"
                )
            )
            if survey_type == SURVEY_TYPE[0] and sec_type == 'Curve' and not group.empty:
                # Annotate at the first point
                x0 = group['Easting Mod'].mean()
                y0 = group['Northing Mod'].mean()
                design_radius = group['Design Route Curve Radius'].iloc[0]
                actual_radius = group['Actual Route Curve Radius'].iloc[0]
                ax.annotate(
                    f"Design: {design_radius:.0f}\nActual: {actual_radius:.0f}",
                    xy=(x0, y0),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.3)
                )
            i += 1
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
    except AttributeError:
        try:
            mng.window.showMaximized()  # For Qt backend
        except AttributeError:
            pass  # If backend does not support maximising
    plt.savefig('example_2_plot1.png', dpi=300, bbox_inches='tight')
    plt.show()

def example2_plot2(df):
    """
    Plot Easting Mod vs Northing Mod for all pipeline groups and section types.
    Creates two subplots: one for Section Type = 'Straight', one for Section Type = 'Curve'.
    Each Pipeline Group is shown as a separate line in each subplot.
    For the Curve subplot, annotates each line with the first value of
    'Design Route Curve Radius' and 'Actual Route Curve Radius'.
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_types = ['Straight', 'Curve']
    for (ax, sec_type) in zip(axes, section_types):
        df_sec = df[df['Section Type'] == sec_type]
        grouped = (
            df_sec[df_sec['Section Type'] == sec_type]
            .groupby(['Development', 'Survey Type', 'Pipeline Group'])
        )
        for (development, survey_type, pipeline_group), group in grouped:
            ax.plot(
                group['Group Easting Mod'],
                group['Group Northing Mod'],
                label=f"{development} - {survey_type} - {pipeline_group}",
                linewidth=0.75,
            )
            ax.plot(
                group['Group Easting Mod'],
                group['Group FFT Smooth Coordinate - Northing Mod'],
                label="FFT Smoothing from Coordinates",
            )
            ax.plot(
                group['Group FFT Smooth Curvature - Easting Mod'],
                group['Group FFT Smooth Curvature - Northing Mod'],
                label="FFT Smoothing from Curvature",
            )
            ax.plot(
                group['Group Easting Mod'],
                group['Group Gaussian Smooth Coordinate - Northing Mod'],
                label="Gaussian Smoothing from Coordinates",
            )
        grouped = (
            df_sec[df_sec['Section Type'] == sec_type]
            .groupby(['Development', 'Survey Type', 'Pipeline Group', 'Group Section No'])
        )
        if sec_type == 'Curve':
            for (_, _, _, _), group in grouped:
                # Annotate at the first point
                x0 = group['Group Easting Mod'].mean()
                y0 = group['Group Northing Mod'].mean()
                design_radius = group['Design Route Curve Radius'].iloc[0]
                actual_radius = group['Actual Route Curve Radius'].iloc[0]
                ax.annotate(
                    f"Design: {design_radius:.0f}\nActual: {actual_radius:.0f}",
                    xy=(x0, y0),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.3)
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
    except AttributeError:
        try:
            mng.window.showMaximized()  # For Qt backend
        except AttributeError:
            pass  # If backend does not support maximising
    plt.savefig('example_2_plot2.png', dpi=300, bbox_inches='tight')

    # Initilise busy flag and store initial lines and limits for each axis
    on_draw_busy = False
    orig_lines_list = [None, None]
    for i, ax in enumerate(axes):
        orig_lines_list[i] = []
        for line in ax.get_lines():
            orig_lines_list[i].append({
                'x': line.get_xdata(),
                'y': line.get_ydata(),
                'label': line.get_label(),
                'color': line.get_color(),
                'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth(),
                'marker': line.get_marker(),
            })
    initial_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    prev_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]

    def on_draw(event):
        nonlocal on_draw_busy, orig_lines_list, initial_limits, prev_limits
        # Check if the draw event is already busy
        if on_draw_busy:
            return
        on_draw_busy = True
        # Check if the limits have changed since the last draw
        current_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
        if all(np.allclose(cur, prev) for cur, prev in zip(current_limits, prev_limits)):
            on_draw_busy = False
            return
        # Store original lines and limits for each axis
        for i, ax in enumerate(axes):
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            prev_xlim, prev_ylim = prev_limits[i]
            if np.allclose(xlim, prev_xlim) and np.allclose(ylim, prev_ylim):
                continue
            xlim0, ylim0 = initial_limits[i]
            is_home = np.allclose(xlim, xlim0) and np.allclose(ylim, ylim0)
            # Home: Restore original lines
            if is_home:
                print(i, on_draw_busy, 'Home Button Pressed - Restoring Original Lines')
                for line in list(ax.get_lines()):
                    line.remove()
                for line_data in orig_lines_list[i]:
                    ax.plot(
                        line_data['x'],
                        line_data['y'],
                        label=line_data['label'],
                        color=line_data['color'],
                        linestyle=line_data['linestyle'],
                        linewidth=line_data['linewidth'],
                        marker=line_data['marker'],
                    )
            # Zoom: Translate and rotate lines
            else:
                print(i, on_draw_busy, 'Zoom/Pan Detected - Transforming')
                for line in ax.get_lines():
                    if line.get_label() == "FFT Smoothing from Coordinates":
                        xdata = line.get_xdata()
                        ydata = line.get_ydata()
                        mask = (
                            (xdata >= xlim[0]) & (xdata <= xlim[1]) &
                            (ydata >= ylim[0]) & (ydata <= ylim[1])
                        )
                        x_zoom = xdata[mask]
                        y_zoom = ydata[mask]
                        if len(x_zoom) == 0 or len(y_zoom) == 0:
                            angle = 0.0
                            x0, y0 = 0.0, 0.0
                            break
                        x0, y0 = x_zoom[0], y_zoom[0]
                        dx = x_zoom[-1] - x_zoom[0]
                        dy = y_zoom[-1] - y_zoom[0]
                        angle = -np.arctan2(dy, dx)
                        break
                all_x_trans = []
                all_y_trans = []
                for line in list(ax.get_lines()):
                    #
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    label = line.get_label()
                    color = line.get_color()
                    linestyle = line.get_linestyle()
                    linewidth = line.get_linewidth()
                    marker = line.get_marker()
                    #
                    line.remove()
                    #
                    x_zoom = xdata[mask]
                    y_zoom = ydata[mask]
                    #
                    if line.get_label() == "FFT Smoothing from Curvature":
                        if len(x_zoom) == 0 or len(y_zoom) == 0:
                            x0_, y0_ = 0.0, 0.0
                            dx_ = dy_ = angle_ = 0.0
                        else:
                            x0_, y0_ = x_zoom[0], y_zoom[0]
                            dx_ = x_zoom[-1] - x_zoom[0]
                            dy_ = y_zoom[-1] - y_zoom[0]
                            angle_ = -np.arctan2(dy_, dx_)
                        x_trans = (x_zoom - x0_) * np.cos(angle_) - (y_zoom - y0_) * np.sin(angle_)
                        y_trans = (x_zoom - x0_) * np.sin(angle_) + (y_zoom - y0_) * np.cos(angle_)
                    else:
                        x_trans = (x_zoom - x0) * np.cos(angle) - (y_zoom - y0) * np.sin(angle)
                        y_trans = (x_zoom - x0) * np.sin(angle) + (y_zoom - y0) * np.cos(angle)
                    #
                    all_x_trans.append(x_trans)
                    all_y_trans.append(y_trans)
                    #
                    ax.plot(
                        x_trans, y_trans,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        marker=marker
                    )
                # Only set axis limits if they are not already correct
                if (
                    all_x_trans and all_y_trans and
                    np.size(np.concatenate(all_x_trans)) > 0 and
                    np.size(np.concatenate(all_y_trans)) > 0
                ):
                    all_x = np.concatenate(all_x_trans)
                    all_y = np.concatenate(all_y_trans)
                    ax.set_xlim(
                        np.nanmin(all_x) - 0.2*(np.nanmax(all_x) - np.nanmin(all_x)),
                        np.nanmax(all_x) + 0.2*(np.nanmax(all_x) - np.nanmin(all_x))
                    )
                    ax.set_ylim(
                        np.nanmin(all_y) - 0.2*(np.nanmax(all_y) - np.nanmin(all_y)),
                        np.nanmax(all_y) + 0.2*(np.nanmax(all_y) - np.nanmin(all_y))
                    )
            prev_limits[i] = (xlim, ylim)
            plt.gcf().canvas.draw_idle()
        on_draw_busy = False

    # Hook both draw and mouse release events
    canvas = plt.gcf().canvas
    canvas.mpl_connect('draw_event', on_draw)
    def _on_mouse_release(event):
        if event.button == 1:
            on_draw(None)
    canvas.mpl_connect('button_release_event', _on_mouse_release)

    plt.show()

def example2_plot3(df1, df2):
    """
    Plot group-wise FFT and Welch PSD spectra for pipeline survey data.

    For each unique combination of Development, Survey Type, Pipeline Group, and Section Type,
    this function plots:
    - The original (raw) FFT spectrum of the group-wise smoothed Northing Mod coordinate.
    - The filtered FFT spectrum after applying the cutoff wavelength.
    - The Welch Power Spectral Density (PSD) estimate for the group.

    All spectra are plotted as functions of wavelength (1/frequency) on a log-log scale.
    The cutoff wavelength is indicated with a vertical dashed line for each group.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame containing group-wise FFT results, with columns:
        'Development', 'Survey Type', 'Pipeline Group', 'Section Type',
        'Group FFT Smooth Frequencies - Raw', 'Group FFT Smooth Spectrum - Raw',
        'Group FFT Smooth Frequencies - Filtered', 'Group FFT Smooth Spectrum - Filtered',
        and 'FFT Cutoff Wavelength'.
    df2 : pandas.DataFrame
        DataFrame containing Welch PSD results, with columns:
        'Development', 'Survey Type', 'Pipeline Group', 'Section Type',
        'Frequencies', and 'PSD'.

    Returns
    -------
    None
        Displays a matplotlib plot window with the group-wise spectra.
    """

    cutoff_wavelength = df1['Group FFT Smooth Coordinate - Cutoff'].unique()[0]

    # Find all unique group combinations present in df1
    group_cols = ['Development', 'Survey Type', 'Pipeline Group', 'Section Type']
    unique_groups = df1[group_cols].drop_duplicates().itertuples(index=False, name=None)

    _, a1 = plt.subplots(1, 1, figsize=(14, 6))

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

        a1.plot(
            1/df1_group['Group FFT Smooth Coordinate - Frequencies - Raw'],
            np.abs(df1_group['Group FFT Smooth Coordinate - Spectrum - Raw']),
            label=f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - Original",
            linewidth = 0.5,
            color=f'C{idx}'
        )
        a1.plot(
            1/df1_group['Group FFT Smooth Coordinate - Frequencies - Filtered'],
            np.abs(df1_group['Group FFT Smooth Coordinate - Spectrum - Filtered']),
            label=(
                f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - "
                f"WL > {int(cutoff_wavelength)}m"
            ),
            linestyle='--',
            color=f'C{idx}'
        )
        a1.axvline(
            cutoff_wavelength,
            linestyle='--',
            color=f'C{idx}'
        )
        freqs = np.array(df2_group.iloc[0]['Coordinate - Frequencies'])
        psd = np.array(df2_group.iloc[0]['Coordinate - PSD'])
        mask = freqs != 0
        a1.plot(
            1/freqs[mask],
            psd[mask],
            label=f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - Welch PSD",
            linestyle=':',
            color=f'C{idx}'
        )

    a1.set_xlabel("Wavelength of the Pipeline Shape Harmonics (m)")
    a1.set_ylabel("Power Spectral Density")
    a1.grid()
    a1.legend()
    a1.set_xscale('log')
    a1.set_yscale('log')

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except AttributeError:
        try:
            mng.window.showMaximized()  # For Qt backend
        except AttributeError:
            pass  # If backend does not support maximising
    plt.savefig('example_2_plot3.png', dpi=300, bbox_inches='tight')
    plt.show()

def example2_plot4(df):
    """
    Plot arc length vs curvature for all pipeline groups and section types.
    Creates two subplots: one for Section Type = 'Straight', one for Section Type = 'Curve'.
    Each Pipeline Group is shown as a separate line in each subplot.
    For the Curve subplot, annotates each line with the first value of
    'Design Route Curve Radius' and 'Actual Route Curve Radius'.
    """

    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    section_types = ['Straight', 'Curve']
    for (ax, sec_type) in zip(axes, section_types):
        index = 0
        # Filter for section type
        df_sec = df[df['Section Type'] == sec_type]
        # Group by Pipeline Group
        for _, (ax, sec_type) in enumerate(zip(axes, section_types)):
            grouped = (
                df_sec[df_sec['Section Type'] == sec_type]
                .groupby(['Development', 'Survey Type', 'Pipeline Group'])
            )
            for (development, survey_type, pipeline_group), group in grouped:
                if survey_type == 'As-Laid':
                    ax.plot(
                        group['Group Arc Length'],
                        group['Group FFT Smooth Curvature - Curvature'],
                        label = "FFT Smoothing" if index == 0 else None,
                        color = 'black'
                    )
        # Group by Pipeline Group and Group Section No
        for _, (ax, sec_type) in enumerate(zip(axes, section_types)):
            grouped = (
                df_sec[df_sec['Section Type'] == sec_type]
                .groupby(['Development', 'Survey Type', 'Pipeline Group', 'Group Section No'])
            )
            for (development, survey_type, pipeline_group, group_section_no), group in grouped:
                ax.plot(
                    group['Group Arc Length'],
                    group['Group Curvature'],
                    label=f"{development} - {survey_type} - {pipeline_group} - S{group_section_no}",
                    linewidth = 0.25,
                    color = f'C{index}'
                )
                index += 1
    for ax, sec_type in zip(axes, section_types):
        ax.set_xlabel('Arc Length (m)')
        ax.set_ylabel('Curvature (1/m)')
        ax.set_title(f'Section Type: {sec_type}')
        ax.legend(fontsize='small')
        ax.grid()

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except AttributeError:
        try:
            mng.window.showMaximized()  # For Qt backend
        except AttributeError:
            pass  # If backend does not support maximising
    plt.savefig('example_2_plot4.png', dpi=300, bbox_inches='tight')
    plt.show()

def example2_plot5(df1, df2):
    """
    Plot group-wise FFT and Welch PSD spectra for pipeline survey data.

    For each unique combination of Development, Survey Type, Pipeline Group, and Section Type,
    this function plots:
    - The original (raw) FFT spectrum of the group-wise smoothed Northing Mod coordinate.
    - The filtered FFT spectrum after applying the cutoff wavelength.
    - The Welch Power Spectral Density (PSD) estimate for the group.

    All spectra are plotted as functions of wavelength (1/frequency) on a log-log scale.
    The cutoff wavelength is indicated with a vertical dashed line for each group.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame containing group-wise FFT results, with columns:
        'Development', 'Survey Type', 'Pipeline Group', 'Section Type',
        'Group FFT Smooth Frequencies - Raw', 'Group FFT Smooth Spectrum - Raw',
        'Group FFT Smooth Frequencies - Filtered', 'Group FFT Smooth Spectrum - Filtered',
        and 'FFT Cutoff Wavelength'.
    df2 : pandas.DataFrame
        DataFrame containing Welch PSD results, with columns:
        'Development', 'Survey Type', 'Pipeline Group', 'Section Type',
        'Frequencies', and 'PSD'.

    Returns
    -------
    None
        Displays a matplotlib plot window with the group-wise spectra.
    """

    cutoff_wavelength = df1['Group FFT Smooth Curvature - Cutoff'].unique()[0]

    # Find all unique group combinations present in df1
    group_cols = ['Development', 'Survey Type', 'Pipeline Group', 'Section Type']
    unique_groups = df1[group_cols].drop_duplicates().itertuples(index=False, name=None)

    _, a1 = plt.subplots(1, 1, figsize=(14, 6))

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

        a1.plot(
            1/df1_group['Group FFT Smooth Curvature - Frequencies - Raw'],
            np.abs(df1_group['Group FFT Smooth Curvature - Spectrum - Raw']),
            label=f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - Original",
            linewidth = 0.5,
            color=f'C{idx}'
        )
        a1.plot(
            1/df1_group['Group FFT Smooth Curvature - Frequencies - Filtered'],
            np.abs(df1_group['Group FFT Smooth Curvature - Spectrum - Filtered']),
            label=(
                f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - "
                f"WL > {int(cutoff_wavelength)}m"
            ),
            linestyle='--',
            color=f'C{idx}'
        )
        a1.axvline(
            cutoff_wavelength,
            linestyle='--',
            color=f'C{idx}'
        )
        freqs = np.array(df2_group.iloc[0]['Curvature - Frequencies'])
        psd = np.array(df2_group.iloc[0]['Curvature - PSD'])
        mask = freqs != 0
        a1.plot(
            1/freqs[mask],
            psd[mask],
            label=f"{group[0]} - {group[1]} - {group[2]} - {group[3]} - Welch PSD",
            linestyle=':',
            color=f'C{idx}'
        )

    a1.set_xlabel("Wavelength of the Pipeline Curvature Harmonics (m)")
    a1.set_ylabel("Power Spectral Density")
    a1.grid()
    a1.legend()
    a1.set_xscale('log')
    a1.set_yscale('log')

    # Maximise the plot window
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # For TkAgg backend (Windows)
    except AttributeError:
        try:
            mng.window.showMaximized()  # For Qt backend
        except AttributeError:
            pass  # If backend does not support maximising
    plt.savefig('example_2_plot5.png', dpi=300, bbox_inches='tight')
    plt.show()

###
### Example 2
###
FILENAME = 'example_1_2.xlsx'
DEVELOPMENT = ['Example']
SURVEY_TYPE = ['As-Laid']
PIPELINE_GROUP = ['PG2']
PIPELINE = ['F1']
FFT_CUTOFF_WAVELENGTH = 16  # Cutoff of smaller wavelengths in meters
FFT_CUTOFF_WAVELENGTH_CURVATURE = 16  # Cutoff of smaller wavelengths in meters
GAUSSIAN_BANDWIDTH = 4.0  # Bandwidth for Gaussian smoothing
dfe2_1, dfe2_2, dfe2_3 = example2_data()
example2_plot1(dfe2_1)
example2_plot2(dfe2_2)
example2_plot3(dfe2_2, dfe2_3)
example2_plot4(dfe2_2)
example2_plot5(dfe2_2, dfe2_3)
