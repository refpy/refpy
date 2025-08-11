'''
Example script
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refpy

def data():
    """
    Example workflow for OOS pipeline survey data processing using refpy.

    Demonstrates:
        - Importing route and survey data from Excel files.
        - Initialising and running the OOSCurvature class.
        - Initialising and running the FFTSmoother class.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame containing the survey data with additional results (arc length, angle, curvature, smoothed coordinates, etc.).
    df2 : pandas.DataFrame
        DataFrame containing the spectral results (PSD and FFT for coordinates and curvature).
    """
    # Import design data
    df1 = pd.read_excel('example_2b.xlsx', sheet_name=SHEETNAME)
    # Initialise the OOS curvature class
    curvature = refpy.OOSCurvature(
        pipeline_group = df1['Pipeline'],
        x = df1['Easting'],
        y = df1['Northing']
    )
    # Run the curvature process
    df1['Arc Length'] = curvature.get_arc_length()
    df1['Angle'] = curvature.get_angle()
    df1['Curvature'] = curvature.get_curvature()
    # Initialise and run FFT smoothing class - Coordinates
    fft_smooth = refpy.FFTSmoother(
        x = df1['Easting'],
        y = df1['Northing'],
        cutoff = FFT_CUTOFF_WAVELENGTH
    )
    df1['Smooth Coordinate - Northing'] = fft_smooth.get_y_smooth()
    df1['Smooth Coordinate - Frequencies - Raw'] = fft_smooth.get_freqs_raw()
    df1['Smooth Coordinate - Spectrum - Raw'] = fft_smooth.get_fft_raw()
    df1['Smooth Coordinate - Frequencies - Filtered'] = fft_smooth.get_freqs()
    df1['Smooth Coordinate - Spectrum - Filtered'] = fft_smooth.get_fft()
    df1['Smooth Coordinate - Cutoff'] = FFT_CUTOFF_WAVELENGTH
    df2 = pd.DataFrame({
        'Development': fft_smooth.get_psd_development(),
        'Survey Type': fft_smooth.get_psd_survey_type(),
        'Pipeline Group': fft_smooth.get_psd_pipeline_group(),
        'Pipeline': fft_smooth.get_psd_group_section_type(),
        'Coordinate - Frequencies': fft_smooth.get_psd_freqs(),
        'Coordinate - PSD': fft_smooth.get_psd_vals(),
    })
    # Initialise FFT smoothing class - Curvatures
    fft_smooth = refpy.FFTSmoother(
        x = df1['Arc Length'],
        y = df1['Curvature'],
        cutoff = FFT_CUTOFF_WAVELENGTH_CURVATURE
    )
    df1['Smooth Curvature - Curvature'] = fft_smooth.get_y_smooth()
    df1['Smooth Curvature - Frequencies - Raw'] = fft_smooth.get_freqs_raw()
    df1['Smooth Curvature - Spectrum - Raw'] = fft_smooth.get_fft_raw()
    df1['Smooth Curvature - Frequencies - Filtered'] = fft_smooth.get_freqs()
    df1['Smooth Curvature - Spectrum - Filtered'] = fft_smooth.get_fft()
    df1['Smooth Curvature - Cutoff'] = FFT_CUTOFF_WAVELENGTH_CURVATURE
    df1['Smooth Curvature - Easting'] = fft_smooth.get_x_recon()
    df1['Smooth Curvature - Northing'] = fft_smooth.get_y_recon()
    df2['Curvature - Frequencies'] = fft_smooth.get_psd_freqs()
    df2['Curvature - PSD'] = fft_smooth.get_psd_vals()
    return df1, df2

def plot1(df):
    """
    Plot 1
    """
    _, ax = plt.subplots(1, 1, figsize=(14, 6))
    for _, pipeline_nb in enumerate(df['Pipeline'].unique()):
        df_temp = df.loc[df['Pipeline'] == pipeline_nb]
        ax.plot(
            df_temp['Easting'],
            df_temp['Northing'],
            label=f'Pipeline {pipeline_nb}'
        )
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend(fontsize='small')
    ax.grid()
    plt.savefig('example_2b_plot1.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot2(df):
    """
    Plot 2
    """
    _, ax = plt.subplots(1, 1, figsize=(14, 6))
    linestyles = ['-', '--']
    for i, pipeline_nb in enumerate(df['Pipeline'].unique()):
        df_temp = df.loc[df['Pipeline'] == pipeline_nb]
        ax.plot(
            df_temp['Easting'],
            df_temp['Northing'],
            label=f'{pipeline_nb} Raw',
            linewidth=0.75,
            linestyle=linestyles[i % len(linestyles)]
        )
        ax.plot(
            df_temp['Easting'],
            df_temp['Smooth Coordinate - Northing'],
            label=f'{pipeline_nb} FFT from Coordinates',
            linestyle=linestyles[i % len(linestyles)]
        )
        if SHEETNAME == 'data1':
            x_raw = df_temp['Easting'].values
            y_raw = df_temp['Northing'].values
            x0, y0 = x_raw[0], y_raw[0]
            dx = x_raw[100] - x_raw[0]
            dy = y_raw[100] - y_raw[0]
            angle_raw = np.arctan2(dy, dx)
            x_smooth = df_temp['Smooth Curvature - Easting'].values
            y_smooth = df_temp['Smooth Curvature - Northing'].values
            dx = x_smooth[100] - x_smooth[0]
            dy = y_smooth[100] - y_smooth[0]
            angle_smooth = np.arctan2(dy, dx)
            angle = angle_raw - angle_smooth
            x_trans = x_smooth * np.cos(angle) - y_smooth * np.sin(angle) + x0
            y_trans = x_smooth * np.sin(angle) + y_smooth * np.cos(angle) + y0
            ax.plot(
                x_trans,
                y_trans,
                label=f'{pipeline_nb} FFT from Curvature',
                linestyle=linestyles[i % len(linestyles)]
            )
        else:
            ax.plot(
                df_temp['Smooth Curvature - Easting'],
                df_temp['Smooth Curvature - Northing'],
                label=f'{pipeline_nb} FFT from Curvature',
                linestyle=linestyles[i % len(linestyles)]
            )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize='small')
    ax.grid()

    plt.savefig('example_2b_plot2.png', dpi=300, bbox_inches='tight')

    # Initilise busy flag and store initial lines and limits for each axis
    on_draw_busy = False
    orig_lines_list = []
    for line in ax.get_lines():
        orig_lines_list.append({
            'x': line.get_xdata(),
            'y': line.get_ydata(),
            'label': line.get_label(),
            'color': line.get_color(),
            'linestyle': line.get_linestyle(),
            'linewidth': line.get_linewidth(),
            'marker': line.get_marker(),
        })
    initial_limits = (ax.get_xlim(), ax.get_ylim())
    prev_limits = (ax.get_xlim(), ax.get_ylim())

    def on_draw(event):
        nonlocal on_draw_busy, orig_lines_list, initial_limits, prev_limits
        # Check if the draw event is already busy
        if on_draw_busy:
            return
        on_draw_busy = True
        # Check if the limits have changed since the last draw
        current_limits = (ax.get_xlim(), ax.get_ylim())
        if all(np.allclose(cur, prev) for cur, prev in zip(current_limits, prev_limits)):
            on_draw_busy = False
            return
        # Store original lines and limits for each axis

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xlim0, ylim0 = initial_limits
        is_home = np.allclose(xlim, xlim0) and np.allclose(ylim, ylim0)
        # Home: Restore original lines
        if is_home:
            print('Home Button Pressed - Restoring Original Lines')
            for line in list(ax.get_lines()):
                line.remove()
            for line_data in orig_lines_list:
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
            print('Zoom/Pan Detected - Transforming')
            for line in ax.get_lines():
                if line.get_label() == "P1 Raw":
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    mask = (
                        (xdata >= xlim[0]) & (xdata <= xlim[1]) &
                        (ydata >= ylim[0]) & (ydata <= ylim[1])
                    )
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
                if x_zoom.size == 0 or y_zoom.size == 0:
                    print(f"No data points within the current zoom/pan limits for line: {label}")
                    continue
                #
                if not 'FFT from Coordinates' in line.get_label():
                    x0_, y0_ = x_zoom[0], y_zoom[0]
                    dx_ = x_zoom[-1] - x_zoom[0]
                    dy_ = y_zoom[-1] - y_zoom[0]
                    angle_ = -np.arctan2(dy_, dx_)
                    x_trans = (x_zoom - x0_) * np.cos(angle_) - (y_zoom - y0_) * np.sin(angle_)
                    y_trans = (x_zoom - x0_) * np.sin(angle_) + (y_zoom - y0_) * np.cos(angle_)
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
        ax.legend()
        prev_limits = (xlim, ylim)
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

def plot3(df1, df2):
    """
    Plot 3
    """

    _, a1 = plt.subplots(1, 1, figsize=(14, 6))

    for idx, pipeline_nb in enumerate(df1['Pipeline'].unique()):
        df1_temp = df1.loc[df1['Pipeline'] == pipeline_nb]
        a1.plot(
            1/df1_temp['Smooth Coordinate - Frequencies - Raw'],
            np.abs(df1_temp['Smooth Coordinate - Spectrum - Raw']),
            linewidth = 0.5,
            label=f"{pipeline_nb} - Spectrum Raw",
            color=f'C{idx}'
        )
        a1.plot(
            1/df1_temp['Smooth Coordinate - Frequencies - Filtered'],
            np.abs(df1_temp['Smooth Coordinate - Spectrum - Filtered']),
            linestyle='--',
            label=f"{pipeline_nb} - Spectrum Filtered",
            color=f'C{idx}'
        )
        freqs = np.array(df2.iloc[0]['Coordinate - Frequencies'])
        psd = np.array(df2.iloc[0]['Coordinate - PSD'])
        mask = freqs != 0
        a1.plot(
            1/freqs[mask],
            psd[mask],
            label=f"{pipeline_nb} - Welch PSD",
            linestyle=':',
            color=f'C{idx}'
        )
    a1.set_xlabel("Wavelength of the Pipeline Shape Harmonics (m)")
    a1.set_ylabel("Power Spectral Density")
    a1.grid()
    a1.legend()
    a1.set_xscale('log')
    a1.set_yscale('log')

    plt.savefig('example_2b_plot3.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot4(df):
    """
    Plot 4
    """

    _, ax = plt.subplots(1, 1, figsize=(14, 6))
    for _, pipeline_nb in enumerate(df['Pipeline'].unique()):
        df_temp = df.loc[df['Pipeline'] == pipeline_nb]
        ax.plot(
            df_temp['Arc Length'],
            df_temp['Curvature'],
            label = f"{pipeline_nb} - Raw",
            color = f'C{_}'
        )
        ax.plot(
            df_temp['Arc Length'],
            df_temp['Smooth Curvature - Curvature'],
            label = f"{pipeline_nb} - FFT Smoothed",
            color = f'C{_}',
            linestyle = '--'
        )
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Curvature (1/m)')
    ax.legend(fontsize='small')
    ax.grid()

    plt.savefig('example_2b_plot4.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot5(df1, df2):
    """
    Plot 5
    """

    _, a1 = plt.subplots(1, 1, figsize=(14, 6))

    for idx, pipeline_nb in enumerate(df1['Pipeline'].unique()):
        df1_temp = df1.loc[df1['Pipeline'] == pipeline_nb]
        a1.plot(
            1/df1_temp['Smooth Curvature - Frequencies - Raw'],
            np.abs(df1_temp['Smooth Curvature - Spectrum - Raw']),
            linewidth = 0.5,
            label=f"{pipeline_nb} - Spectrum Raw",
            color=f'C{idx}'
        )
        a1.plot(
            1/df1_temp['Smooth Curvature - Frequencies - Filtered'],
            np.abs(df1_temp['Smooth Curvature - Spectrum - Filtered']),
            linestyle='--',
            label=f"{pipeline_nb} - Spectrum Filtered",
            color=f'C{idx}'
        )
        freqs = np.array(df2.iloc[0]['Curvature - Frequencies'])
        psd = np.array(df2.iloc[0]['Curvature - PSD'])
        mask = freqs != 0
        a1.plot(
            1/freqs[mask],
            psd[mask],
            label=f"{pipeline_nb} - Welch PSD",
            linestyle=':',
            color=f'C{idx}'
        )
    a1.set_xlabel("Wavelength of the Pipeline Shape Harmonics (m)")
    a1.set_ylabel("Power Spectral Density")
    a1.grid()
    a1.legend()
    a1.set_xscale('log')
    a1.set_yscale('log')

    plt.savefig('example_2b_plot5.png', dpi=300, bbox_inches='tight')
    plt.show()

###
### Example 2
###
SHEETNAME = 'data2' #Options: 'data1' or 'data2'
FFT_CUTOFF_WAVELENGTH = 16  # Cutoff of smaller wavelengths in meters
FFT_CUTOFF_WAVELENGTH_CURVATURE = 16  # Cutoff of smaller wavelengths in meters
dfe1, dfe2 = data()
plot1(dfe1)
plot2(dfe1)
plot3(dfe1, dfe2)
plot4(dfe1)
plot5(dfe1, dfe2)
