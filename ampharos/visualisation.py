import numpy as np
import matplotlib.pyplot as plt
import os

from .support_functions import get_frequency_track, interpolate_frequency_track
from .io_functions import read_MESA_history_columns



##### MISCELANEOUS #########################################################################


def make_ticks(ax):
    """
    Turn on the minor and right/top ticks on the given matplotlib.axes object

    Author: Dario Fritzewski

    ------- Parameters -------
    ax: matplotlib.axes.Axes
        axes object of the figure

    """
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()


def plot_frequency_tracks(tracks, ls, ms, n_pgs, frot_rel=0.00,
                          interpolate=True,
                          title="",
                          save=False, save_dir="plots/",
                          save_name="frequency_track",
                          show=True) :
    """
    Make an interactive plot of the frequency tracks of the given degrees and
    radial orders against the central hydrogen fraction.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    ls: list of ints
        Degrees of the modes to be plotted
    ms: list of ints
        Azimuthal orders of the modes to be plotted
    n_pgs: list of ints
        Radial orders of the modes to be plotted
    frot_rel: float
        Rotation rate relative to critical.
        Defaults to 0.00
    interpolate: bool
        Whether to show the cubic interpolation of the frequency tracks
        Defaults to True
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. ".png" is automatically added at the end of the filename.
        Defaults to "frequency_track"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    ax: matplotlib.axes.Axes
        Axes object encapsulating all elements of this plot
    """
    assert len(ls) == len(n_pgs)
    assert len(ls) == len(ms)

    markers = {0:"o", 1:"^", 2:"p"} # per degree
    colours = {-3:'navy', -2:'blue', -1:'dodgerblue', 0:"purple", 1:'violet',
               2:'salmon', 3:'red', 4:'red', 5:'red'} # per radial order

    fig, ax = plt.subplots(1,1,figsize=(10,6))

    for i, (l, m, n_pg) in enumerate(zip(ls, ms, n_pgs)) :
        if n_pg in colours.keys() :
            colour = colours[n_pg]
        elif n_pg > 0 : # Higher order p-mode
            colour = "maroon"
        else : # higher order g-mode
            colour = "midnightblue"

        freqs = get_frequency_track(tracks, l, m, n_pg, frot_rel)
        if np.sum(~np.isnan(freqs)) < 3 :
            continue
        ax.scatter(freqs, tracks["Xcs"], \
                   label=f"l = {l} ; m = {m} ; n_pg = {n_pg}",
                   marker=markers[l], alpha=0.7, color=colour)
        if interpolate :
            Xcs_interp, freqs_interp = interpolate_frequency_track(tracks,
                                                                   l, m, n_pg,
                                                                   frot_rel)
            if len(Xcs_interp) == 0 :
                continue
            ax.plot(freqs_interp, Xcs_interp, ls="dotted", alpha=0.7,
                    color=colour)

    ax.set_xlim(2, 15)
    ax.set_ylim(0, 0.701)
    ax.set_xlabel("frequency [c/d]", fontsize=12)
    ax.set_ylabel("central hydrogen fraction", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=12)
    make_ticks(ax)

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{save_name}", dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, ax


def add_instability_strip(ax, y_axis='logL',
                          datadir='instability_strips/', Zini='0.014', rot='0.0',
                          IStype='both', p_mode_col='k', p_mode_ls='solid',
                          g_mode_col='k', g_mode_ls='dashed',
                          zorder=0, fill=False, alpha=0.2, lw=.5):
    """
    Plot the SPB and/or beta Cep instability strip(s) for a certain metallicity

    Author: Siemen Burssens

    ------- Parameters -------
    ax: matplotlib.Axes
        Axes object the instability strip is to be plotted on
    y_axis: string
        What is shown on the y-axis
        Must be 'logL' or 'logLsp'
        Defaults to 'logL'
    datadir: string
        Path to the .txt files containing the boundaries. Must end with '/'
        Defaults to 'instability_strip/'
    Zini: string
        Metallicity of the models used to compute the instability strip
        Must be one of '0.010', '0.014', '0.020'
        Defaults to '0.014'
    rot: string
        Rotation rate relative to the Keplerian critical rotation rate of the
        models used to compute the instability strip
        Must be one of '0.0', '0.25', '0.50'
        Defaults to '0.0'
    IStype:
        Which instability strip(s) to plot
        Must be one of 'both', 'p-mode', 'g-mode'
        Defaults to 'both'
    p_mode_col: string
        Matplotlib's colour to use for the lines indicating the beta Cep
        instability strip
        Defaults to 'k'
    p_mode_ls: string
        Matplotlib's linestyle to use for the lines indicating the beta Cep
        instability strip
        Defaults to 'solid'
    g_mode_col: string
        Matplotlib's colour to use for the lines indicating the SPB
        instability strip
        Defaults to 'k'
    g_mode_ls: string
        Matplotlib's linestyle to use for the lines indicating the SPB
        instability strip
        Defaults to 'dashed'
    zorder: float
        Matplotlib's zorder to set the drawing order of the elements on an axes
        Defaults to 0
    lw: float
        Matplotlib's linewidth to set the width of the lines indicating
        the instability strips
        Defaults to 0.5
    fill: bool
        Whether to fill the space between the red and blue edges of the
        instability strip(s) with coloured hatching
        Defaults to False
    alpha: float
        Matplotlib's alpha to set the transparancy of the filling
        Defaults to 0.2
    """
    # Spec Lum always in files, logg or logL not

    if y_axis == 'logLsp':
        Lum_min = 'logLsp_min'
        Lum_max = 'logLsp_max'
    elif y_axis == 'logL':
        Lum_min = 'logL_min'
        Lum_max = 'logL_max'

    # -----Read data------ #

    header = [('Mass', float), ('logTe_min', float), ('logL_min', float),
              ('logLsp_min', float), ('logTe_max', float), ('logL_max', float),
              ('logLsp_max', float)]

    if Zini == '0.014':
        if rot == '0.0':
            inst_file_g = 'instability_edges_g_modes_SB19_Z0014.txt'
            inst_file_p = 'instability_edges_p_modes_SB19_Z0014.txt'
        elif rot == '0.25':
            inst_file_g = 'instability_edges_g_modes_SB19_Z0014_rot_25_roche.txt'
            inst_file_p = 'instability_edges_p_modes_SB19_Z0014_rot_25_roche.txt'
        elif rot == '0.50':
            inst_file_g = 'instability_edges_g_modes_SB19_Z0014_rot_50_roche.txt'
            inst_file_p = 'instability_edges_p_modes_SB19_Z0014_rot_50_roche.txt'
    elif Zini == '0.020':
        if rot == '0.0':
            inst_file_g = 'instability_edges_g_modes_SB19_Z0020_solar.txt'
            inst_file_p = 'instability_edges_p_modes_SB19_Z0020_solar.txt'
        else:
            print("Rotation setting not available. Exiting...")
            quit()
    elif Zini == '0.010':
        if rot == '0.0':
            inst_file_g = 'instability_edges_g_modes_SB19_Z0010_solar.txt'
            inst_file_p = 'instability_edges_p_modes_SB19_Z0010_solar.txt'
        else:
            print("Rotation setting not available. Exiting...")
            quit()
    else:
        print("Selected metallicity not available. Using solar.")
        inst_file_g = 'instability_edges_g_modes_SB19_solar.txt'
        inst_file_p = 'instability_edges_p_modes_SB19_solar.txt'

    data_g_modes = np.loadtxt(datadir + inst_file_g, dtype=header)
    data_p_modes = np.loadtxt(datadir + inst_file_p, dtype=header)

    # -----Plot boundaries------ #

    if IStype == 'both':
        ax.plot(data_g_modes['logTe_min'], data_g_modes[Lum_min],
                c=g_mode_col, ls=g_mode_ls, lw=lw, zorder=zorder)
        ax.plot(data_g_modes['logTe_max'], data_g_modes[Lum_max],
                c=g_mode_col, ls=g_mode_ls, lw=lw, zorder=zorder,
                label='g-mode IS')

        ax.plot(data_p_modes['logTe_min'], data_p_modes[Lum_min],
                c=p_mode_col, ls=p_mode_ls, lw=lw, zorder=zorder)
        ax.plot(data_p_modes['logTe_max'], data_p_modes[Lum_max],
                c=p_mode_col, ls=p_mode_ls, lw=lw, zorder=zorder,
                label='p-mode IS')

        tot_x = np.concatenate([data_g_modes['logTe_min'],
                                data_g_modes['logTe_max'][::-1]])
        tot_y = np.concatenate([data_g_modes[Lum_min],
                                data_g_modes[Lum_max][::-1]])

        if fill:
            ax.fill_between(tot_x, tot_y, min(tot_y), interpolate=True,
                            facecolor="darkorange", hatch="xxxxxxxx",
                            edgecolor=g_mode_col, linewidth=0.0, alpha=alpha,
                            label=r'g modes ($l$=1-2)', zorder=zorder)

        tot_x = np.concatenate([data_p_modes['logTe_min'],
                                data_p_modes['logTe_max'][::-1]])
        tot_y = np.concatenate([data_p_modes[Lum_min],
                                data_p_modes[Lum_max][::-1]])

        if fill:
            ax.fill_between(tot_x, tot_y, min(tot_y), interpolate=True,
                            facecolor="cyan", hatch="xxxxxxxx",
                            edgecolor=g_mode_col, linewidth=0.0, alpha=alpha,
                            label=r'p modes ($l$=0-2)', zorder=zorder)

    elif IStype == 'p-mode':
        ax.plot(data_p_modes['logTe_min'], data_p_modes[Lum_min], c=p_mode_col,
                linestyle=p_mode_ls, lw=lw, zorder=zorder)
        ax.plot(data_p_modes['logTe_max'], data_p_modes[Lum_max], c=p_mode_col,
                linestyle=p_mode_ls, lw=lw, zorder=zorder)

        tot_x = np.concatenate([data_p_modes['logTe_min'],
                                data_p_modes['logTe_max'][::-1]])
        tot_y = np.concatenate([data_p_modes[Lum_min],
                                data_p_modes[Lum_max][::-1]])

        if fill:
            ax.fill_between(tot_x, tot_y, min(tot_y), interpolate=True,
                            facecolor="cyan", hatch="xxxxxxxx",
                            edgecolor=p_mode_col, linewidth=0.0, alpha=alpha,
                            label=r'p modes ($l$=0-2)', zorder=zorder)

    elif IStype == 'g-mode':
        ax.plot(data_g_modes['logTe_min'], data_g_modes[Lum_min], c=g_mode_col,
                linestyle=g_mode_ls, lw=lw, zorder=zorder)
        ax.plot(data_g_modes['logTe_max'], data_g_modes[Lum_max], c=g_mode_col,
                linestyle=g_mode_ls, lw=lw, zorder=zorder)

        tot_x = np.concatenate([data_g_modes['logTe_min'],
                                data_g_modes['logTe_max'][::-1]])
        tot_y = np.concatenate([data_g_modes[Lum_min],
                                data_g_modes[Lum_max][::-1]])

        if fill:
            ax.fill_between(tot_x, tot_y, min(tot_y), interpolate=True,
                            facecolor="darkorange", hatch="xxxxxxxx",
                            edgecolor=g_mode_col, linewidth=0.0, alpha=alpha,
                            label=r'g modes ($l$=1-2)', zorder=zorder)


##### Hertzsprung-Russell diagrams #############################################


def plot_all_chi2_in_HRD(results, observation, n_pg_main,
                         show_best_model=True, show_statistical_model=False,
                         show_IS=False, log=False,
                         hollow_spectro_incompatible=False, title="",
                         colourmap="plasma_r", cmin=None, cmax=None,
                         save=False, save_dir="plots/", save_name="chi2_HRD",
                         show=True) :
    """
    Plot a Hertzsprung-Russell diagram showing the model matching the first
    observed frequency of each evolutionary track, coloured by their chi-square.
    The observations are shown in black, alongside the best model in the set of
    provided results in cyan, if requested, and the statistical model in purple,
    if requested. The p-mode instability strip of Burssens et al. (2020) is
    included if requested.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    results: list of dicts
        The dictionaries describing all the results of comparing the given
        observation to the grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    n_pg_main: int
        Radial order of the fixed main mode.
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    show_IS: bool
        Whether to show the p-mode instability strip of Burssens et al. (2020)
        Defaults to False
    log: bool
        Whether to plot chi2 on a log scale
        Defaults to False
    hollow_spectro_incompatible: bool
        Whether to plot the models that are not compatible with the observed
        Teff and L using hollow symbols
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    colourmap: string
        Name of the matplotlib colourmap to employ. Choose from the colourmaps
        described here:
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
        Defaults to "plasma"
    cmin: float
        The lower limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated lower limit
    cmax: float
        The upper limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated upper limit
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    ax: matplotlib.axes.Axes
        Axes object encapsulating all elements of this plot
    """
    fig, ax = plt.subplots(1,1,figsize=(10,6))

    log_Teffs = np.array([model["log_Teff"] for model in results["models"]])
    log_Ls = np.array([model["log_L"] for model in results["models"]])
    chi2s = np.array(results["chi2s"])

    if log :
        cols = np.log10(chi2s)
        label = r"$\log{( \chi^2 )}$"
    else :
        cols = chi2s
        label = r"$\chi^2$"
    if cmin is None :
        cmin = np.nanmin(cols)
    if cmax is None :
        cmax = np.nanmax(cols)

    # Highlighting the observation
    ax.errorbar(observation["log_Teff"], observation["log_L"],
                xerr=observation["log_Teff_err"], yerr=observation["log_L_err"],
                marker="s", color="black", markersize=5, zorder=2,
                label="observation")

    if show_statistical_model :
        # Highlighting the statistical model
        model = results["statistical_model"]
        ax.errorbar(x=model["log_Teff_mean"], y=model["log_L_mean"],
                    xerr=model["log_Teff_std"], yerr=model["log_L_std"],
                    marker="*", color="crimson", markersize=10, zorder=3,
                    label="statistical model")
    if show_best_model :
        # Highlighting the best fitting model
        model = results["best_model"]
        if not type(model) == float :
            ax.scatter(model["log_Teff"], model["log_L"], marker="*",
                       color="cyan", s=150, zorder=3, label="best model")

    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        cmap = plt.get_cmap(colourmap)
        norm = plt.Normalize(cmin, cmax)
        ax.scatter(log_Teffs[~compatible], log_Ls[~compatible],
                   c=cmap(norm(cols[~compatible])), fc='none',
                   ec=cmap(norm(cols[~compatible])), lw=1, zorder=1)
        c = ax.scatter(log_Teffs[compatible], log_Ls[compatible],
                       c=cmap(norm(cols[compatible])), cmap=colourmap,
                       vmin=cmin, vmax=cmax, zorder=1)
        c.set_cmap(colourmap)
        c.set_clim(cmin, cmax)
        fig.colorbar(c, label=label)
    else :
        c = ax.scatter(log_Teffs, log_Ls, c=cols, cmap=colourmap, vmin=cmin,
                       vmax=cmax, zorder=1)
        fig.colorbar(c, label=label)

    ax.invert_xaxis()

    if show_IS :
        add_instability_strip(ax, datadir='code/instability_strips/',
                              zorder=0, Zini='0.014', rot='0.0', alpha=0.2,
                              y_axis='logL', IStype='p-mode',
                              p_mode_col='k', fill=False, p_mode_ls='dotted',
                              g_mode_col='k', g_mode_ls='dashed', lw=.5)

    ax.legend(fontsize=10)
    ax.set_xlabel("log(Teff) [K]", fontsize=10)
    ax.set_ylabel(r"log(L) [L$_\odot$]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{observation["name"]}_{save_name}_npg{n_pg_main}_{observation["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, ax


def plot_models_and_observed_HRD(models, observation,
                                 use_statistical_model=False, show_IS=False,
                                 labels=[""], title="", save=False,
                                 save_dir="plots/", save_name="comparing_HRD",
                                 show=True) :
    """
    Plot a Hertzsprung-Russell diagram showing the observed effective
    temperature and luminosity and their uncertainties in black, alongside the
    provided MESA models and the corresponding evolutionary tracks. The p-mode
    instability strip of Burssens et al. (2020) is included if requested.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    models: list of dicts
        The dictionaries describing the particular models as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    use_statistical_model: bool
        Whether the provided models are statistical models, meaning their Teff
        and L are means of different models and have an std. If True, we use
        those and show the standard deviations as errorbars.
        Defaults to False
    show_IS: bool
        Whether to show the p-mode instability strip of Burssens et al. (2020)
        Defaults to False
    labels: list of strings
        The labels to be given to each model. Should be of the same length as
        models.
        Defaults to [""]
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    ax: matplotlib.axes.Axes
        Axes object encapsulating all elements of this plot
    """
    fig, ax = plt.subplots(1,1,figsize=(10,6))

    colours_dot = ["gold", "deeppink", "forestgreen", "darksalmon",
                   "navy", "darkviolet"]
    colours_track = ["orange", "hotpink", "limegreen", "lightsalmon",
                     "mediumblue", "violet"]

    ax.errorbar(observation["log_Teff"], observation["log_L"],
                xerr=observation["log_Teff_err"], yerr=observation["log_L_err"],
                marker="s", color="black", markersize=5, zorder=2)
    for k,model in enumerate(models) :
        if not type(model) == dict :
            continue
        if use_statistical_model :
            ax.errorbar(x=model["log_Teff_mean"], y=model["log_L_mean"],
                        xerr=model["log_Teff_std"], yerr=model["log_L_std"],
                        marker="*", color=colours_dot[k%len(colours_dot)],
                        markersize=10, zorder=3, label=labels[k])
        else :
            ax.scatter(model["log_Teff"], model["log_L"], marker="*",
                       color=colours_dot[k%len(colours_dot)], s=150, zorder=3,
                       label=labels[k])
        log_Teff_history, log_L_history = read_MESA_history_columns(model["histfile"],
                                                                    ["log_Teff", "log_L"],
                                                                    Xc_min=model["Xc_min"],
                                                                    Xc_max=model["Xc_max"])

        ax.plot(log_Teff_history, log_L_history,
                color=colours_track[k%len(colours_track)],
                linestyle="solid", zorder=1)

    ax.invert_xaxis()

    if show_IS:
        add_instability_strip(ax, datadir='code/instability_strips/',
                              zorder=0, Zini='0.014', rot='0.0', alpha=0.2,
                              y_axis='logL', IStype='p-mode',
                              p_mode_col='k', fill=False, p_mode_ls='dotted',
                              g_mode_col='k', g_mode_ls='dashed', lw=.5)

    ax.legend(fontsize=10)
    ax.set_xlabel("log(Teff) [K]", fontsize=10)
    ax.set_ylabel(r"log(L) [L$_\odot$]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{observation["name"]}_{save_name}_{observation["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, ax



##### Parameter correlations and distributions #################################


def plot_chi2_against_grid_params(grid, results, n_pg_main, show_best_model=True,
                                  show_statistical_model=False, log=False,
                                  hollow_spectro_incompatible=False, title="",
                                  save=False, save_dir="plots/",
                                  save_name="chi2_grid_params", show=True) :
    """
    Plot the chi2 against the initial mass, envelope mixing strength and
    overshoot parameter for each evolutionary track in the grid. The best model
    in the set of provided results is shown in cyan, if requested, and the
    statistical model as a purple shaded area between the mean and +- its
    standard deviation, if requested.


    Author: Mathijs Vanrespaille

    ------- Parameters -------
    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    n_pg_main: int
        Radial order of the fixed main mode.
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    log: bool
        Whether to plot chi2 on a log scale
        Defaults to False
    hollow_spectro_incompatible: bool
        Whether to plot the models that are not compatible with the observed
        Teff and L using hollow symbols
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot

    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    chi2s = np.array(results["chi2s"])
    Ms = np.array(grid["Ms"])
    logDs = np.array(grid["logDs"])
    fovs = np.array(grid["fovs"])
    index = results["best_model_index"]

    ax = axs[0]
    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        ax.scatter(Ms[~compatible], chi2s[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(Ms[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(Ms, chi2s, color="blue")
    if show_best_model and not type(index) == float :
        ax.scatter(Ms[index], chi2s[index], marker="*", color="cyan", s=150,
                   zorder=3, label="best model")
    if show_statistical_model :
        statistical_model = results["statistical_model"]
        mean = statistical_model["M_mean"]
        std = statistical_model["M_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"initial mass [M$_\odot$]", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if log :
        ax.set_yscale("log")

    ax = axs[1]
    ax.set_title(title, fontsize=12)
    if hollow_spectro_incompatible :
        ax.scatter(logDs[~compatible], chi2s[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(logDs[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(logDs, chi2s, color="blue")
    if show_best_model and not type(index) == float :
        ax.scatter(logDs[index], chi2s[index], marker="*", color="cyan", s=150,
                   zorder=3, label="best model")
    if show_statistical_model :
        mean = statistical_model["logD_mean"]
        std = statistical_model["logD_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"$\log{D_\mathrm{mix}}$", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if log :
        ax.set_yscale("log")

    ax = axs[2]
    if hollow_spectro_incompatible :
        ax.scatter(fovs[~compatible], chi2s[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(fovs[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(fovs, chi2s, color="blue")
    if show_best_model and not type(index) == float :
        ax.scatter(fovs[index], chi2s[index], marker="*", color="cyan", s=150,
                   zorder=3, label="best model")
    if show_statistical_model :
        mean = statistical_model["fov_mean"]
        std = statistical_model["fov_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"f$_\mathrm{ov}$", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if log :
        ax.set_yscale("log")

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_npg{n_pg_main}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, axs


def plot_chi2_against_model_params(results, show_best_model=True,
                                   show_statistical_model=False, chi2_log=False,
                                   age_log=False,
                                   hollow_spectro_incompatible=False,
                                   title="", save=False, save_dir="plots/",
                                   save_name="model_scatter", show=True) :
    """
    Scatterplots of chi2 against each evolutionary track's model optimised
    central hydrogen fraction, age in years and selected rotation rate. The best
    model is highlighted by cyan star, if requested, and the statistical model
    as a purple shaded area between the mean and +- its standard deviation, if
    requested.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    chi2_log: bool
        Whether to plot chi2 on a log scale
        Defaults to False
    age_log: bool
        Whether to plot the age on a log scale
    hollow_spectro_incompatible: bool
        Whether to plot the models that are not compatible with the observed
        Teff and L using hollow symbols
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot

    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    chi2s = np.array(results["chi2s"])
    Xcs_cross = np.array(results["Xcs_cross"])
    ages_cross = np.array(results["ages_cross"])
    frots_best = np.array(results["frots_best"])
    if hollow_spectro_incompatible :
        compatible = np.array(results["spectro_compatible"])

    ax = axs[0]
    if hollow_spectro_incompatible :
        ax.scatter(Xcs_cross[~compatible], chi2s[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(Xcs_cross[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(Xcs_cross, chi2s, color="blue")
    if show_best_model :
        model_best = results["best_model"]
        if not type(model_best) == float :
            chi2_best = results["chi2s"][results["best_model_index"]]
            ax.scatter(model_best["Xc"], chi2_best, marker="*", color="cyan",
                       s=150, zorder=3)
    if show_statistical_model :
        statistical_model = results["statistical_model"]
        mean = statistical_model["Xc_mean"]
        std = statistical_model["Xc_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"Xc", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if chi2_log :
        ax.set_yscale("log")

    ax = axs[1]
    ax.set_title(title, fontsize=12)
    if hollow_spectro_incompatible :
        ax.scatter(ages_cross[~compatible], chi2s[~compatible],
                   facecolor="none", color="blue")
        ax.scatter(ages_cross[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(ages_cross, chi2s, color="blue")
    if show_best_model :
        ax.scatter(model_best["age"], chi2_best, marker="*", color="cyan",
                   s=150, zorder=3)
    if show_statistical_model :
        mean = statistical_model["age_mean"]
        std = statistical_model["age_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"age [yr]", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if chi2_log :
        ax.set_yscale("log")
    if age_log :
        ax.set_xscale("log")

    ax = axs[2]
    if hollow_spectro_incompatible :
        ax.scatter(frots_best[~compatible], chi2s[~compatible],
                   facecolor="none", color="blue")
        ax.scatter(frots_best[compatible], chi2s[compatible], color="blue")
    else :
        ax.scatter(frots_best, chi2s, color="blue")
    if show_best_model :
        ax.scatter(model_best["frot_best"], chi2_best, marker="*", color="cyan",
                   s=150, zorder=3)
    if show_statistical_model :
        mean = statistical_model["frot_mean"]
        std = statistical_model["frot_std"]
        ax.axvspan(mean-std, mean+std, alpha=0.5, color="crimson")
    ax.set_xlabel(r"$f_\mathrm{rot}$ [c/d]", fontsize=12)
    ax.set_ylabel(r"$\chi^2$", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()
    if chi2_log :
        ax.set_yscale("log")

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return


def plot_chi2_coloured_grid_params(grid, results, show_best_model=True,
                                   show_statistical_model=False, log=False,
                                   only_plot_compatible=False, title="",
                                   colourmap="plasma_r", cmin=None, cmax=None,
                                   save=False, save_dir="plots/",
                                   save_name="chi2_coloured_grid_params",
                                   show=True) :
    """
    Scatterplots between mass, envelope mixing strength and overshooting
    parameters coloured by chi2. The scattered dots are ordered such that the
    best chi2 is always plotted on top. The best model is highlighted by a cyan
    star, if requested, and the statistical model and its standard deviation is
    plotted as a purple star, if requested.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    log: bool
        Whether to plot chi2 on a log scale
        Defaults to False
    only_plot_compatible: bool
        Whether to exclude the best models that do not match the effective
        temperature and luminosity observations
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    colourmap: string
        Name of the matplotlib colourmap to employ. Choose from the colourmaps
        described here:
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
        Defaults to "plasma"
    cmin: float
        The lower limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated lower limit
    cmax: float
        The upper limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated upper limit
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot
    """

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    chi2s = np.array(results["chi2s"])
    Ms = np.array(grid["Ms"])
    logDs = np.array(grid["logDs"])
    fovs = np.array(grid["fovs"])

    index = results["best_model_index"]
    if show_best_model and not type(index) == float :
        M_best = Ms[index]
        logD_best = logDs[index]
        fov_best = fovs[index]
    if show_statistical_model :
        statistical_model = results["statistical_model"]
        M_mean = statistical_model["M_mean"]
        M_std = statistical_model["M_std"]
        logD_mean = statistical_model["logD_mean"]
        logD_std = statistical_model["logD_std"]
        fov_mean = statistical_model["fov_mean"]
        fov_std = statistical_model["fov_std"]

    I = np.argsort(-chi2s) # Indices to sort from largest to smallest chi2
    chi2s = chi2s[I]
    Ms = Ms[I]
    logDs = logDs[I]
    fovs = fovs[I]

    if only_plot_compatible :
        compatible = np.array(results["spectro_compatible"])[I]
        chi2s = chi2s[compatible]
        Ms = Ms[compatible]
        logDs = logDs[compatible]
        fovs = fovs[compatible]

    ax = axs[0]
    if log :
        ax.scatter(logDs, Ms, c=np.log10(chi2s), cmap=colourmap, vmin=cmin,
                   vmax=cmax, zorder=1, s=20)
    else :
        ax.scatter(logDs, Ms, c=chi2s, cmap=colourmap, vmin=cmin, vmax=cmax,
                   zorder=1, s=20)
    if show_best_model :
        ax.scatter(logD_best, M_best, color="cyan", marker="*", s=100)
    if show_best_model and not type(index) == float :
        ax.errorbar(x=logD_mean, y=M_mean, xerr=logD_std, yerr=M_std,
                    color="crimson", marker="*", markersize=10)
    ax.set_xlabel(r"$\log{(D_\mathrm{mix})}$ [$cm^2/s$]", fontsize=10)
    ax.set_ylabel(r"M [M$_\odot$]", fontsize=10)

    ax = axs[1]
    ax.set_title(title, fontsize=12)
    if log :
        ax.scatter(fovs, Ms, c=np.log10(chi2s), cmap=colourmap, vmin=cmin,
                   vmax=cmax, zorder=1, s=20)
    else :
        ax.scatter(fovs, Ms, c=chi2s, cmap=colourmap, vmin=cmin, vmax=cmax,
                   zorder=1, s=20)
    if show_best_model and not type(index) == float :
        ax.scatter(fov_best, M_best, color="cyan", marker="*", s=100)
    if show_statistical_model :
        ax.errorbar(x=fov_mean, y=M_mean, xerr=fov_std, yerr=M_std,
                    color="crimson", marker="*", markersize=10)
    ax.set_xlabel(r"f$_\mathrm{ov}$", fontsize=10)
    ax.set_ylabel(r"M [M$_\odot$]", fontsize=10)

    ax = axs[2]
    if log :
        c = ax.scatter(logDs, fovs, c=np.log10(chi2s), cmap=colourmap,
                       vmin=cmin, vmax=cmax, zorder=1, s=20)
    else :
        c = ax.scatter(logDs, fovs, c=chi2s, cmap=colourmap, vmin=cmin,
                       vmax=cmax, zorder=1, s=20)
    if show_best_model and not type(index) == float :
        ax.scatter(logD_best, fov_best, color="cyan", marker="*", s=100)
    if show_statistical_model :
        ax.errorbar(x=logD_mean, y=fov_mean, xerr=logD_std, yerr=fov_std,
                    color="crimson", marker="*", markersize=10)
    ax.set_xlabel(r"$\log{(D_\mathrm{mix})}$ [$cm^2/s$]", fontsize=10)
    ax.set_ylabel(r"f$_\mathrm{ov}$", fontsize=10)

    if log :
        fig.colorbar(c, label=r"$\log{( \chi^2 )}$")
    else :
        fig.colorbar(c, label=r"$\chi^2$")

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, axs


def plot_grid_params_chi2_corner(grid, results, show_best_model=True,
                                 show_statistical_model=False, log=False,
                                 hollow_spectro_incompatible=False, title="",
                                 colourmap="plasma_r", cmin=None, cmax=None,
                                 save=False, save_dir="plots/",
                                 save_name="chi2_coloured_grid_params",
                                 show=True) :
    """
    A custom corner plot the relations betweem mass, envelope mixing strength
    and overshooting parameters and chi2. The scattered dots are ordered such
    that the best chi2 is always plotted on top. The best model is highlighted
    by a cyan star

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    log: bool
        Whether to plot the chi2 colourbar as a log scale
        Defaults to False
    hollow_spectro_incompatible: bool
        Whether to plot the models that are not compatible with the observed
        Teff and L using hollow symbols
        Defaults to False
    show_best_model: bool
        Whether to show the best fitting model.
        Defaults to True
    show_statistical_model: bool
        Whether to show the statistical model and its the standard deviations.
        Defaults to False
    log: bool
        Whether to plot chi2 on a log scale
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    colourmap: string
        Name of the matplotlib colourmap to employ. Choose from the colourmaps
        described here:
        https://matplotlib.org/stable/users/explain/colors/colormaps.html
        Defaults to "plasma"
    cmin: float
        The lower limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated lower limit
    cmax: float
        The upper limit of the colourbar.
        Defaults to None, which uses matplotlib's calculated upper limit
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot

    """

    fig, axs = plt.subplots(3, 3, figsize=(15,15))

    chi2s = np.array(results["chi2s"])
    Ms = grid["Ms"]
    logDs = grid["logDs"]
    fovs = grid["fovs"]

    index = results["best_model_index"]
    if show_best_model and not type(index) == float :
        chi2_best = chi2s[index]
        data_best = [Ms[index], logDs[index], fovs[index]]
    if show_statistical_model :
        statistical_model = results["statistical_model"]
        data_means = [statistical_model["M_mean"],
                      statistical_model["logD_mean"],
                      statistical_model["fov_mean"]]
        data_stds = [statistical_model["M_std"],
                     statistical_model["logD_std"],
                     statistical_model["fov_std"]]

    I_toKeep = ~np.isnan(chi2s)
    chi2s = chi2s[I_toKeep]
    Ms = Ms[I_toKeep]
    logDs = logDs[I_toKeep]
    fovs = fovs[I_toKeep]

    I = np.argsort(-chi2s) # Indices to sort from largest to smallest chi2
    chi2s = chi2s[I]
    data = [Ms[I], logDs[I], fovs[I]]
    labels = [r"M [M$_\odot$]", r"$\log{(D_\mathrm{mix})}$ [$cm^2/s$]",
              r"f$_\mathrm{ov}$"]

    for i in range(3) :
        for j in range(3) :
            ax = axs[j][i]
            if i == j :
                if hollow_spectro_incompatible :
                    compatible = results["spectro_compatible"][I]
                    ax.scatter(data[i][~compatible], chi2s[~compatible],
                               fc='none', lw=1, zorder=1, color="blue", s=20)
                    ax.scatter(data[i][compatible], chi2s[compatible], zorder=1,
                               color="blue", s=20)
                else :
                    ax.scatter(data[i], chi2s, color="blue", s=20)
                if show_best_model :
                    ax.scatter(data_best[i], chi2_best, color="cyan",
                               marker="*", s=100)
                if show_statistical_model :
                    ax.axvspan(data_means[i]-data_stds[i],
                               data_means[i]+data_stds[i],
                               alpha=0.5, color="crimson")
                ax.set_xlabel(labels[i], fontsize=10)
                ax.set_ylabel(r"$\chi^2$", fontsize=10)
                ax.tick_params(bottom=True, left=True, top=True, right=True,
                               which='both')
                ax.minorticks_on()
                if log :
                    ax.set_yscale("log")

            else :
                if log :
                    cols = np.log10(chi2s)
                else :
                    cols = chi2s
                if cmin is None :
                    cmin = np.nanmin(cols)
                if cmax is None :
                    cmax = np.nanmax(cols)
                if hollow_spectro_incompatible :
                    cmap = plt.get_cmap(colourmap)
                    norm = plt.Normalize(cmin, cmax)
                    ax.scatter(data[i][~compatible], data[j][~compatible],
                               c=cmap(norm(cols[~compatible])), fc='none',
                               ec=cmap(norm(cols[~compatible])), lw=1, zorder=1)
                    c = ax.scatter(data[i][compatible], data[j][compatible],
                                   c=cmap(norm(cols[compatible])),
                                   cmap=colourmap, vmin=cmin, vmax=cmax,
                                   zorder=1)
                    c.set_cmap(colourmap)
                    c.set_clim(cmin, cmax)
                else :
                    c = ax.scatter(data[i], data[j], c=cols, cmap=colourmap,
                                   vmin=cmin, vmax=cmax, zorder=1, s=20)
                if show_best_model :
                    ax.scatter(data_best[i], data_best[j], color="cyan",
                               marker="*", s=100)
                if show_statistical_model :
                    ax.errorbar(x=data_means[i], y=data_means[j],
                                xerr=data_stds[i], yerr=data_stds[j],
                                color="crimson", marker="*", markersize=10)
                ax.set_xlabel(labels[i], fontsize=10)
                ax.set_ylabel(labels[j], fontsize=10)

    axs[0][1].set_title(title, fontsize=12)

    plt.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return


def plot_model_params_histograms(results, age_log=False,
                                 hollow_spectro_incompatible=False, title="",
                                 save=False, save_dir="plots/",
                                 save_name="model_histograms", show=True) :
    """
    Plot histograms of each evolutionary track's model matching the main mode's
    frequency against central hydrogen fraction, age in years and selected
    rotation rate. The best model is highlighted by a dashed cyan vertical line.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    age_log: bool
        Whether to plot the age in the second panel on a log scale
        Defaults to False
    hollow_spectro_incompatible: bool
        Whether to separate the models incompatible with the observed Teff and L
        and plot them as a hollow histogram on top of the compatible models as a
        solid histogram
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot

    """

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    ax = axs[0]
    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        (n, bins, patches) = ax.hist(results["Xcs_cross"], bins=50, fill=False,
                                     color="navy")
        ax.hist(results["Xcs_cross"][compatible], bins=bins, fill=True,
                color="navy")
    else :
        ax.hist(results["Xcs_cross"], bins=50, color="navy")
    ax.set_xlabel("Xc", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.axvline(results["best_model"]["Xc"], linestyle='dashed', color='cyan')
    ax.minorticks_on()

    ax = axs[1]
    if age_log :
        t = np.log10(results["ages_cross"])
        t_best = np.log10(results["best_model"]["age"])
        label = "log(age) [yr]"
    else :
        t = results["ages_cross"]
        t_best = results["best_model"]["age"]
        label = "age [yr]"
    if hollow_spectro_incompatible :
        (n, bins, patches) = ax.hist(t, bins=50, fill=False, color="navy")
        ax.hist(t[compatible], bins=bins, fill=True, color="navy")
    else :
        ax.hist(t, bins=50, color="navy")
    ax.axvline(t_best, linestyle='dashed', color='cyan')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.minorticks_on()

    ax = axs[2]
    if hollow_spectro_incompatible :
        (n, bins, patches) = ax.hist(results["frots_best"], bins=50, fill=False,
                                     color="navy")
        ax.hist(results["frots_best"][compatible], bins=bins, fill=True,
                color="navy")
    else :
        ax.hist(results["frots_best"], bins=50, color="navy")
    ax.set_xlabel(r"$f_\mathrm{rot}$ [c/d]", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.minorticks_on()

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()


def plot_Xc_grid_params(grid, results, show_best_model=True,
                        show_statistical_model=False,
                        hollow_spectro_incompatible=False, title="", save=False,
                        save_dir="plots/", save_name="model_histograms",
                        show=True) :
    """
    Plot histograms of each evolutionary track's model matching the main mode
    frequency against central hydrogen fraction, age in years and selected
    rotation rate. The best model is highlighted by a cyan star, if requested,
    and the statistical model in purple, if requested.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    results: list of dicts
        The dictionaries describing all the results of comparing a given
        observation to the grid
    age_log: bool
        Whether to plot the age in the second panel on a log scale
        Defaults to False
    hollow_spectro_incompatible: bool
        Whether to separate the models incompatible with the observed Teff and L
        and plot them as a hollow histogram on top of the compatible models as a
        solid histogram
        Defaults to False
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and the
        "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_HRD"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    axs: array of matplotlib.axes.Axes
        Array of three Axes object encapsulating all elements of this plot

    """
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    Xcs = results["Xcs_cross"]
    Ms = grid["Ms"]
    logDs = grid["logDs"]
    fovs = grid["fovs"]

    index = results["best_model_index"]
    if show_best_model and not type(index) == float :
        Xc_best = Xcs[index]
        M_best = Ms[index]
        logD_best = logDs[index]
        fov_best = fovs[index]
    if show_statistical_model :
        statistical_model = results["statistical_model"]
        Xc_mean = statistical_model["Xc_mean"]
        Xc_std = statistical_model["Xc_std"]
        M_mean = statistical_model["M_mean"]
        M_std = statistical_model["M_std"]
        logD_mean = statistical_model["logD_mean"]
        logD_std = statistical_model["logD_std"]
        fov_mean = statistical_model["fov_mean"]
        fov_std = statistical_model["fov_std"]

    ax = axs[0]
    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        ax.scatter(Ms[~compatible], Xcs[~compatible], facecolor="none",
                   edgecolor="blue")
        ax.scatter(Ms[compatible], Xcs[compatible], color="blue")
    else :
        ax.scatter(Ms, Xcs, color="blue")
    if show_best_model :
        ax.scatter(M_best, Xc_best, marker="*", color="cyan", s=150, zorder=3)
    if show_statistical_model :
        ax.errorbar(x=M_mean, y=Xc_mean, xerr=M_std, yerr=Xc_std, marker="*",
                    color="crimson", markersize=10, zorder=3)
    ax.set_xlabel(r"M [M$_\odot$]", fontsize=12)
    ax.set_ylabel(r"Xc", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()

    ax = axs[1]
    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        ax.scatter(logDs[~compatible], Xcs[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(logDs[compatible], Xcs[compatible], color="blue")
    else :
        ax.scatter(logDs, Xcs, color="blue")
    if show_best_model :
        ax.scatter(logD_best, Xc_best, marker="*", color="cyan", s=150,
                   zorder=3)
    if show_statistical_model :
        ax.errorbar(x=logD_mean, y=Xc_mean, xerr=logD_std, yerr=Xc_std,
                    marker="*", color="crimson", markersize=10, zorder=3)
    ax.set_xlabel(r"$\log{(D_\mathrm{mix})}$ [$cm^2/s$]", fontsize=12)
    ax.set_ylabel(r"Xc", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()

    ax = axs[2]
    if hollow_spectro_incompatible :
        compatible = results["spectro_compatible"]
        ax.scatter(fovs[~compatible], Xcs[~compatible], facecolor="none",
                   color="blue")
        ax.scatter(fovs[compatible], Xcs[compatible], color="blue")
    else :
        ax.scatter(fovs, Xcs, color="blue")
    if show_best_model :
        ax.scatter(fov_best, Xc_best, marker="*", color="cyan", s=150, zorder=3)
    if show_statistical_model :
        ax.errorbar(x=fov_mean, y=Xc_mean, xerr=fov_std, yerr=Xc_std,
                    marker="*", color="crimson", markersize=10, zorder=3)
    ax.set_xlabel(r"f$_\mathrm{ov}$", fontsize=12)
    ax.set_ylabel(r"Xc", fontsize=12)
    ax.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax.minorticks_on()

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{results["name"]}_{save_name}_{results["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()



##### Evaluating model quality #################################################


def plot_model_and_observed_frequencies(model, observation, n_pg_main, title="",
                                        save=False, save_dir="plots/",
                                        save_name="comparing_frequencies",
                                        show=True) :
    """
    Plot all observed frequencies alongside all potential frequencies predicted
    for the given model. Observed frequencies and the square root of their
    amplitudes are shown as solid grey vertical lines, with the identified modes
    highlighted in black. The identified modes' lines are topped with a symbol
    whose shape represents the degree: dots for radial modes, triangles for
    dipole modes and pentagons for quintuplets. The model's predicted
    frequencies are shown as vertical lines covering the full plot as their
    amplitudes are unknown, containing a symbol representing the degree at a
    y-value representing the radial order shown on the right axis. The linestyle
    also shows the degree: dashed lines for radial degrees, dotted for dipoles
    and dashdotted for quintuplets. Similar lines in light green and olive
    connect frequencies belonging to the same predicted or identified observed
    modes. Likewise, the radial order is also indicated by the colour, with
    g-modes in blueish tints, specifically n_pg=-3 in navy, -2 in blue and
    -1 in dodgerblue, while f-modes are in purple and p-modes in reddish tints,
    specifically n_pg=1 in salmon, 2 in tomato and 3 in red. If the observations
    provide the radial order of the identified modes, these will use the same
    colours scheme.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    n_pg_main: int
        Radial order of the fixed main mode.
    title: string
        Title of the plot
        Defaults to empty string
    save: bool
        Whether to save the figure
        Defaults to False
    save_dir: string
        Path to the directory where the figure is to be saved. Only relevant if
        save is True
        Defaults to "plots/"
    save_name: string
        The filename to use when saving the figure. Only relevant if save is
        True. The observations "name" is appended as a prefix and
        the "save_suffix" as a suffix, alongside ".png".
        Defaults to "comparing_frequencies"
    show: bool
        Whether to show the plot
        Defaults to True

    ------- Returns -------
    fig: matplotlib.figure.Figure
        Figure object as the top level container of all the plot elements drawn
        in this function
    ax1: matplotlib.axes.Axes
        Axes object encapsulating the elements of this plot using the left
        y-axis holding the square root of the amplitude
    ax2: matplotlib.axes.Axes
        Axes object encapsulating the elements of this plot using the right
        y-axis holding the radial order
    """

    # Marker based on mode degree
    markers = {0:'o', 1:'^', 2:'p'}
    # Linestyles based on mode degree
    linestyles = {0:"dashed", 1:"dotted", 2:"dashdot"}
    # Colour based on radial order
    colours = {-3:'navy', -2:'blue', -1:'dodgerblue', 0:"purple", 1:'violet',
               2:'salmon', 3:'red', 4:'red', 5:'red'} # per radial order
    connect_colour_predicted = "yellowgreen"
    connect_colour_observed = "olive"

    fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
    ax2 = ax1.twinx() # To have a second y-axis for n_pg

    # All signals
    ax1.vlines(observation["all_freqs"], np.zeros(len(observation["all_amps"])),
               np.sqrt(observation["all_amps"]),
               color="grey", alpha=0.8, ls="solid", lw=2, zorder=0)
    # Highlighting the modelled frequencies
    ax1.vlines(observation["freqs"], np.zeros(len(observation["amps"])),
               np.sqrt(observation["amps"]),
               color="black", alpha=0.8, ls="solid", lw=3, zorder=1)
    for i in range(observation["N_m"]) :
        if np.isnan(observation["n_pgs"][i]) :
            c = "black"
        else :
            c = colours[observation["n_pgs"][i]]
        freq_m0 = observation["freqs"][i]
        amp_m0 = observation["amps"][i]
        l = observation["ls"][i]
        ax1.scatter(freq_m0, np.sqrt(amp_m0),
                    marker=markers[observation["ls"][i]], color=c, s=30,
                    zorder=4)
        freqs, amps = [], []
        m = -l
        for split, amp_split in zip(observation["splittings"][i],
                                    observation["amps_splittings"][i]) :
            if m == 0 :
                freqs.append(freq_m0)
                amps.append(amp_m0)
                m += 1
            if np.isnan(split) :
                m += 1
                continue
            if m < 0 :
                freqs.append(freq_m0 - split)
            else :
                freqs.append(freq_m0 + split)
            amps.append(amp_split)
            m += 1
        ax1.scatter(freqs, np.sqrt(amps),
                    marker=markers[observation["ls"][i]], color=c, s=30,
                    zorder=4)
        # Highlighting the observed frequencies
        ax1.vlines(freqs, np.zeros(len(freqs)), np.sqrt(amps),
                   color="black", alpha=0.8, ls="solid", lw=3, zorder=1)
        ax1.plot(freqs, np.sqrt(amps), color=connect_colour_observed, zorder=4)

    # Predictions
    for l in np.unique(model["ls"]) :
        for n_pg in np.unique(model["n_pgs"]) :
            freqs = model["freqs"][(model["ls"] == l) & (model["n_pgs"] == n_pg)]
            for freq in freqs :
                ax1.axvline(freq, linestyle=linestyles[l], color=colours[n_pg],
                            alpha=0.6, lw=1., zorder=2)
                ax2.scatter(freq, n_pg, marker=markers[l], color=colours[n_pg],
                            s=25, zorder=3)
            ax2.plot(freqs, [n_pg]*len(freqs), color=connect_colour_predicted,
                     zorder=4)

    freqs = np.array(observation["all_freqs"])
    freqs = freqs[(freqs > 2) & (freqs < 15)]
    xmin = max(2, np.min(freqs)-0.25)
    xmax = min(15, np.max(freqs)+0.25)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(0, None)

    ax1.set_xlabel("frequency [c/d]", fontsize=10)
    ax1.set_ylabel(r"$\sqrt{\rm{amplitude}}$", fontsize=10)
    ax1.set_title(title, fontsize=10)
    ax1.tick_params(bottom=True, left=True, top=True, right=True, which='both')
    ax1.minorticks_on()

    ax2.set_ylabel("n_pg", fontsize=10)
    ax2.set_ylim(-6, 6)
    ax2.set_yticks([-3,-2,-1,0,1,2,3])
    ax2.grid(axis="y", alpha=0.5, zorder=-1)

    fig.tight_layout()

    if save :
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir, exist_ok=True)
        save_file = f'{observation["name"]}_{save_name}_npg{n_pg_main}_{observation["save_suffix"]}'
        fig.savefig(f'{save_dir}/{save_file}', dpi=200, bbox_inches="tight")
    if show :
        plt.show(block=False)
    else :
        plt.close()

    return fig, ax1, ax2
