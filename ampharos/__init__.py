from .support_functions import initialise_grid, find_age_cross_zonal_mode, \
                               interpolate_track_to_model_in_age, \
                               interpolate_models_in_frot_with_StORM, \
                               is_model_compatible_with_spectroscopy, \
                               number_observations, standardised_chi2, \
                               interpolate_history_columns_in_age, \
                               interpolate_model_in_frot_with_GYRE
from .io_functions import write_results, create_observation_dict

from .visualisation import *

import pandas as pd
import numpy as np
from ast import literal_eval


##### Starting up ################################################################


def initialise_all(observation_file, parameter_file, MESA_output_dir,
                   frots_rel_to_include=[], Xc_min=1e-4, Xc_max=0.7012,
                   add_beta=False,
                   ls_to_include=[0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2],
                   n_pgs_to_include=[1,2,3,4,-2,-1,1,2,3,4,-2,-1,0,1,2,3,4],
                   sort_by_l=True, sort_by_freq=False,
                   o_delimiter="\t", p_header=True, p_delimiter=",") :
    """
    Collect the necessary input for the analysis of a set of stars, namely to
    read and prepare all the observational input and to load in the model grid
    based on the given model parameter file and requested rotation rates.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    observation_file: string
        The path to the file containing all the observational input
    parameter_file: string
        The path to the file containing all sets of model parameters Z, M, logD,
        aov, fov to be read in
    MESA_output_dir: dict
        The path to the top level directory in which the MESA and StORM
        (and GYRE) models are saved
    frots_rel_to_include : list of floats
        Which relative rotation rates to extract from the StORM summaries
        Leave empty to include all frots_rel available in the StORM summaries
        Defaults to []
    Xc_min: float
        Smallest central hydrogen mass fraction to include in the grid. StORM
        summaries for an Xc smaller than this minimum are not loaded in and
        extrapolating beyond this minimum in the age fixing is not permitted
        If None, no filtering based on the lower limit of the central hydrogen
        fraction is done
        Defaults to 0.0001
    Xc_max: float
        Largest central hydrogen mass fraction to include in the grid. StORM
        summaries for an Xc greater than this maximum are not loaded in and
        extrapolating beyond this maximum in the age fixing is not permitted. If
        None, no filtering based on the upper limit of the central hydrogen
        fraction is done
        Defaults to 0.7012
    add_beta:
        Whether to also load in the beta_nl parameter from GYRE. The use_GYRE
        flag in analyse_observations can only be used if beta_nl was loaded in.
        Slightly slows down the loading in of the grid.
        Defaults to False
    ls_to_include: list of int
        The degrees of the modes to be read in. One entry for each combination
        of (l,n_pg). Must be of the same length as n_pgs_to_include
        Defaults to [0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2]
    n_pgs_to_include: list of int
        The radial orders of the modes to be read in. These are defined from the
        Eckart-Scuflaire-Osaki scheme, except for l=1 modes, which use the
        Takata scheme. One entry for each combination of (l,n_pg). Must be of
        the same length as ls_to_include
        Defaults to [1,2,3,4,-2,-1,1,2,3,4,-2,-1,0,1,2,3,4]
    sort_by_l: bool
        Whether to sort the read multiplets by their degree
    sort_by_freq: bool
        Whether to sort the read multiplets by the frequency of their zonal mode
        from largest to smallest
    o_delimiter: string
        The character(s) used to separate the columns in the observation file.
        Defaults to ","
    p_header: bool
        Whether the parameter file has a header describing each column. If not,
        the file is assumed to have only four columns in the order Z, M, logD,
        fov
        Defaults to True
    p_delimiter: string
        The character(s) used to separate the values in the parameter file.
        Defaults to ","

    ------- Returns -------

    grid: dict
        "MESA_output_dir": string
            Path to the top level directory holding the MESA and StORM output
        "Zs": numpy.array of floats
            Initial metallicity of each MESA-StORM run
        "Ms": numpy.array of floats
            Initial mass in solar masses of each MESA-StORM run
        "logDs": numpy.array of floats
            Base 10 logarithm of the mixing strength at the base of the
            radiative envelope of each MESA-StORM run
        "fovs": numpy.array of floats
            Overshooting parameter of each MESA-StORM run
        "frots_rel": numpy.array of floats
            Rotation rate in relative fraction of Keplerian critical rotation
            rate used in each StORM computation or only the subset of these
            included in the frots_file
        "all_tracks": list of dicts
            Dictionaries describing the StORM output of each MESA evolutionary
            track, as detailed in the documentation of read_storm_tracks
    Each of these arrays is off the same length, namely the number of lines in
    the parameter file (minus header). In other words, there is one tracks
    dictionary for each evolution model containing the StORM analysis of all
    output steps and rotation rates.

    all_observation: list of dicts
        For each line in the observation file, a dictionary containing all the
        observational info required for the forward modelling. Contains:
        "name": str
            an identifying name of the star
        "save_suffix": str
            a suffix to be added on the end of all output files and plots of
            this target's analysis. Useful when modelling the same star (i.e.
            the same name) with different input. Can be empty.
        "log_Teff": float
            Base 10 logarithm of the observed effective temperature
        "log_Teff_err": float
            Uncertainty on "log_Teff"
        "log_L": float
            Base 10 logarithm of the observed luminosity
        "log_L_err": float
            Uncertainty on "log_L"
        "N_f": int
            Number of identified frequencies provided
        "N_m": int
            Number of identified multiplets described
        "freqs": list of floats
            Observed zonal frequencies of the identified modes in cycles per day
            Length of this list equals N_m
        "freqs_err": list of floats
            Uncertainty on the observed frequencies of the identified modes in
            cycles per day
            Length of this list equals N_m
        "splittings": list of list of floats
            The observed rotational splitting of the identified multiplets in
            cycles per day. np.nan if no splitting is given, i.e. if the
            multiplet is incomplete.
            Length of the first list equals N_m, length of the second depends on
            the degree as 2*l
        "splittings_err": list of list of floats
            Uncertainty on the observed rotational splitting in cycles per day
            np.nan if no splitting is given, i.e. if the multiplet is
            incomplete.
            Length of the first list equals N_m, length of the second depends on
            the degree as 2*l
        "amps": list of floats
            Amplitudes of the identified zonal modes. Only used in visualisation
            Length of this list equals N_m
        "ls": list of ints
            Identified degrees of the identified zonal modes.
            Length of this list equals N_m
        "n_pgs": list of ints
            If available, the radial orders of the identified modes. np.nan if
            the radial order is not known a priori
            Length of this list equals N_m
        "all_freqs": list of floats
            Frequencies in cycles per day of all detected signals, not only the
            identified signals.
        "all_amps": list of floats
            Amplitudes of all detected signals, not only the identified signals.
        "n_pg_main": int or np.nan
            Radial order of the main, fixed mode if known. np.nan if unknown

    """
    all_observations = read_observations(observation_file,
                                         delimiter=o_delimiter,
                                         sort_by_l=sort_by_l,
                                         sort_by_freq=sort_by_freq)
    grid = initialise_grid(MESA_output_dir, parameter_file,
                           frots_rel_to_include=np.array(frots_rel_to_include, dtype=np.float32),
                           ls_to_include=ls_to_include,
                           n_pgs_to_include=n_pgs_to_include,
                           Xc_min=Xc_min, Xc_max=Xc_max, add_beta=add_beta,
                           header=p_header, delimiter=p_delimiter)
    return grid, all_observations


def read_observations(observation_file, delimiter="\t", sort_by_l=True,
                      sort_by_freq=False) :
    """
    Read in the (.tsv) file containing all relevant observations and summarise
    each set of observations in a dictionary.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    observation_file: string
        Path to the tab-separated value file containing all the necessary
        observational information for the modelling
    delimiter: string
        The character(s) used to separate the values in the file.
        Defaults to "\t", i.e. tab-separated values
    sort_by_l: bool
        Whether to sort the read multiplets by their degree
        Defaults to True
    sort_by_freq: bool
        Whether to sort the read multiplets by the frequency of their zonal mode
        from largest to smallest
        Defaults to False

    ------- Returns -------
    all_observations: list of dicts
        list containing an observation dictionary as defined in
        create_observation_dict for each line in the input file,
        except for the header
    """
    df = pd.read_csv(observation_file, sep=delimiter, header=0,
                     converters={"all_f":literal_eval, "all_a":literal_eval})
    all_observations = []
    for index, row in df.iterrows() :
        all_observations.append(create_observation_dict(row,
                                                        sort_by_l=sort_by_l,
                                                        sort_by_freq=sort_by_freq))
    return all_observations


##### Finding the optimal model for each evolutionary track ####################


def find_age_cross_and_frot(grid, observation, results, n_pg_main,
                            frots_rel_to_include=[], use_GYRE=False) :
    """
    For each evolutionary track in the model grid, find the optimal age and
    rotation frequency and add them into the results. Also check if the optimal
    model lies within the HRD's 2sigma error ellipse.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    results: dicts
        The dictionary describing the outcome of comparing a given observation
        to the grid
    n_pg_main: int
        Radial order of the main, fixed mode frequency.
    frots_rel_to_include : list of floats
        Which relative rotation rates for which to find the optimal age and then
        use in the interpolation in rotation. Leave empty to include all
        frots_rel available in the loaded in StORM summaries
        Defaults to []
    use_GYRE: bool
        Whether to optimise the rotation rate using the mean rotational
        splitting and the first-order rotational treatment by GYRE, rather than
        the second-order treatment optimising all the individual splitting using
        the StORM grid.

    """
    results["ages_cross"], results["Xcs_cross"] = [], []
    results["frots_best"], results["frots_rel_best"] = [], []
    results["models"] = []
    results["spectro_compatible"], results["in_statistical_model"]  = [], []
    results["n_pg_main"] = n_pg_main
    for tracks in grid["all_tracks"] :
        if len(frots_rel_to_include) == 0 :
            frots_rel = tracks["frots_rel"]
        else :
            frots_rel = frots_rel_to_include
        if not use_GYRE : # When using StORM as recommended
            models = []
            for frot_rel in frots_rel :
                age_cross = find_age_cross_zonal_mode(tracks, n_pg_main,
                                                      observation["freqs"][0],
                                                      observation["ls"][0],
                                                      frot_rel=frot_rel)
                models.append(interpolate_track_to_model_in_age(tracks,
                                                                age_cross,
                                                                frot_rel=frot_rel,
                                                                add_beta=False))
            model_best_frot = interpolate_models_in_frot_with_StORM(observation,
                                                                       models)
        else :
            age_cross = find_age_cross_zonal_mode(tracks, n_pg_main,
                                                  observation["freqs"][0],
                                                  observation["ls"][0],
                                                  frot_rel=0.0)
            model_nonRot = interpolate_track_to_model_in_age(tracks, age_cross,
                                                             frot_rel=0.0,
                                                             add_beta=True)
            model_best_frot = interpolate_model_in_frot_with_GYRE(observation,
                                                                  model_nonRot)
        results["ages_cross"].append(model_best_frot["age"])
        results["Xcs_cross"].append(model_best_frot["Xc"])
        results["frots_best"].append(model_best_frot["frot_best"])
        results["frots_rel_best"].append(model_best_frot["frot_rel_best"])
        results["models"].append(model_best_frot)
        results["spectro_compatible"].append(is_model_compatible_with_spectroscopy(model_best_frot, observation, sigma_tol=2.))
        results["in_statistical_model"].append(is_model_compatible_with_spectroscopy(model_best_frot, observation, sigma_tol=2.))
    # Turn the lists into numpy.arrays to simplify later analysis and plotting
    for key in ("ages_cross", "Xcs_cross", "frots_best", "spectro_compatible",
                "in_statistical_model") :
        results[key] = np.array(results[key])


def compute_chi2s(observation, results, exclude_splitting_from_chi2=False) :
    """
    Compare the optimal model of each evolutionary track against the observation
    and compute the chi-squared and reduced chi-squared cost functions

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    results: dicts
        The dictionary describing the outcome of comparing a given observation
        to the grid
    exclude_splitting_from_chi2: bool
        Whether to not include rotational splitting terms in the chi2 cost
        function
        Defaults to False

    """
    # Number of degrees of freedom when including all observations
    DoF_withSplitting = max((number_observations(observation, exclude_splitting=False) - 5),1)
    chi2s, chi2s_reduced = [], []
    if exclude_splitting_from_chi2 :
        # Number of degrees of freedom when not counting the rotational
        # splittings
        DoF_withoutSplitting = max((number_observations(observation, exclude_splitting=True) - 4),1)
        chi2s_withSplitting, chi2s_withSplitting_reduced = [], []
        for model in results["models"] :
            chi2 = standardised_chi2(model, observation, exclude_splitting=True)
            chi2s.append(chi2)
            chi2s_reduced.append(chi2 / DoF_withoutSplitting)
            chi2 = standardised_chi2(model, observation,
                                     exclude_splitting=False)
            chi2s_withSplitting.append(chi2)
            chi2s_withSplitting_reduced.append(chi2 / DoF_withSplitting)
        results["chi2s_withSplitting"] = np.array(chi2s_withSplitting)
        results["chi2s_withSplitting_reduced"] = np.array(chi2s_withSplitting_reduced)
    else :
        for model in results["models"] :
            chi2 = standardised_chi2(model, observation,
                                     exclude_splitting=False)
            chi2s.append(chi2)
            chi2s_reduced.append(chi2 / DoF_withSplitting)
    results["chi2s"] = np.array(chi2s)
    results["chi2s_reduced"] = np.array(chi2s_reduced)


def find_best_chi2_all(observation, results, exclude_splitting_from_chi2=False) :
    """
    Find the best fitting model within the 2 sigma error ellipse in the HRD and
    add it to the results. If there is no model within that error ellipse,
    report the model with the best chi2. If no model was found for any track,
    meaning none of the frequency-age tracks crosses the observed mode frequency
    for an assumed radial order.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    results: dicts
        The dictionary describing the outcome of comparing a given observation
        to the grid as created in the function trying_all_n_pg_main.
    exclude_splitting_from_chi2: bool
        Whether to not include rotational splitting terms in the chi2 cost
        function
        Defaults to False

    """
    compute_chi2s(observation, results,
                  exclude_splitting_from_chi2=exclude_splitting_from_chi2)
    if not np.sum(results["spectro_compatible"]) == 0 :
        # At least one result is compatible with observed L and Teff within
        # 2sigma
        results["chi2_best"] = np.nanmin(results["chi2s"][results["spectro_compatible"]])
        results["best_model_index"] = np.where(results["spectro_compatible"])[0][np.nanargmin(results["chi2s"][results["spectro_compatible"]])]
        results["best_model"] = results["models"][results["best_model_index"]]
        assert results["chi2s"][results["best_model_index"]] == results["chi2_best"]
    elif not np.isnan(results["chi2s"]).all() :
        results["best_model_index"] = np.nanargmin(results["chi2s"])
        results["best_model"] = results["models"][results["best_model_index"]]
        results["chi2_best"] = results["chi2s"][results["best_model_index"]]
        assert results["chi2s"][results["best_model_index"]] == results["chi2_best"]
    else :
        results["best_model_index"] = np.nan
        results["best_model"] = np.nan
        results["chi2_best"] = np.nan
        print("All chi2 are NaN! No useable model was found!")


##### Figuring out n_pg ########################################################


def analyse_observations(grid, all_observations, n_pgs=[1,2,3],
                         Nmodels_to_show=5, use_GYRE=False,
                         exclude_splitting_from_chi2=False) :
    """
    Compare the set of observations against the grid assuming the different
    radial orders for the fixed mode, using the selected methodology, creating
    plots output for each considered radial order and then creating output to
    help the user identify the best radial order for the fixed mode.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    all_observations: list of dicts
        The dictionaries describing all the observations of a star as defined in
        create_observation_dict
    n_pgs: list of ints
        Radial orders to try out for the fixed, main mode
    Nmodels_to_show: int
        How many models' frequencies to compare to the observations to help the
        user find the best fitting radial order
        Defaults to 5
    use_GYRE: bool
        Whether to optimise the rotation rate using the mean rotational
        splitting and the first-order rotational treatment by GYRE, rather than
        the second-order treatment optimising all the individual splitting using
        the StORM grid.
        Defaults to False
    exclude_splitting_from_chi2: bool
        Whether to not include rotational splitting terms in the chi2 cost
        function
        Defaults to False

    """
    for observation in all_observations :
        # If not instructed to test specific radial orders for the fixed, main
        # mode, only include the n_pg given in the observational input. If n_pg
        # is not given in the observational input just skip that star. If given
        # a specific list of radial orders, just try all of them
        if len(n_pgs) == 0 :
            if np.isnan(observation["n_pg_main"]) :
                continue
            n_pgs_main = [int(observation["n_pg_main"])]
        else :
            n_pgs_main = n_pgs
        print(observation["name"])
        # Setting the paths for where to create the output.
        plots_dir = f"plots/{observation['name']}"
        results_dir = f"results/{observation['name']}"
        if exclude_splitting_from_chi2 :
            plots_dir += "_chi2SplittingExcluded"
            results_dir += "_chi2SplittingExcluded"
        if use_GYRE :
            plots_dir += "_GYRE/"
            results_dir += "_GYRE/"

        # Trying all n_pg_main options and collecting results
        all_results = trying_all_n_pg_main(grid, observation, n_pgs=n_pgs_main,
                                           use_GYRE=use_GYRE,
                                           exclude_splitting_from_chi2=exclude_splitting_from_chi2)

        # Write output files
        for i,results in enumerate(all_results) :
            add_history_results(grid, results)
            statistical_model = make_statistical_model(grid, results)
            results["statistical_model"] = statistical_model
            write_results(grid, observation, results, statistical_model,
                          f'{results_dir}/n_pg_main{n_pgs_main[i]}/')

        # Creating plots
        make_all_the_plots(grid, observation, all_results, n_pgs=n_pgs_main,
                           save_dir=plots_dir, show=False)

        make_plots_to_pick_npg(observation, all_results,
                               f"{plots_dir}/picking_npg/", n_pgs=n_pgs_main,
                               Nmodels_to_show=Nmodels_to_show, show=False)


def trying_all_n_pg_main(grid, observation, n_pgs=[1,2,3], use_GYRE=False,
                         exclude_splitting_from_chi2=False) :
    """
    Comparing the observations of a given star to the grid of models to find
    optimal model for each evolutionary track, compute the cost functions for
    each and then finding the one best fitting model.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    n_pgs: list of ints
        Radial orders to try out for the fixed, main mode
        Defaults to [1,2,3]
    use_GYRE: bool
        Whether to optimise the rotation rate using the mean rotational
        splitting and the first-order rotational treatment by GYRE, rather than
        the second-order treatment optimising all the individual splitting using
        the StORM grid.
        Defaults to False
    exclude_splitting_from_chi2: bool
        Whether to not include rotational splitting terms in the chi2 cost
        function
        Defaults to False

    ------- Returns -------

    all_results: list of dicts
        A set modelling results for each radial order in n_pgs. Each of these
        sets contains:
            "name": str
                an identifying name of the star
            "save_suffix": str
            a suffix to be added on the end of all output files and plots of
            this target's analysis. Useful when modelling the same star (i.e.
            the same name) with different input. Can be empty.
            "n_pg_main": int
                The radial order of the fixed mode
            "ages_cross": numpy.array of floats
                The ages in years at which an evolutionary track matches the
                fixed mode frequency
                Length of this list equals N
            "Xcs_cross": numpy.array of floats
                The central hydrogen mass fractions in years at which an
                evolutionary track matches the fixed mode frequency
                Length of this list equals N
            "frots_best": numpy.array of floats
                The optimal rotation frequencies in cycles per day best
                reproducing the observed rotational splitting
                Length of this list equals N
            "frots_rel_best": numpy.array of floats
                The optimal rotation frequencies best reproducing the observed
                rotational splitting relative to the Keplerian critical rotation
                frequency
                Length of this list equals N
            "models": list of dicts
                The optimal model for each evolutionary track as created by the
                functions interpolate_track_to_model_in_age and
                interpolate_models_in_frot
                Length of this list equals N
            "spectro_compatible": numpy.array of bools
                Whether the interpolated optimal model at this index is within
                observed 2sigma error ellipse in the HRD
                Length of this list equals N
            "in_statistical_model": numpy.array of bools
                Whether the interpolated optimal model at this index should be
                averaged over when performing the statistical parameter
                estimation
                Length of this list equals N
            "chi2s": numpy.array of floats
                The chi-square cost function of the model at this index
                Length of this list equals N
            "chi2s_reduced": numpy.array of floats
                The reduced chi-square cost function of the model at this index
                Length of this list equals N
            "chi2_best": float
                The chi-square of the best model, selected as the model within
                the 2sigma error ellipse in the HRD with the smallest
                chi-square.
            "best_model_index": int
                The index of the best model, selected as the model within the
                2sigma error ellipse in the HRD with the smallest chi-square.
                Lies in the range of [0, N[
            "best_model": dict
                The best model as created by the functions
                interpolate_track_to_model_in_age and
                interpolate_models_in_frot.
        with N the number of evolutionary track in the grid
    """
    n_pgs = np.array(n_pgs)
    all_results = []
    for n_pg in n_pgs :
        results = {"name":observation["name"],
                   "save_suffix":observation["save_suffix"]}
        find_age_cross_and_frot(grid, observation, results, n_pg,
                                use_GYRE=use_GYRE)
        find_best_chi2_all(observation, results,
                           exclude_splitting_from_chi2=exclude_splitting_from_chi2)
        results["n_pg_main"] = n_pg
        if ("best_model" in list(results.keys())) and \
            not (type(results["best_model"]) == float) :
            results["best_model"]["n_pg_main"] = n_pg
        all_results.append(results)
    return all_results


##### Creating output and plots ################################################


def add_history_results(grid, results) :
    """
    For each evolutionary track's optimised model, find a number of MESA history
    parameters at the optimal age and add them to the results dictionary.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: dict
        The dictionary summarising all the modelling results as created by
        trying_all_n_pg_main
        This function will add the following fields to the results dictionary:
            "log_Teffs": numpy.array of floats
                The base-10 logarithm of the effective temperature in kelvin of
                each optimal model
                Length of this list equals N
            "log_Ls": numpy.array of floats
                The base-10 logarithm of the luminosity in solar luminosities of
                each optimal model
                Length of this list equals N
            "Ms_cc": numpy.array of floats
                The convective core mass in solar masses of each optimal model
                Length of this list equals N
            "log_Rs": numpy.array of floats
                The base-10 logarithm of the stellar radius in solar radii of
                each optimal model
                Length of this list equals N
            "log_gs": numpy.array of floats
                The base-10 logarithm of the surface gravity in cgs (cm/s^2) of
                each optimal model
                Length of this list equals N
            "Pi0s": numpy.array of floats
                The asteroseismic Pi_0 in seconds of each optimal model
                Length of this list equals N
        with N the number of evolutionary track in the grid

    """
    results["log_Teffs"], results["log_Ls"] = [], []
    results["Ms_cc"], results["log_Rs"] = [], []
    results["log_gs"], results["Pi0s"] = [], []

    for tracks, age_cross, model in zip(grid["all_tracks"],
                                        results["ages_cross"],
                                        results["models"]) :
        # Bring these up to the results level for easier access
        results["log_Teffs"].append(model["log_Teff"])
        results["log_Ls"].append(model["log_L"])

        interp_cols = interpolate_history_columns_in_age(tracks, age_cross,
                                                         ["mass_conv_core", "log_R", "log_g", "Asymptotic_dP"])
        M_cc, log_R, log_g, Pi0 = interp_cols
        results["Ms_cc"].append(M_cc)
        results["log_Rs"].append(log_R)
        results["log_gs"].append(log_g)
        results["Pi0s"].append(Pi0)
    for key in ("log_Teffs", "log_Ls", "Ms_cc", "log_Rs", "log_gs", "Pi0s") :
        results[key] = np.array(results[key])


def make_all_the_plots(grid, observation, all_results, save_dir="plots",
                       n_pgs=[1,2,3], show=False) :
    """
    Creates a collection of plots of the forward modelling results and best
    model. To further understand what each figure shows, we refer to the
    relevant functions under visualisation. Be aware that these figures are
    meant to help the user understand the modelling results and are therefore
    not necessarily publication-quality plots. Users are encouraged to alter
    which plots are created and how to suit their own needs.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    all_results: list of dicts
        The dictionaries summarising the forward modelling results assuming
        different radial orders for the fixed mode created by the functions
        trying_all_n_pg_main, add_history_results, and make_statistical_model
    save_dir: string
        Path to the top level directory containing subdirectories named after
        the provided names of the stars where the figures are saved.
        Defaults to "plots"
    n_pgs: list of ints
        Radial orders to try out for the fixed, main mode
        Defaults to [1,2,3]
    show: bool
        Whether to not open the plots in interactive windows or directly save
        them as .png files
        Defaults to False

    """
    labels, best_models = [], []

    for i,n_pg_main in enumerate(n_pgs) :
        results = all_results[i]
        labels.append(f"n_pg_main = {n_pg_main}")
        best_models.append(results["best_model"])

        if np.isnan(results["chi2s"]).all() :
            continue

        plot_all_chi2_in_HRD(results, observation, n_pg_main,
                             title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                             show_best_model=True,
                             show_statistical_model=True, show_IS=True,
                             log=True, hollow_spectro_incompatible=True,
                             save=True, save_dir=save_dir,
                             save_name=f"chi2_in_HRD_npg{n_pg_main}",
                             show=show)

        plot_chi2_against_grid_params(grid, results, n_pg_main,
                                      show_best_model=True,
                                      show_statistical_model=True, log=True,
                                      hollow_spectro_incompatible=True,
                                      title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                      save=True, save_dir=save_dir,
                                      save_name=f"chi2_grid_params_npg{n_pg_main}",
                                      show=show)

        plot_chi2_against_model_params(results, show_best_model=True,
                                       show_statistical_model=True,
                                       chi2_log=True, age_log=True,
                                       hollow_spectro_incompatible=True,
                                       title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                       save=True, save_dir=save_dir,
                                       save_name=f"chi2_model_params_npg{n_pg_main}",
                                       show=show)

        plot_chi2_coloured_grid_params(grid, results, show_best_model=True,
                                       show_statistical_model=True,
                                       log=True, only_plot_compatible=False,
                                       title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                       colourmap="plasma_r", cmin=None,
                                       cmax=None, save=True, save_dir=save_dir,
                                       save_name=f"chi2_coloured_grid_params_npg{n_pg_main}",
                                       show=show)

        plot_grid_params_chi2_corner(grid, results, show_best_model=True,
                                     show_statistical_model=True, log=True,
                                     hollow_spectro_incompatible=True,
                                     title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                     colourmap="plasma_r", cmin=None,
                                     cmax=None, save=True,  save_dir=save_dir,
                                     save_name=f"corner_grid_params_npg{n_pg_main}",
                                     show=show)

        plot_model_params_histograms(results, age_log=True,
                                     hollow_spectro_incompatible=True,
                                     title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                     save=True, save_dir=save_dir,
                                     save_name=f"model_params_histograms_npg{n_pg_main}",
                                     show=show)

        plot_Xc_grid_params(grid, results, show_best_model=True,
                            show_statistical_model=True,
                            hollow_spectro_incompatible=True,
                            title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                            save=True, save_dir=save_dir,
                            save_name=f"Xc_model_params_npg{n_pg_main}",
                            show=show)

        plot_model_and_observed_frequencies(results["best_model"],
                                            observation, n_pg_main,
                                            title=f'{results["name"]} - n_pg_main = {n_pg_main}',
                                            save=True, save_dir=save_dir,
                                            save_name=f"comparing_frequencies_npg{n_pg_main}",
                                            show=show)

    plot_models_and_observed_HRD(best_models, observation,
                                 title=f'{results["name"]}', labels=labels,
                                 save=True, save_dir=save_dir,
                                 save_name=f"comparing_HRD_best_models",
                                 show=show, show_IS=True)


def make_plots_to_pick_npg(observation, all_results, save_dir, n_pgs=[1,2,3],
                           Nmodels_to_show=5, show=False) :
    """
    Creates a set of plots of the designed to assist the user in figuring out
    the most likely radial order of the fixed, main mode. There are two types of
    plots. Firstly, for each considered radial order, the position of the
    `Nmodels_to_show best` models and their evolutionary tracks are plotted in
    the HRD alongside the observations. Secondly, for each considered radial
    order, `Nmodels_to_show` plots are created comparing the best models'
    predicted frequencies to the observed signals, including the unidentified
    frequencies. Be aware that these figures are meant to help the user identify
    the most likely radial order for their chosen fixed mode and are therefore
    not publication-quality plots.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    all_results: list of dicts
        The dictionaries summarising the forward modelling results assuming
        different radial orders for the fixed mode created by the functions
        trying_all_n_pg_main, add_history_results, and make_statistical_model
    save_dir: string
        Path to the directory where these figures are saved.
    n_pgs: list of ints
        Radial orders to try out for the fixed, main mode
        Defaults to [1,2,3]
    Nmodels_to_show: int
        How many optimised models' frequencies to compare to the observations
        for each considered radial order for the fixed, main mode
        Defaults to 5
    show: bool
        Whether to not open the plots in interactive windows or directly save
        them as pngs
        Defaults to False

    """
    for n_pg_main, results in zip(n_pgs, all_results) :
        models_to_show, labels = [], []

        models_spectro_compatible, chi2s_spectro_compatible = [], []
        I_toInclude = np.where(results["spectro_compatible"])[0]
        for i in I_toInclude :
            chi2 = results["chi2s"][i]
            if not np.isnan(chi2) :
                chi2s_spectro_compatible.append(chi2)
                models_spectro_compatible.append(results["models"][i])
        chi2s_spectro_compatible = np.array(chi2s_spectro_compatible)

        if np.isnan(chi2s_spectro_compatible).all() :
            print("make_plots_to_pick_npg : No models compatible with log_Teff and log_L!")
            continue

        K_sort = np.argsort(chi2s_spectro_compatible)[:min(Nmodels_to_show,len(chi2s_spectro_compatible))]

        for j,k in enumerate(K_sort) :

            model = models_spectro_compatible[k]
            label = chi2s_spectro_compatible[k]
            models_to_show.append(model)
            labels.append(f"chi2 = {label}")

            plot_model_and_observed_frequencies(model, observation, n_pg_main,
                                                title=f'{results["name"]} - n_pg_main = {n_pg_main} - j = {j} - chi2 = {label}',
                                                save=True, save_dir=save_dir,
                                                save_name=f"comparing_frequencies_npg{n_pg_main}_k{j}",
                                                show=show)

        plot_models_and_observed_HRD(models_to_show, observation,
                                     title=f'{all_results[0]["name"]} - n_pg_main = {n_pg_main}',
                                     labels=labels, save=True,
                                     save_dir=save_dir,
                                     save_name=f"comparing_HRD_npg{n_pg_main}",
                                     show=show, show_IS=True)


def make_statistical_model(grid, results) :
    """
    Compute the statistical parameter estimation as a exp(-chi^2/2) weighted
    mean and standard deviation

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: dict
        The dictionary summarising the forward modelling results assuming a
        certain radial order for the fixed mode created by the functions
        trying_all_n_pg_main and add_history_results

    --------- Returns ---------

    statistical_model: dict containing
        "M_mean": float
            Statistical estimate for initial mass in solar masses
        "M_std": float
            Uncertainty on statistical initial mass estimate in solar masses
        "logD_mean": float
            Statistical estimate for base-10 logarithm of the mixing strength at
            the base of the envelope in cgs
        "logD_std": float
            Uncertainty on statistical base-10 logarithm of the mixing strength
            at the base of the envelope estimate in cgs
        "fov_mean": float
            Statistical estimate for core overshoot parameter in pressure scale
            heights
        "fov_std": float
            Uncertainty on statistical core overshoot estimate in pressure scale
            heights
        "age_mean": float
            Statistical estimate for age in years
        "age_std": float
            Uncertainty on statistical age estimate in years
        "log_age_mean": float
            Uncertainty on statistical base-10 logarithm of age estimate in
            years
        "log_age_std": float
            Uncertainty on statistical initial mass estimate in solar masses
        "Xc_mean": float
            Statistical estimate for central hydrogen mass fraction
        "Xc_std": float
            Uncertainty on statistical central hydrogen mass fraction estimate
        "frot_mean": float
            Statistical estimate for rotation frequency in cycles per day
        "frot_std": float
            Uncertainty on statistical rotation frequency estimate in
            cycles per day
        "frot_rel_mean": float
            Statistical estimate for rotation frequency relative to the
            Keplerian critical rotation frequency
        "frot_rel_std": float
            Uncertainty on statistical rotation frequency relative to the
            Keplerian critical rotation frequency estimate
        "fK_mean": float
            Statistical estimate for Keplerian critical rotation frequency in
            cycles per day
        "fK_std": float
            Uncertainty on statistical Keplerian critical rotation frequency in
            cycles per day
        "log_Teff_mean": float
            Statistical estimate for base-10 logarithm of effective temperature
            in kelvin
        "log_Teff_std": float
            Uncertainty on statistical base-10 logarithm of effective
            temperature estimate in kelvin
        "log_L_mean": float
            Statistical estimate for base-10 logarithm of luminosity in
            solar luminosity
        "log_L_std": float
            Uncertainty on statistical base-10 logarithm of luminosity estimate
            in solar luminosity
        "R_mean": float
            Statistical estimate for stellar radius in solar radii
        "R_std": float
            Uncertainty on statistical stellar radius estimate in solar radii
        "log_R_mean": float
            Statistical estimate for base-10 logarithm of stellar radius in
            solar radii
        "log_R_std": float
            Uncertainty on statistical base-10 logarithm of stellar radius
            estimate in solar radii
        "log_g_mean": float
            Statistical estimate for base-10 logarithm of surface gravity in cgs
        "log_g_std": float
            Uncertainty on statistical base-10 logarithm of surface gravity
            estimate in cgs
        "Mcc_mean": float
            Statistical estimate for convective core mass in solar masses
        "Mcc_std": float
            Uncertainty on statistical convective core mass estimate in
            solar masses
        "Mcc_over_M_mean": float
            Statistical estimate for convective core mass relative to i
            nitial mass
        "Mcc_over_M_std": float
            Uncertainty on statistical convective core mass relative to
            initial mass
        "Pi0_mean": float
            Statistical estimate for the asteroseismic Pi_0 quantity in seconds
        "Pi0_std": float
            Uncertainty on statistical the asteroseismic Pi_0 quantity estimate
            in seconds
    """
    assert "in_statistical_model" in results.keys()

    # Find the useable models
    toInclude = (results["in_statistical_model"]) & \
                (~np.isnan(results["chi2s"]))

    # Grabbing all necessary data of the models within our considered range
    # From grid: M, logD, fov (I wanted to include Z, but that has no steps at
    # the moment and thus breaks the uncertainty estimate)
    # From results: age, Xc, frot_best, Mcc, Mcc_over_M, Pi0
    # Collect these quantities and their names in two lists to automate some
    # computations below
    columns = np.array([grid["Ms"], grid["logDs"], grid["fovs"],
                        results["ages_cross"], np.log10(results["ages_cross"]),
                        results["Xcs_cross"], results["frots_best"],
                        results["frots_rel_best"],
                        results["frots_best"]/results["frots_rel_best"],
                        results["log_Teffs"], results["log_Ls"],
                        10**results["log_Rs"], results["log_Rs"],
                        results["log_gs"], results["Ms_cc"],
                        results["Ms_cc"]/grid["Ms"], results["Pi0s"]])
    columns = np.array(columns)[:,toInclude]
    names = ["M", "logD", "fov", "age", "log_age", "Xc", "frot", "frot_rel",
             "fK", "log_Teff", "log_L", "R", "log_R", "log_g", "Mcc",
             "Mcc_over_M", "Pi0"]

    assert len(columns) == len(names)

    # Set up dictionary
    statistical_model = {"name":results["name"],
                         "save_suffix":results["save_suffix"], "warning":""}

    if np.sum(toInclude) == 0 :
        print("No models to use in the statistical model provided!")
        # If there are no acceptable models, all values and uncertainties are
        # NaN
        for name in names :
            statistical_model[f"{name}_main"] = np.nan
            statistical_model[f"{name}_std"] = np.nan
        statistical_model["warning"] = "No models provided! "
        return statistical_model
    if np.sum(toInclude) == 1 :
        # If there is exactly one acceptable model, all uncertainties are NaN
        print("Only one model to use in the statistical model!")
        for column, name in zip(columns, names) :
            statistical_model[f"{name}_mean"] = column[0]
            statistical_model[f"{name}_std"] = np.nan
        statistical_model["warning"] += "Only one model provided! "
        return statistical_model

    # In the "normal" case with multiplet of course I need the chi squares to
    # compute the weights
    chi2s = results["chi2s_reduced"][toInclude]
    chi2_min = np.min(chi2s)
    weights = np.exp(-(chi2s-chi2_min)/2) # weights
    total_weight = np.sum(weights)
    n = len(weights)

    # Compute the means as our best values
    for column, name in zip(columns, names) :
        statistical_model[f"{name}_mean"] = np.sum(column*weights)/total_weight
        statistical_model[f"{name}_std"] = np.sqrt(n/(n-1) * np.sum(weights * (column - statistical_model[f"{name}_mean"])**2) / total_weight)

    return statistical_model
