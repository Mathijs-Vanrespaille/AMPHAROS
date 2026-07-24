import numpy as np
import scipy.interpolate as scint
from concurrent.futures import ProcessPoolExecutor

from .io_functions import read_model_parameters, read_storm_tracks, \
                          read_MESA_history_columns


##### MISCELLANEOUS ############################################################


def half_max_x(x, y) :
    """
    Linearly interpolate to the points where the curve crosses the full width
    half maximum of the distribution y.

    Author: ColonelFazackerley. Taken from
    https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python

    ------- Parameters -------

    x: list of float
        Parameter values
    y: list of float
        Kernel density estimate distribution

    ------- Returns -------

    crossings: list of two floats
        Interpolated values of x where the distribution crosses the half-maximum
        level left and right of the mode
    """

    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def initialise_grid(MESA_output_dir, parameter_file, frots_rel_to_include=[],
                    ls_to_include=[0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2],
                    n_pgs_to_include=[1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3],
                    Xc_min=None, Xc_max=None,
                    header=True, delimiter=",", add_beta=False) :
    """
    Create and return a dictionary holding all the required MESA and StORM
    output and some supporting fields to help navigate them. See the 'Returns'
    section below for details.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    MESA_output_dir: string
        Path to the top level directory holding the MESA and GYRE output
    parameter_file: string
        Path to the file containing all combinations of the varied model
        parameters Zini, Mini, logD, fov; i.e. the different parameters fed to 
        MESA, without affecting the StORM analysis.
    frots_rel_to_include: list of floats
        The rotation rates relative to the Keplerian critical rotation rate to
        be included. If left empty, all the rotation rates provided in the StORM
        summaries are used.
        Defaults to []
    Xc_min: float
        The smallest central hydrogen fraction to consider. MESA history output
        and StORM summaries at a central hydrogen fraction smaller than this
        limit are not read in. If None, no filtering based on the lower limit of
        the central hydrogen fraction is done
        Defaults to None
    Xc_max: float
        The largest central hydrogen fraction to consider. MESA history output
        and StORM summaries at a central hydrogen fraction greater than this
        limit are not read in. If None, no filtering based on the upper limit of
        the central hydrogen fraction is done
        Defaults to None
    header: bool
        Whether the parameter file has a header describing each column. If not,
        the file is assumed to have only four columns in the order Zini, Mini, 
        logD, fov
        Defaults to True
    delimiter: string
        The character(s) used to separate the values in the parameter file.
        Defaults to ","
    add_beta: bool
        Whether to include the beta_nl constant from GYRE for each multiplet in
        the model. Setting this to True increases loading time
        Defaults to False

    ------- Returns -------
    grid: dict containing
        "MESA_output_dir": string
            Path to the top level directory holding the MESA and StORM output
        "Zinis": numpy.array of floats
            Initial metallicity of each MESA-StORM run
        "Minis": numpy.array of floats
            Initial mass in solar masses of each MESA-StORM run
        "logDs": numpy.array of floats
            Base 10 logarithm of the mixing strength at the base of the
            radiative envelope of each MESA-StORM run
        "fovs": numpy.array of floats
            Overshooting parameter of each MESA-StORM run
        "frots_rel": numpy.array of floats
            Rotation rate in relative fraction of Keplerian critical rotation
            rate used in each StORM computation
            or only the subset of these included in the frots_file
        "all_tracks": list of dicts
            Dictionaries describing the StORM output of each MESA evolutionary
            track, as detailed in the documentation of read_storm_tracks
    Each of these arrays is off the same length, namely the number of lines in
    the parameter file (minus header). In other words, there is one tracks
    dictionary for each evolution model containing the StORM analysis of all
    output steps and rotation rates.

    """
    assert len(ls_to_include) == len(n_pgs_to_include)
    Zinis, Minis, logDs, fovs = read_model_parameters(parameter_file, 
                                                      header=header,
                                                      delimiter=delimiter)
    grid = {"Zinis":Zinis, "Minis":Minis, "logDs":logDs, "fovs":fovs,
            "frots_rel":frots_rel_to_include, "all_tracks":[],
            "MESA_output_dir":MESA_output_dir}
    N = len(Zinis)
    
    args = [
        (
            MESA_output_dir,
            Zini,
            Mini,
            logD,
            fov,
            frots_rel_to_include,
            Xc_min,
            Xc_max,
            ls_to_include,
            n_pgs_to_include,
            add_beta
        )
        for Zini, Mini, logD, fov
        in zip(Zinis, Minis, logDs, fovs)
    ]
    with ProcessPoolExecutor() as executor:
        for i, track in enumerate(
            executor.map(read_storm_tracks_wrapper, args)
        ):
            grid["all_tracks"].append(track)
            print(f"Finished reading {i+1}/{N}")
    if len(frots_rel_to_include) == 0 :
        # Note that this assumes all the StORM summaries contain the same
        # relative rotation rates
        grid["frots_rel"] = grid["all_tracks"][0]["frots_rel"]
    return grid


# Only here because ProcessPoolExecutor wants a function with only one argument
def read_storm_tracks_wrapper(args):
    return read_storm_tracks(*args)


def lin_interp(x, y, i, half) :
    """
    Linearly interpolate to the point just beyond x[i] where the curve crosses
    the FWHM of the distribution y.

    Author: ColonelFazackerley. Taken from
    https://stackoverflow.com/questions/49100778/fwhm-calculation-using-python

    ------- Parameters -------

    x: list of float
        Parameter values
    y: list of float
        Kernel density estimate distribution
    i: int
        Index of the point before the FWHM crosses the distribution
    half: float
        Half of the FWHM of the distribution y

    ------- Returns -------

    crossing: float
        Interpolated value of x where the distribution crosses the half-maximum
        level
    """
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


##### EXTRACTING DATA FROM DICTIONARIES ########################################


def get_frequency_track(tracks, l, m, n_pg, frot_rel=0.00) :
    """
    Extract the frequencies of a mode with the given ID at the given relative
    rotation rate from the track dictionary.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the STORM output as created by
        read_storm_tracks
    l: int
        Degree of the mode
    m: int
        Azimuthal order of the mode
    n_pg: int
        Radial order of the mode
    frot_rel: float
        Rotation rate relative to critical. Must be in tracks["frot_rel"]
        Defaults to 0.00

    ------- Returns -------
    freq_track: numpy.array of floats
        The frequencies contained in the track matching the given degree and
        radial order. The tracks are expected to be sorted from smallest Xc to
        largest Xc. Some IDs are repeated, usually when modes are switching
        their radial order identity from +1 to -1 or vice versa. As such, we
        only track the last frequency if the requested radial order is positive
        or zero and the first frequency if the requested n_pg is negative.
    """
    return tracks["freqs"][tracks["ID_lookup"][(l,m,n_pg)], \
                           tracks["frot_lookup"][frot_rel], :]


def get_mode_frequency_model(model, l, m, n_pg) :
    """
    Extract the frequency of a mode with the given degree, azimuthal order and
    radial order from the model dictionary. If the given identity is not in the
    model, np.nan is returned

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    l: int
        Degree of the mode
    m: int
        Azimuthal order of the mode
    n_pg: int
        Radial order of the mode

    ------- Returns -------
    freq: float
        The model frequency of the given identity in cycles per day. NaN if not
        available
    """
    try :
        return model["freqs"][model["ID_lookup"][(l,m,n_pg)]][0]
    except IndexError :
        return np.nan


def get_mode_splitting_model(model, l, m, n_pg) :
    """
    Find the splitting of the given mode ID in a model to the zonal mode of the
    frequency with the same degree and radial order.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    l: int
        Degree of the mode
    m: int
        Azimuthal order of the mode
    n_pg: int
        Radial order of the mode

    ------- Returns -------
    splitting: float
        The difference between the predicted mode with the given ID and the
        corresponding zonal mode in cycles per day. Will be np.nan if either the
        given ID or zonal ID is not in the model.
    """
    assert m != 0
    freq = get_mode_frequency_model(model, l, m, n_pg)
    freq_m0 = get_mode_frequency_model(model, l, 0, n_pg)
    if np.isnan(freq) or np.isnan(freq_m0) :
        return np.nan
    return abs(freq - freq_m0)


##### INTERPOLATION ############################################################


def age_to_Xc(tracks, age) :
    """
    Find the central hydrogen fraction at a given age by interpolating the
    history output

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    age: float
        The age in years to which to interpolate

    ------- Returns -------
    Xc: float
        Interpolated central hydrogen fraction

    """
    hist_age, hist_Xc = read_MESA_history_columns(tracks["histfile"], 
                                                  tracks["lookup_histfile"], 
                                                  ["star_age", "center_h1"],
                                                  Xc_min=tracks["Xc_min"],
                                                  Xc_max=tracks["Xc_max"],
                                                  flip=False)
    spline = scint.CubicSpline(hist_age, hist_Xc, extrapolate=False)
    Xc = float(spline(age, extrapolate=False))
    if Xc < tracks["Xc_min"] or Xc > tracks["Xc_max"] :
        print("age_to_Xc: Xc outside of limits in", tracks["histfile"],
              " : ",  Xc)
    return Xc


def find_age_cross_zonal_mode(tracks, n_pg, freq_obs, l=0, frot_rel=0.00) :
    """
    Look for the age at which a given observed frequency of a radial mode
    crosses the interpolated track of given radial order. In case there are
    multiple crossings, we assume the most likely one is where the derivative of
    the radial frequency to time is the smallest, since it is more likely to see
    a star at that point.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    n_pg: int
        Radial order of the mode
    freq_obs: float
        The observed frequency in cycles per day of the identified radial mode
    l: int
        Degree of the mode.
        Defaults to 0
    frot_rel: float
        The rotation rate relative to the Keplerian critical rotation rate of
        the track to use.

    ------- Returns -------
    root: float
        Central hydrogen fraction at which the observed frequency crosses the
        track. If there are multiple crossing points for a radial mode, that
        indicates there is a second crossing around or just after the TAMS.
        In that case we pick the one with the largest central hydrogen fraction.
        np.nan if no crossing point is found in the frequency range covered by
        this track.
    """
    freq_track = get_frequency_track(tracks, l, 0, n_pg, frot_rel=frot_rel)
    ages = tracks["ages"]
    ages = ages[~np.isnan(freq_track)]
    freq_track = freq_track[~np.isnan(freq_track)]
    freq_track -= freq_obs
    # Interim solution. Later on, I might add a check on Xc_interp to only
    # include points between closely spaced freq_track
    if len(ages) < 2 :
        return np.nan
    spline = scint.CubicSpline(ages, freq_track, extrapolate=False)
    roots = spline.roots(extrapolate=False)
    # No crossing found in our range. Note that we do not extrapolate!
    if len(roots) == 0 :
        return np.nan

    derivative = spline.derivative(nu=1) # First order derivative
    derivatives_at_roots = derivative(roots)
    return roots[np.argmin(abs(derivatives_at_roots))]


def interpolate_frequency_track(tracks, l, m, n_pg, frot_rel=0.00, Xc_min=1e-4,
                                Xc_max=0.702, N_interp=1000) :
    """
    Cubic interpolation of a track of frequencies of the mode with the given
    identity. This does not do any extrapolation, the interpolated frequencies
    outside the range between the smallest and largest Xc at which the frequency
    is not NaN are always NaN.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    l: int
        Degree of the mode
    m: int
        Azimuthal degree of the mode
    n_pg: int
        Radial order of the mode
    frot_rel: float
        Rotation rate relative to critical of the track. Must be in
        tracks["frot_rel"]
        Defaults to 0.00
    Xc_min: float
        Smallest hydrogen fraction to which to interpolate
        Defaults to 0.0001
    Xc_max: float
        Largest hydrogen fraction to which to interpolate
        Defaults to 0.702
    N_interp: int
        Number of points at which to interpolate
        Defaults to 1000

    ------- Returns -------
    Xc_interp: numpy.array of floats
        The central hydrogen fractions at which the frequency is interpolated
    freqs_interp: numpy.array of floats
        The interpolated frequency. Will equal NaN if outside the frequency
        range of the original track
    """
    assert Xc_min < Xc_max
    assert N_interp > 0
    Xc_interp = np.linspace(Xc_min, Xc_max, N_interp)
    Xcs = tracks["Xcs"]
    freq_track = get_frequency_track(tracks, l, m, n_pg, frot_rel=frot_rel)
    Xcs = Xcs[~np.isnan(freq_track)]
    freq_track = freq_track[~np.isnan(freq_track)]
    # Interim solution. Later on add a check on Xc_interp to only include points
    # between closely spaced freq_track
    if len(Xcs) < 2 :
        a = np.empty(N_interp)
        a[:] = np.nan
        freqs_interp = a
        return Xcs, freqs_interp
    spline = scint.CubicSpline(np.flip(Xcs), np.flip(freq_track),
                               extrapolate=False)
    freqs_interp = spline(Xc_interp)
    return Xc_interp, freqs_interp


def interpolate_history_columns_in_age(tracks, age, column_names) :
    """
    Find the values of the given MESA history columns at the given age by
    interpolating the history output

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    age: float
        The age in years to which to interpolate
    column_names: list of string
        The history columns to interpolate to. Each string must be a column in
        the MESA history output.

    ------- Returns -------
    interpolated_columns: list of floats
        History columns interpolated to the given age, with the i'th element of
        this list corresponding to the i'th history column name in columns. Each
        element will be np.nan if age lies outside of the age range covered
        between Xc_min and Xc_max of the tracks. Each element is converted to
        float.

    """
    column_names.append("star_age")
    columns = read_MESA_history_columns(tracks["histfile"], 
                                        tracks["lookup_histfile"], column_names,
                                        Xc_min=tracks["Xc_min"],
                                        Xc_max=tracks["Xc_max"], flip=False)
    history_age = columns[-1]
    interpolated_columns = np.empty(len(column_names))
    for i,column in enumerate(columns[:-1]) :
        spline = scint.CubicSpline(history_age, column, extrapolate=False)
        interpolated_columns[i] = spline(age)
    return interpolated_columns


def interpolate_model_in_frot_with_GYRE(observation, model_nonRot) :
    """
    Get a new model with frequencies interpolated to the rotation frequency best
    matching the provided observed splittings. This optimal rotation rate is
    computed from the mean splitting of all multiplets and the Ledoux constants
    computed with GYRE on a non-rotating model. Subsequently, the model only
    contains a first order approximation of rotational splitting. Models also
    only needs to contain one model with frot==0.0 . To achieve this, set
    frot_rels_to_include = [0.00]

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    model_nonRot: dicts
        One model with rotation rate of zero as created by
        interpolate_track_to_model_in_age of the same evolutionary track at
        different rotation rates. The model should also contain the beta of each
        multiplet

    ------- Returns -------

    model_best_frot: dict containing
        "age": float
            The age of the model in years
        "Xc": float
            The central hydrogen fraction
        "Xc_min": float
            Smallest hydrogen fraction to which to interpolate
        "Xc_max": float
            Largest hydrogen fraction to which to interpolate
        "frot_rel_best": float
            Best matching rotation frequency relative to the critical rotation
            frequency
        "frot_best": float
            Best matching rotation frequency in cycles per day
        "ls": numpy.array of ints
            Degrees of the modes
            array length equals K
        "ms": numpy.array of ints
            Azimuthal orders of the modes
            array length equals K
        "n_pgs": numpy.array of ints
            Radial orders of the modes
            array length equals K
        "freqs": numpy.array of floats
            Frequency in c/d for each mode identity in ls and n_pgs,
            interpolated to Xc
            array length equals K
        "log_Teff": float
            Base 10 logarithm of the effective temperature of the MESA model,
            interpolated to Xc
        "log_L": float
            Base 10 logarithm of the luminosity of the MESA model, interpolated
            to Xc
        "M": float
            Stellar mass in solar masses. *Not* the initial stellar mass. 
        "MESA_model_dir": string
            Path to the directory containing the output of this particular MESA
            run
        "histfile": string
            Path to the history file of this MESA model
        with K the number of mode identities.
    """

    model_best_frot = {}
    # Copy over the basics that should remain constant with the non-rotating model
    for key in ["Xc_min", "Xc_max", "MESA_model_dir", "histfile", "age", "Xc",
                "log_Teff", "log_L", "M", "ls", "ms", "n_pgs", "betas"] :
        model_best_frot[key] = model_nonRot[key]

    if np.isnan(model_nonRot["freqs"]).all() :
        # No workable solutions. Set everything to nan and send it out the door.
        model_best_frot["frot_best"], model_best_frot["frot_rel_best"] = np.nan, np.nan
        model_best_frot["age"], model_best_frot["Xc"] = np.nan, np.nan
        model_best_frot["log_Teff"], model_best_frot["log_L"], model_best_frot["M"] = np.nan, np.nan, np.nan
        model_best_frot["freqs"] = np.full(len(model_best_frot["ls"]), np.nan)
        return model_best_frot

    assert model_nonRot["frot"] == 0.0

    mask = (model_nonRot["ms"] == 0) & (model_nonRot["ls"] != 0)

    freqs_zonal_model = model_nonRot["freqs"][mask]
    ls_zonal_model = model_nonRot["ls"][mask]
    n_pgs_zonal_model = model_nonRot["n_pgs"][mask]
    betas_zonal_model = model_nonRot["betas"][mask]

    # Find the mean splitting of each observed multiplet
    ms_abs_all, splittings_all, splittings_err_all, betas_all = np.array([]), np.array([]), np.array([]), np.array([])
    for l, freq, splittings, splittings_err in zip(observation["ls"],
                                                   observation["freqs"],
                                                   observation["splittings"],
                                                   observation["splittings_err"]) :
        # Find the beta of each multiplet
        # Find the radial order of the zonal mode with identical l nearest in zonal frequency
        try :
            n_pgs_match = np.array(n_pgs_zonal_model)[(np.array(ls_zonal_model) == l)]
            freqs_match = np.array(freqs_zonal_model)[(np.array(ls_zonal_model) == l)]
            n_pg_min = n_pgs_match[np.nanargmin(np.abs(freqs_match - freq))]
        except ValueError :
            # No workable solutions. Set everything to nan and send it out the door.
            model_best_frot["frot_best"], model_best_frot["frot_rel_best"] = np.nan, np.nan
            model_best_frot["age"], model_best_frot["Xc"] = np.nan, np.nan
            model_best_frot["log_Teff"], model_best_frot["log_L"], model_best_frot["M"] = np.nan, np.nan, np.nan
            model_best_frot["freqs"] = np.full(len(model_best_frot["ls"]), np.nan)
            return model_best_frot

        try :
            beta = np.array(betas_zonal_model)[(np.array(ls_zonal_model) == l) &
                                               (np.array(n_pgs_zonal_model) == n_pg_min)][0]
        except IndexError :
            raise Exception("no matching beta found")

        ms = np.arange(-l, l+1)
        ms = ms[ms != 0]
        valid = ~np.isnan(splittings)

        ms_abs_all = np.concatenate(ms_abs_all, np.abs(ms)[valid])
        splittings_all = np.concatenate(splittings_all, splittings[valid])
        splittings_err_all = np.concatenate(splittings_err_all, splittings_err[valid])
        betas_all = np.concatenate(betas_all, np.full(len(ms[valid]), beta))

    ms_abs_all = np.array(ms_abs_all)
    splittings_all = np.array(splittings_all)
    splittings_err_all = np.array(splittings_err_all)
    betas_all = np.array(betas_all)

    # Find the rotation frequency optimally reproducing all rotational splitting
    frot_best = np.sum(abs(ms_abs_all) * betas_all * splittings_all / splittings_err_all**2) / \
                np.sum((ms_abs_all * betas_all / splittings_err_all)**2)

    # Add the (relative) rotation to the final model
    model_best_frot["frot_best"] = frot_best
    model_best_frot["frot_rel_best"] = frot_best / model_nonRot["f_K"]

    # Compute all the predicted modes in the model with first order 
    # rotationally-induced frequency shift
    freqs = model_nonRot["freqs"]
    ms = model_nonRot["ms"]
    betas = model_nonRot["betas"]
    model_best_frot["freqs"] = freqs + ms*betas*frot_best

    return model_best_frot


def interpolate_models_in_frot_with_StORM(observation, models) :
    """
    Get a new model with frequencies interpolated to the rotation frequency best
    matching the provided observed splittings. Herein the optimal rotation
    frequency is found by comparing all observed, identified rotational
    splitting to their predicted counterparts for each provided model. Then we
    compute the chi2 for each model and search for the minimum using cubic
    interpolation. Finally, we linearly interpolate all frequencies and stellar
    parameters to that rotation rate.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    models: list of dicts
        Models as created by interpolate_track_to_model_in_age of the same
        evolutionary track at different rotation rates

    ------- Returns -------

    model_best_frot: dict containing
        "age": float
            The age of the model in years
        "Xc": float
            The central hydrogen fraction
        "Xc_min": float
            Smallest hydrogen fraction to which to interpolate
        "Xc_max": float
            Largest hydrogen fraction to which to interpolate
        "frot_rel_best": float
            Best matching rotation frequency relative to the critical rotation
            frequency
        "frot_best": float
            Best matching rotation frequency in cycles per day
        "ls": numpy.array of ints
            Degrees of the modes
            array length equals K
        "ms": numpy.array of ints
            Azimuthal orders of the modes
            array length equals K
        "n_pgs": numpy.array of ints
            Radial orders of the modes
            array length equals K
        "freqs": numpy.array of floats
            Frequency in c/d for each mode identity in ls and n_pgs,
            interpolated to Xc
            array length equals K
        "log_Teff": float
            Base 10 logarithm of the effective temperature of the MESA model,
            interpolated to Xc
        "log_L": float
            Base 10 logarithm of the luminosity of the MESA model, interpolated
            to Xc
        "M": float
            Stellar mass in solar masses. *Not* the initial stellar mass. 
        "MESA_model_dir": string
            Path to the directory containing the output of this particular MESA
            run
        "histfile": string
            Path to the history file of this MESA model
        with K the number of mode identities.
    """
    model_best_frot = {}
    # Copy over the things that should remain constant across all models
    for key in ("Xc_min", "Xc_max", "ls", "ms", "n_pgs", "MESA_model_dir",
                "histfile") :
        model_best_frot[key] = models[0][key]

    # Extract the quantities to interpolate from all the models:
    #   age, Xc, Teff, L, all the frequencies. And frot & frot_rel of course!
    F = len(models)
    ages      = np.array([m["age"]      for m in models])
    Xcs       = np.array([m["Xc"]       for m in models])
    log_Teffs = np.array([m["log_Teff"] for m in models])
    log_Ls    = np.array([m["log_L"]    for m in models])
    Ms        = np.array([m["M"]        for m in models])
    frots     = np.array([m["frot"]     for m in models])
    frots_rel = np.array([m["frot_rel"] for m in models])
    all_freqs = np.stack([m["freqs"] for m in models], axis=1)

    # Remove outliers in Xc! Outlier = 3sigma away from the median Xc
    Xc_median = np.median(Xcs)
    Xc_mad = np.median(np.abs(Xcs - Xc_median))
    toKeep = np.abs(Xcs - Xc_median) < 3 * 1.4826 * Xc_mad
    # Sort by frot for interpolation
    I = np.argsort(frots_rel[toKeep])

    ages = ages[toKeep][I]
    Xcs = Xcs[toKeep][I]
    log_Teffs = log_Teffs[toKeep][I]
    log_Ls = log_Ls[toKeep][I]
    Ms = Ms[toKeep][I]
    all_freqs = all_freqs[:,toKeep][I]
    frots = frots[toKeep][I]
    frots_rel = frots_rel[toKeep][I]

    # Find the rotation producing the smallest chi2 in splitting around the
    # selected zonal frequency.
    chi2s = np.zeros(F)
    # The arrays below were used in debugging and some old experiments.
    # May prove useful again in the future, so I'm not deleting them just yet
    # chi2s_left = np.zeros(F)
    # chi2s_right = np.zeros(F)
    # splittings_left = np.zeros(F)
    # splittings_right = np.zeros(F)

    # We use the non-rotating model to find the closest radial order
    idx = np.flatnonzero(toKeep)
    model_nonRot = models[idx[I[0]]]
    assert model_nonRot["frot_rel"] == 0.0
    mask = (model_nonRot["ls"] == l) & (model_nonRot["ms"] == 0)
    n_pgs = model_nonRot["n_pgs"][mask]

    obs_ls = observation["ls"]
    obs_npgs = observation["ls"]
    obs_freqs = observation["freqs"]
    obs_splits = observation["splittings"]
    obs_errs = observation["splittings_err"]

    for k in range(observation["N_m"]) :
        l = obs_ls[k]
        n_pg = obs_npgs[k]
        observation_freq_m0 = obs_freqs[k]
        observation_splittings = np.array(obs_splits[k])
        observation_splittings_err = np.array(obs_errs[k])

        # If the radial order is given, compute the chi2 contribution of this
        # zonal frequency using that specific radial order in the model
        if not np.isnan(n_pg) :
            n_pg_min = n_pg
        else :
            # Find the zonal frequency with the correct degree closest to the
            # observed frequency
            freqs_model = np.array([
                get_mode_frequency_model(model_nonRot, l, 0, n_pg)
                for n_pg in n_pgs
            ])
            n_pg_min = n_pgs[np.nanargmin(np.abs(freqs_model - observation_freq_m0))]

        # Get the predicted splitting at each frot and add them to the chi2
        ms = np.arange(-l, l+1)
        ms = ms[ms != 0]

        valid = ~np.isnan(observation_splittings)

        ms = ms[valid]
        observation_splittings_nonNaN = observation_splittings[valid]
        observation_splittings_err_nonNaN = observation_splittings_err[valid]

        for m, observation_split, observation_split_err in zip(ms, 
                                                               observation_splittings_nonNaN, 
                                                               observation_splittings_err_nonNaN):
            splittings_model = np.array([
                get_mode_splitting_model(models[i], l, m, n_pg_min)
                for i in I
            ])
            inv_var = 1.0 / observation_split_err**2
            chi2s += (splittings_model - observation_split)**2 * inv_var
            # if m == -1 :
            #     chi2s_left[i] += chi2
            #     splittings_left[i] = get_mode_splitting_model(models[i], l,
            #                                                   m, n_pg_min)
            # if m == 1 :
            #     chi2s_right[i] += chi2
            #     splittings_right[i] = get_mode_splitting_model(models[i], l,
            #                                                    m, n_pg_min)

    toKeep = [(~np.isnan(frots)) & (~np.isnan(chi2s))][0]
    if np.isnan(chi2s).all() or np.sum(toKeep) < 4 :
        # No workable solutions. Set everything to nan and send it out the door.
        model_best_frot["frot_best"], model_best_frot["frot_rel_best"] = np.nan, np.nan
        model_best_frot["age"], model_best_frot["Xc"] = np.nan, np.nan
        model_best_frot["log_Teff"], model_best_frot["log_L"], model_best_frot["M"] = np.nan, np.nan, np.nan
        model_best_frot["freqs"] = np.full(len(model_best_frot["ls"]), np.nan)
        return model_best_frot

    # Look for where the minimum roughly is
    frot_min = frots[np.nanargmin(chi2s)]
    # Get the spline, its derivative and find the root of said derivative, which
    # should correspond to the minimum.
    spline = scint.CubicSpline(frots[toKeep], chi2s[toKeep], extrapolate=False)
    derivative = spline.derivative(nu=1)
    roots = derivative.roots(extrapolate=False)
    if len(roots) == 0 or np.sum(~np.isnan(roots)) == 0 :
        frot_best = frot_min
    elif np.isnan(frot_min) :
        # Not sure how to handle these, so just kill off the rotation completely
        frot_best = 0.0
    else :
        frot_best = roots[np.nanargmin(abs(roots - frot_min))]
    # Add frot and frot_rel (scaled by the last values) to the model dictionary
    model_best_frot["frot_best"] = frot_best
    model_best_frot["frot_rel_best"] = frot_best * frots_rel[-1]/frots[-1]
    model_best_frot["age"] = np.interp(frot_best, frots[toKeep], ages[toKeep])
    model_best_frot["Xc"] = np.interp(frot_best, frots[toKeep], Xcs[toKeep])
    model_best_frot["log_Teff"] = np.interp(frot_best, frots[toKeep],
                                            log_Teffs[toKeep])
    model_best_frot["log_L"] = np.interp(frot_best, frots[toKeep],
                                         log_Ls[toKeep])
    model_best_frot["M"] = np.interp(frot_best, frots[toKeep], Ms[toKeep])

    # Treat freqs separately as they are lists instead of floats
    K = len(model_best_frot["ls"])
    freqs_interp = np.empty()
    for k in range(K) :
        # Each model should have the same l, m, n_pg in the same order
        freq_track = all_freqs[k,:]
        toKeep = [~np.isnan(freq_track)][0]
        if np.sum(toKeep) <= 2 or np.min(abs(frots_rel[toKeep] - \
                                             frot_best/model_nonRot["f_K"])) > 0.03 :
            freqs_interp[k] = np.nan
        else :
            freqs_interp[k] = np.interp(frot_best, frots[toKeep], 
                                        freq_track[toKeep])
    model_best_frot["freqs"] = freqs_interp

    return model_best_frot


def interpolate_track_to_model_in_age(tracks, age, frot_rel=0.00,
                                      add_beta=False) :
    """
    Get a dictionary describing a model interpolated to the given central
    hydrogen fraction of the given relative rotation frequency

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    age: float
        The age in years to which to interpolate
    frot_rel: float
        The rotation frequency relative to the Keplerian critical rotation
        frequency
        Defaults to zero
    add_beta: bool
        Whether to include the beta_nl constant from GYRE for each multiplet in
        the model
        Defaults to False

    ------- Returns -------
    model: dict containing
        "age": float
            The age of the model in years
        "Xc": float
            The central hydrogen fraction
        "Xc_min": float
            Smallest hydrogen fraction to which to interpolate
        "Xc_max": float
            Largest hydrogen fraction to which to interpolate
        "frot_rel": float
            Rotation frequency relative to the critical rotation frequency
        "frot": float
            Rotation frequency in cycles per day
        "f_K": float
            Keplerian critical rotation frequency in cycles per day
        "ls": numpy.array of ints
            Degrees of the modes
            array length equals K
        "ms": numpy.array of ints
            Azimuthal orders of the modes
            array length equals K
        "n_pgs": numpy.array of ints
            Radial orders of the modes
            array length equals K
        "freqs": numpy.array of floats
            Frequency in c/d for each mode identity in ls and n_pgs,
            interpolated to Xc
            array length equals K
        "log_Teff": float
            Base 10 logarithm of the effective temperature of the MESA model,
            interpolated to Xc
        "log_L": float
            Base 10 logarithm of the luminosity of the MESA model, interpolated
            to Xc
        "M": float
            Stellar mass in solar masses. *Not* the initial stellar mass. 
        "MESA_model_dir": string
            Path to the directory containing the output of this particular MESA
            run
        "histfile": string
            Path to the history file of this MESA model
        "betas": numpy array of floats
            The beta_nl of each mode
            array length equals K
            Only included if add_beta is True
        with K the number of mode identities.
    """
    model = {"age":age ,
             "ls":tracks["ls"], "ms":tracks["ms"], "n_pgs":tracks["n_pgs"],
             "Xc_min":tracks["Xc_min"], "Xc_max":tracks["Xc_max"]}
    history_age, history_Xc, history_log_Teff, history_log_L, history_M = \
        read_MESA_history_columns(tracks["histfile"], tracks["lookup_histfile"],
                                  ["star_age","center_h1","log_Teff","log_L","star_mass"],
                                  Xc_min=tracks["Xc_min"],
                                  Xc_max=tracks["Xc_max"], flip=False)
    spline = scint.CubicSpline(history_age, history_log_Teff)
    model["log_Teff"] = float(spline(age, extrapolate=False))
    spline = scint.CubicSpline(history_age, history_log_L)
    model["log_L"] = float(spline(age, extrapolate=False))
    spline = scint.CubicSpline(history_age, history_M)
    model["M"] = float(spline(age, extrapolate=False))
    spline = scint.CubicSpline(history_age, history_Xc)
    model["Xc"] = float(spline(age, extrapolate=False))
    spline = scint.CubicSpline(tracks["ages"], tracks["f_Ks"])
    model["f_K"] = float(spline(age, extrapolate=False))

    freqs_interp = np.empty(len(model["ls"]))
    ages = tracks["ages"]
    Xcs = tracks["Xcs"]
    for k,(l,m,n_pg) in enumerate(zip(model["ls"],model["ms"],model["n_pgs"])) :
        freq_track = get_frequency_track(tracks, l, m, n_pg, frot_rel=frot_rel)
        ages_noNan = ages[~np.isnan(freq_track)]
        Xcs_noNan = Xcs[~np.isnan(freq_track)]
        # If there is no output within 5% Xc, the interpolation would be so
        # unreliable as to be virtually useless, so we manually add a NaN
        # Likewise, if all (but one) freqs are NaN, add a NaN as the
        # interpolated frequency
        if (len(ages_noNan) <= 1) or \
           (np.nanmin(abs(Xcs_noNan - model["Xc"])) > 0.05) :
            freqs_interp[k] = np.nan
        else :
            freq_track = freq_track[~np.isnan(freq_track)]
            spline = scint.CubicSpline(ages_noNan, freq_track)
            freqs_interp[k] = spline(age, extrapolate=False)
    model["freqs"] = freqs_interp #np.array(freqs_interp[~np.isnan(freqs_interp)])
    model["lookup_ID"] = tracks["lookup_ID"]
    model["MESA_model_dir"] = tracks["MESA_model_dir"]
    model["histfile"] = tracks["histfile"]
    model["lookup_histfile"] = tracks["lookup_histfile"]
    model["frot_rel"] = frot_rel
    model["frot"] = frot_rel * model["f_K"]

    if add_beta :
        betas_interp = np.empty(len(tracks["betas"]))
        for k,beta_track in enumerate(tracks["betas"]) :
            ages_noNan = ages[~np.isnan(beta_track)]
            betas_noNan = beta_track[~np.isnan(beta_track)]
            if len(betas_noNan) <= 2 :
                betas_interp[k] = np.nan
            else :
                spline = scint.CubicSpline(ages_noNan, betas_noNan)
                betas_interp[k] = float(spline(age, extrapolate=False))
        model["betas"] = betas_interp

    return model


def Xc_to_age(tracks, Xc) :
    """
    Find the age at a given central hydrogen fraction by interpolating the
    history output

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    Xc: float
        The central hydrogen fraction to which to interpolate

    ------- Returns -------
    age: float
        Interpolated age in years

    """
    history_age, history_Xc = read_MESA_history_columns(tracks["histfile"],
                                                        tracks["lookup_histfile"],
                                                        ["star_age", "center_h1"],
                                                        Xc_min=tracks["Xc_min"],
                                                        Xc_max=tracks["Xc_max"],
                                                        flip=True)
    spline = scint.CubicSpline(history_Xc, history_age)
    age = float(spline(Xc, extrapolate=False))
    return age



##### MODEL QUALITY FUNCTIONS ##################################################


def AIC(observation, model, Nfit=5) :
    """
    Computes the Akaike Information Criterion of the provided model given the
    provided observations.

    Author: Mathijs Vanrespaille, drawing from Luc IJspeert

    ------- Parameters -------
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    Nfit: int
        Number of fit parameters.
        Defaults to 5 (for Xc, frot, M, logD and fov)

    ------- Returns -------
    aic: float
        Akaike Information Criterion

    Notes
    -----
    AIC = −2 ln(L(θ)) + 2 k
    where L is the likelihood as function of the parameters θ and k the number
    of free parameters.

    Under the assumption that the errors are independent and distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -sum(ln(sqrt(2 pi) σ) + (y_obs-y_model)^2/(2σ^2)).
    """
    k = Nfit
    aic = k*2
    # Contribution of effective temperature
    if not np.isnan(observation["log_Teff"]) :
        aic += np.log(2*np.pi*observation["log_Teff_err"]**2) + \
                2*(model["log_Teff"] - observation["log_Teff"])**2 / \
                    (2*observation["log_Teff_err"]**2)
    # Contribution of luminosity
    if not np.isnan(observation["log_L"]) :
        aic += np.log(2*np.pi*observation["log_L_err"]**2) + \
                2*(model["log_L"] - observation["log_L"])**2 / \
                    (2*observation["log_L_err"]**2)
    for l, n_pg, freq, freq_err, splittings, splittings_err in \
        zip(observation["ls"], observation["n_pgs"],
            observation["freqs"], observation["freqs_err"],
            observation["splittings"], observation["splittings_err"]):
        # Contribution of each zonal frequency
        aic += np.log(2*np.pi*freq_err**2) + \
                2*(freq - get_mode_frequency_model(model, l, 0, n_pg))**2 / \
                    (2*freq_err**2)
        m = -l
        for split, split_err in zip(splittings, splittings_err) :
            # Skip zonal mode, already taken care of
            if m == 0 :
                m += 1
            if np.isnan(split) :
                # Update azimuthal order
                m += 1
                continue
            # Contribution of each splitting
            aic += np.log(2*np.pi*split_err**2) + \
                    2*(freq - get_mode_splitting_model(model, l, m, n_pg))**2 / \
                        (2*split_err**2)
            # Update azimuthal order
            m += 1
    return aic


def AICc(observation, model, Nfit=5) :
    """
    Computes the corrected Akaike Information Criterion of the provided model
    given the provided observations.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    Nfit : int
        Number of fit parameters.
        Defaults to 5 (for Xc, frot, M, logD and fov)

    ------- Returns -------
    aicc: float
        corrected Akaike Information Criterion

    Notes
    -----
    AICc = −2 ln(L(θ)) + 2 k + 2 k (k+1) / (n - k - 1)
    where L is the likelihood as function of the parameters θ, n the number of
    data points and k the number of free parameters.

    Under the assumption that the errors are independent and distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -sum(ln(sqrt(2 pi) σ) + (y_obs-y_model)^2/(2σ^2)).
    """
    k = Nfit
    n = number_observations(observation)
    aicc = AIC(observation, model) + 2*k*(k+1) / (n-k-1)
    return aicc


def BIC(observation, model, Nfit=5):
    """
    Computes the Bayesian Information Criterion of the provided model given the
    provided observations.

    Author: Mathijs Vanrespaille, drawing from Luc IJspeert

    ------- Parameters -------
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    Nfit : int
        Number of fit parameters.
        Defaults to 5 (for Xc, frot, M, logD and fov)

    ------- Returns -------
    bic: float
        Bayesian Information Criterion

    Notes
    -----
    BIC = −2 ln(L(θ)) + k ln(n)
    where L is the likelihood as function of the parameters θ, n the number of
    data points and k the number of free parameters.

    Under the assumption that the errors are independent and distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -sum(ln(sqrt(2 pi) σ) + (y_obs-y_model)^2/(2σ^2)).
    """
    n = number_observations(observation)
    k = Nfit
    bic = k*np.log(n)
    # Contribution of effective temperature
    if not np.isnan(observation["log_Teff"]) :
        bic += np.log(2*np.pi*observation["log_Teff_err"]**2) + \
                2*(model["log_Teff"] - observation["log_Teff"])**2 / \
                    (2*observation["log_Teff_err"]**2)
    # Contribution of luminosity
    if not np.isnan(observation["log_L"]) :
        bic += np.log(2*np.pi*observation["log_L_err"]**2) + \
                2*(model["log_L"] - observation["log_L"])**2 / \
                    (2*observation["log_L_err"]**2)
    for l, n_pg, freq, freq_err, splittings, splittings_err in \
        zip(observation["ls"], observation["n_pgs"],
            observation["freqs"], observation["freqs_err"],
            observation["splittings"], observation["splittings_err"]):
        # Contribution of each zonal frequency
        bic += np.log(2*np.pi*freq_err**2) + \
                2*(freq - get_mode_frequency_model(model, l, 0, n_pg))**2 / \
                    (2*freq_err**2)
        m = -l
        for split, split_err in zip(splittings, splittings_err) :
            # Skip zonal mode, already taken care of
            if m == 0 :
                m += 1
            if np.isnan(split) :
                # Update azimuthal order
                m += 1
                continue
            # Contribution of each splitting
            bic += np.log(2*np.pi*split_err**2) + \
                    2*(freq - get_mode_splitting_model(model, l, m, n_pg))**2 / \
                        (2*split_err**2)
            # Update azimuthal order
            m += 1
    return bic


def is_model_compatible_with_spectroscopy(model, observation, sigma_tol=2.) :
    """
    Check if the model is sufficiently close to the spectroscopic observations
    to be considered plausible.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    sigma_tol: float
        How many times the uncertainty the model can be removed from the
        observations
        Defaults to 2

    ------- Returns -------
    compatible: bool
        Whether the model log_Teff and log_L are within 2sigma of the
        observations
    """
    log_Teff_model = model["log_Teff"]
    log_L_model = model["log_L"]
    log_Teff_obs = observation["log_Teff"]
    log_Teff_obs_err = observation["log_Teff_err"]
    log_L_obs = observation["log_L"]
    log_L_obs_err = observation["log_L_err"]
    return np.sqrt(((log_Teff_model - log_Teff_obs) / log_Teff_obs_err)**2 + \
                   ((log_L_model - log_L_obs) / log_L_obs_err)**2) < sigma_tol


def number_observations(observation, exclude_splitting=False) :
    """
    Compute the number of observations being fitted.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    exclude_splitting: bool
        Whether to not include rotational splitting in the count of the number
        of observational constraints
        Defaults to False

    ------- Returns -------
    Nobs: int
        Number of provided observations
    """
    if exclude_splitting :
        Nobs = observation["N_m"]
    else :
        Nobs = observation["N_f"]

    if not np.isnan(observation["log_Teff"]) :
        Nobs += 1
    if not np.isnan(observation["log_L"]) :
        Nobs += 1
    return Nobs


def reduced_chi2(model, observation) :
    """
    Compute the reduced chi-square of the provided model given the provided
    observations. This is a sum of the chi2 of log_Teff, log_L and each ID'ed
    frequency. If the provided observations do not include a specific radial
    order, all radial orders in the model are checked and the one closest to the
    observed frequency is used in the chi2 computation.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict

    ------- Returns -------
    chi2: float
        The reduced chi-square
        Will be NaN if the model does not feature a frequency of the degree
        (and radial order) provided in the observation
    """
    return standardised_chi2(model, observation) / \
        max((number_observations(observation) - 5),1)


def standardised_chi2(model, observation, exclude_splitting=False) :
    """
    Compute the standardised chi-square of the provided model given the provided
    observations. This is a sum of the chi2 of log_Teff, log_L and each ID'ed
    frequency. If the provided observations do not include a specific radial
    order, all radial orders in the model are checked and the one closest to the
    observed frequency is used in the chi2 computation.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    exclude_splitting: bool
        Whether to not include rotational splitting terms in the chi2 sum
        Defaults to False

    ------- Returns -------
    chi2: float
        The (smallest) standardised chi-square
        Will be NaN if the model does not feature a frequency of the degree
        (and radial order) provided in the observation
    """
    chi2 = 0
    if not np.isnan(observation["log_Teff"]) :
        chi2 += (model["log_Teff"] - observation["log_Teff"])**2 / \
                observation["log_Teff_err"]**2
    if not np.isnan(observation["log_L"]) :
        chi2 += (model["log_L"] - observation["log_L"])**2 / \
                observation["log_L_err"]**2

    # Skipping the first because the chi2 contribution of the first frequency is
    # zero as we force the model frequency onto the observed frequency
    ls_obs = observation["ls"]
    observation_freqs_m0 = observation["freqs"]
    observation_freqs_m0_err = observation["freqs_err"]
    observation_n_pgs = observation["n_pgs"]
    for k in range(observation["N_m"]) :
        l = ls_obs[k]
        observation_freq_m0 = observation_freqs_m0[k]
        observation_freq_m0_err = observation_freqs_m0_err[k]
        observation_n_pg = observation_n_pgs[k]
        # If the radial order is given, compute the chi2 contribution of this
        # zonal frequency using that specific radial order in the model
        if not np.isnan(observation_n_pg) :
            n_pg_min = observation_n_pg
            chi2 += (get_mode_frequency_model(model, l, 0, n_pg_min) - \
                    observation_freq_m0)**2 / observation_freq_m0_err**2
        # Otherwise, go over each radial order in the model and use the one with
        # the smallest chi2 contribution as the zonal frequency
        else :
            # Smallest chi2 of a mode with the given degree.
            # Start huge to ensure the first identity produces a smaller chi2
            mask = (model["ls"] == l) & (model["ms"] == 0)
            n_pgs_masked = model["n_pgs"][mask]
            freqs_masked = model["freqs"][mask]
            try :
                i_min = np.nanargmin(np.abs(freqs_masked - observation_freq_m0))
            except :
                return np.nan
            n_pg_min = n_pgs_masked[i_min]
            chi2 += (freqs_masked[i_min] - observation_freq_m0)**2 / \
                                observation_freq_m0_err**2

        # Now we know the best n_pg, check the predicted splittings and compare
        # to available observations, unless excluded of course
        if not exclude_splitting :
            observation_splittings = observation["splittings"][k]
            observation_splittings_err = observation["splittings_err"][k]
            m = -l
            ms = np.arange(-l, l+1)
            ms = ms[ms != 0]
            for m, observation_split, observation_split_err in \
                zip(ms, observation_splittings, observation_splittings_err) :
                model_split = get_mode_splitting_model(model, l, m, n_pg_min)
                chi2 += (model_split - observation_split)**2 / \
                        observation_split_err**2

    return chi2


def standardised_chi2_components(model, observation) :
    """
    Compute the standardised chi-square of the provided model given the provided
    observations and report each term, i.e. the terms from log_Teff, log_L,
    zonal frequencies and each splitting is returned in a dictionary.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict

    ------- Returns -------
    chi2: float
        The (smallest) standardised chi-square
        Will be NaN if the model does not feature a frequency of the degree
        (and radial order) provided in the observation
    """
    chi2_components = {}
    if not np.isnan(observation["log_Teff"]) :
        chi2_components["log_Teff"] = (model["log_Teff"] - \
                                       observation["log_Teff"])**2 / \
                                       observation["log_Teff_err"]**2
    else :
        chi2_components["log_Teff"] = np.nan
    if not np.isnan(observation["log_L"]) :
        chi2_components["log_L"] = (model["log_L"] - observation["log_L"])**2 / \
                                    observation["log_L_err"]**2
    else :
        chi2_components["log_L"] = np.nan

    chi2_components["freqs"] = []
    chi2_components["splittings"] = []
    ls = observation["ls"]
    observation_freqs_m0 = observation["freqs"]
    observation_freqs_m0_err = observation["freqs_err"]
    observation_splittingss = observation["splittings"]
    observation_splittingss_err = observation["splittings_err"]
    observation_n_pgs = observation["n_pgs"]
    for k in range(observation["N_m"]) :
        l = ls[k]
        observation_freq_m0 = observation_freqs_m0[k]
        observation_freq_m0_err = observation_freqs_m0_err[k]
        observation_splittings = observation_splittingss[k]
        observation_splittings_err = observation_splittingss_err[k]
        observation_n_pg = observation_n_pgs[k]

        # If the radial order is given, compute the chi2 contribution of this
        # zonal frequency using that specific radial order in the model
        if not np.isnan(observation_n_pg) :
            n_pg_min = observation_n_pg
            chi2_freq_min = (get_mode_frequency_model(model, l, 0, n_pg_min) - \
                            observation_freq_m0)**2 / observation_freq_m0_err**2
        # Otherwise, go over each radial order in the model and use the one with
        # the smallest chi2 contribution as the zonal frequency
        else :
            # Smallest chi2 of a mode with the given degree.
            # Start huge to ensure the first identity produces a smaller chi2
            mask = (model["ls"] == l) & (model["ms"] == 0)
            n_pgs_masked = model["n_pgs"][mask]
            freqs_masked = model["freqs"][mask]
            try :
                i_min = np.nanargmin(np.abs(freqs_masked - observation_freq_m0))
            except :
                return np.nan
            n_pg_min = n_pgs_masked[i_min]
            chi2_freq_min = (freqs_masked[i_min] - observation_freq_m0)**2 / \
                                observation_freq_m0_err**2
        chi2_components["freqs"].append(chi2_freq_min)

        # Now we know the best n_pg, check the predicted splittings and compare
        # to available observations, unless excluded of course
        observation_splittings = observation["splittings"][k]
        observation_splittings_err = observation["splittings_err"][k]
        m = -l
        ms = np.arange(-l, l+1)
        ms = ms[ms != 0]
        for m, observation_split, observation_split_err in \
            zip(ms, observation_splittings, observation_splittings_err) :
            model_split = get_mode_splitting_model(model, l, m, n_pg_min)
            chi2 = (model_split - observation_split)**2 / \
                    observation_split_err**2
            chi2_components["splittings"][-1].append(chi2)

    return chi2_components



##### NAVIGATING GRID ##########################################################


def does_track_cross_HRD_error_ellipse(tracks, observation, sigma_tol=2.1) :
    """
    Check if the model is sufficiently close to the spectroscopic observations
    to be considered plausible.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The list of dictionaries summarising the StORM output of a MESA
        evolutionary run for each dictionary as created read_storm_tracks
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    sigma_tol: float
        How many times the uncertainty the model can be removed from the
        observations
        Defaults to 2.1

    ------- Returns -------
    tracks_cross: array of bools
        Whether these evolutionary tracks cross the 2sigma error ellipse of the
        observations on the HRD
    """
    log_Teff_track = tracks["log_Teffs"]
    log_L_track = tracks["log_Ls"]
    log_Teff_obs = observation["log_Teff"]
    log_Teff_obs_err = observation["log_Teff_err"]
    log_L_obs = observation["log_L"]
    log_L_obs_err = observation["log_L_err"]
    return (np.sqrt(((log_Teff_track - log_Teff_obs) / log_Teff_obs_err)**2 + \
           ((log_L_track - log_L_obs) / log_L_obs_err)**2) < sigma_tol).any()


def find_nearest_model_number(tracks, Xc) :
    """
    Find the MESA model number at which MESA saved output with the actual
    hydrogen fraction closest to the given central hydrogen fraction. The
    difference with the similar function find_nearest_saved_model_number is that
    this function searches for any model number of the MESA run, while
    find_nearest_saved_model_number only considers the model numbers at which
    MESA wrote detailed output.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    Xc: float
        The central hydrogen fraction around which we seek the nearest saved
        central hydrogen fraction

    ------- Returns -------
    mn_closest: float
        The central hydrogen fraction around which MESA saved output closest in
        central hydrogen fraction to the provided Xc
    """
    model_numbers, center_h1s = read_MESA_history_columns(tracks["histfile"],
                                                          tracks["lookup_histfile"],
                                                          ["star_age", "center_h1"])
    # Index of closest Xc
    i = np.argmin(abs(center_h1s - Xc))
    mn_closest = model_numbers[i]
    return mn_closest


def find_nearest_saved_model_number(tracks, Xc) :
    """
    Find the MESA model number at which MESA saved output with the actual
    hydrogen fraction closest to the given central hydrogen fraction. The
    difference with the similar function find_nearest_model_number is that this
    function searches for only those model number at which MESA wrote detailed
    output, instead of any model number like find_nearest_model_number

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    Xc: float
        The central hydrogen fraction around which we seek the nearest saved
        central hydrogen fraction

    ------- Returns -------
    mn_save_closest: float
        The central hydrogen fraction around which MESA saved output closest in
        central hydrogen fraction to the provided Xc
    """
    # Index of closest Xc
    i = np.argmin(abs(tracks["Xcs"] - Xc))
    mn_save_closest = tracks["mns"][i]
    return mn_save_closest


def find_nearest_Xc_save(tracks, Xc) :
    """
    Find the Xc_save of the MESA output profile with the actual hydrogen
    fraction closest to the given central hydrogen fraction.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    tracks: dict
        The dictionary summarising the StORM output of a MESA run as created by
        read_storm_tracks
    Xc: float
        The central hydrogen fraction around which we seek the nearest saved
        central hydrogen fraction

    ------- Returns -------
    Xc_save_closest: float
        The central hydrogen fraction around which MESA saved output closest in
        central hydrogen fraction to the provided Xc
    """
    # Index of closest Xc
    i = np.argmin(np.min(abs(tracks["Xcs"] - Xc)))
    Xc_save_closest = tracks["Xc_saves"][i]
    return Xc_save_closest
