import numpy as np
import glob
import h5py
import os


##### MISCELLANEOUS ############################################################


def build_lookup_ID(ls, ms, n_pgs) :
    lookup_ID = {}
    IDs = np.column_stack((ls, ms, n_pgs))
    for (l, m, n_pg) in np.unique(IDs, axis=0) :
        # In case of duplicate IDs, we keep the one with the lower n_pg if it is
        # a p-mode and the higher n_pg if it is a g-mode, i.e. the lower-order 
        # mode. This only works if the provided n_pgs_to_include is sorted from
        # small to large
        if n_pg >= 0 :
            choice = -1
        else :
            choice = 0
        lookup_ID[(l, m, n_pg)] = np.where((ls==l) & (ms==m) & (n_pgs==n_pg))[0][choice]
    return lookup_ID


def split_line(line, sep) :
    """
    Splits a string in 2 parts.

    Author: Mathias Michielsen

    ------- Parameters -------
    line: string
        String to split in 2.
    sep: string
        Separator where the string has to be split around.

    ------- Returns -------
    head: string
        Part 1 of the string before the separator.
    tail: string
        Part 2 of the string after the separator.
    """
    head, sep_, tail = line.partition(sep)
    assert sep_ == sep
    return head, tail


def substring(line, sep_first, sep_second) :
    """
    Get part of a string between 2 specified separators.
    If second separator is not found, return everyting after first separator.

    Author: Mathias Michielsen

    ------- Parameters -------
    line: string
        String to get substring from.
    sep_first: string
        First separator after which the returned substring should start.
    sep_second: string
        Second separator at which the returned substring should end.

    ------- Returns -------
    head: string
        Part of the string between the 2 separators.
    """
    head, tail = split_line(line, sep = sep_first)
    if sep_second not in tail:
        return tail
    head, tail = split_line(tail, sep =sep_second)
    return head


##### INPUT ####################################################################


def create_observation_dict(row, sort_by_l, sort_by_freq) :
    """
    Read in the .tsv file containing all relevant observations and summarise
    each set of observations in a dictionary.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    row: pandas.core.series.Series
        Row of the pandas DataFrame containing all observational info of every
        run
    sort_by_l: bool
        Whether to sort the read multiplets by their degree
    sort_by_freq: bool
        Whether to sort the read multiplets by the frequency of their zonal mode
        from largest to smallest

    ------- Returns -------
    observation: dicts
        dictionary summarising all the observational info required for the
        forward modelling. Contains:
        "name": str
            an identifying name of the star
        "suffix": str
            a suffix to be added on the end of output files and plots of this .
            Useful when modelling the same star (i.e. the same name) with
            different input. Can be empty.
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
    # I immediately start gathering data in the dictionary. This is technically
    # inefficient, but since this bit of the code runs in the blink of an eye
    # anyway, I don't really care and I can't be bothered recoding all of this. 
    observation={"name":row["name"], "save_suffix":row["save_suffix"],
                 "log_Teff":row["log_Teff"], "log_Teff_err":row["log_Teff_err"],
                 "log_L":row["log_L"], "log_L_err":row["log_L_err"],
                 "N_m":int(row["Ni_ID"]), "N_f":int(row["Nm_ID"]),
                 "freqs":[], "freqs_err":[],
                 "splittings":[], "splittings_err":[],
                 "amps":[], "amps_splittings":[], "ls":[], "n_pgs":[],
                 "all_freqs":list(row["all_f"]), "all_amps":list(row["all_a"]),
                 "n_pg_main":row["n_pg_main"]}
    # First collect all freqs, l and i and check the zonal modes
    # i_multiplet indicates to which multiplet a frequency belongs
    i_multiplet, ms, freqs, freqs_err, amps = [], [], [], [], []
    for j in range(1,observation["N_f"]+1) :
        ms.append(int(row[f"m{j}"]))
        i_multiplet.append(int(row[f"i{j}"]))
        freqs.append(row[f"f{j}"])
        freqs_err.append(row[f"f_err{j}"])
        amps.append(row[f"a{j}"])
        if row[f"m{j}"] == 0 :
            observation["ls"].append(int(row[f"l{j}"]))
            observation["freqs"].append(row[f"f{j}"])
            observation["freqs_err"].append(row[f"f_err{j}"])
            observation["amps"].append(row[f"a{j}"])
            if f"n_pg{j}" in row.keys() :
                observation["n_pgs"].append(row[f"n_pg{j}"])
            else :
                observation["n_pgs"].append(np.nan)

    ms = np.array(ms)
    i_multiplet = np.array(i_multiplet)

    print(observation["name"])

    # Loop over each multiplet and, depending on degree, find each's splittings
    for j,i in enumerate(np.unique(i_multiplet)) :
        l = observation["ls"][j]
        freq_m0 = observation["freqs"][j]
        freq_err_m0 = observation["freqs_err"][j]
        if l > 2 or l < 0 :
            raise ValueError("create_observation_dict: observation file " \
                             "should not contain a degree greater than 2 or " \
                             "smaller than 0.")
        splitting, splitting_err, amps_splitting = [], [], []
        for m in range(-l, l+1) : # Goes over -l, -l+1,..., l-1,l, but skip m=0
            if m == 0 :
                continue
            if (m,i) in zip(ms,i_multiplet) :
                k = np.where((ms==m) & (i_multiplet==i))[0][0]
                splitting.append(abs(freqs[k] - freq_m0))
                splitting_err.append(np.sqrt(freqs_err[k]**2 + freq_err_m0**2))
                amps_splitting.append(amps[k])
            else :
                splitting.append(np.nan)
                splitting_err.append(np.nan)
                amps_splitting.append(np.nan)
        observation["splittings"].append(splitting)
        observation["splittings_err"].append(splitting_err)
        observation["amps_splittings"].append(amps_splitting)

    if sort_by_freq :
        # Now sort freqs, splittings and their errors by zonal frequency from
        # largest to smallest!
        I_toSort = np.argsort(-np.array(observation["freqs"]))
        observation["ls"] = [observation["ls"][i] for i in I_toSort]
        observation["n_pgs"] = [observation["n_pgs"][i] for i in I_toSort]
        observation["freqs"] = [observation["freqs"][i] for i in I_toSort]
        observation["freqs_err"] = [observation["freqs_err"][i] for i in I_toSort]
        observation["splittings"] = [observation["splittings"][i] for i in I_toSort]
        observation["splittings_err"] = [observation["splittings_err"][i] for i in I_toSort]
        observation["amps_splittings"] = [observation["amps_splittings"][i] for i in I_toSort]
        observation["amps"] = [observation["amps"][i] for i in I_toSort]

    if sort_by_l :
        # Now sort freqs, splittings and their errors by l such that a radial
        # mode always comes first!
        I_toSort = np.argsort(observation["ls"])
        observation["ls"] = [observation["ls"][i] for i in I_toSort]
        observation["n_pgs"] = [observation["n_pgs"][i] for i in I_toSort]
        observation["freqs"] = [observation["freqs"][i] for i in I_toSort]
        observation["freqs_err"] = [observation["freqs_err"][i] for i in I_toSort]
        observation["splittings"] = [observation["splittings"][i] for i in I_toSort]
        observation["splittings_err"] = [observation["splittings_err"][i] for i in I_toSort]
        observation["amps_splittings"] = [observation["amps_splittings"][i] for i in I_toSort]
        observation["amps"] = [observation["amps"][i] for i in I_toSort]

    return observation


def process_summary_file(storm_summary_path, 
                         ls_track, ms_track, n_pgs_track,
                         frots_rel_to_include, mn_lookup,
                         add_beta, MESA_model_dir):
    """
    Read in and summarise all the STORM output of a MESA run with the given
    parameters. This collects the predicted frequencies of the given
    combinations of degree l and radial order n_pgs at every output step. All
    output is collected into a dictionary, described below.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    storm_summary_path: string
        The full or relative path to StORM summary file to be loaded in
    ls_track: list of ints
        Degrees of the predicted modes to be read in
        Must be the same length as n_pgs_track and ms_track
    ms_track: list of ints
        Azimuthal orders of the predicted modes to be read in
        Must be the same length as n_pgs and ls_track
    n_pgs_track: list of ints
        Radial orders, within the Eckart-Scuflaire-Osaki-Takata scheme, of the
        predicted modes to be read in
        Must be the same length as ls_track and ms_track
    frots_rel_to_include: list of floats
        The rotation rates relative to the Keplerian critical rotation rate to
        be included. If left empty, all the rotation rates provided in the StORM
        summaries are used.
        Defaults to []
    mn_lookup: dict
        Dictionary used to look up the index of a predicted frequency in the
        summary file matching a particular mode identity.
        Keys are 3-element tuples like (l, m, n_pg) and values are ints 
    add_beta: bool
        Whether to include the beta_nl constant from GYRE for each multiplet in
        the model
        Setting this to True increases loading time considerably
        Defaults to False
    MESA_model_dir: string
        The full or relative path to the folder containing the MESA and StORM 
        output of this MESA evolutionary track
        Only used if add_beta is True

    ------- Returns -------
    tracks: dict containing
        "age": float
            The age in years of this instance of MESA output
        "mn": numpy.array of ints
            Model numbers at which MESA saved
        "Xc": float
            The actual central hydrogen fraction of this instance of MESA output
        "Xc_save": float
            The central hydrogen fraction around which MESA this instance of 
            MESA output
        "log_Teff": float
            base-10 logarithm of effective temperature in K
        "log_L": float
            base-10 logarithm of luminosity in solar luminosities
        "freqs": numpy.array of numpy.arrays of floats
            Contains an array for each mode identity (l,m,n_pgs), rotation rate
            array dimensions equals (K, F)
        "track_reliable": numpy.array of bools
            Whether the summarising algorithm did not detect any discrepancies
            with this each track
            array length equals K
        "f_K": float
            The Keplerian critical rotation rate of this instance of MESA output
            in cycles per day
        "betas": numpy array of floats
            The beta_nl of each mode
            Only included if add_beta is True
            array length equals K

    with K the number of mode identities read in and M the number of output
    steps of the model and F the number of considered rotation rates.
    Each array over time (those of length M) is sorted by Xc
    """
    # Get the model number 
    # (and central hydrogen fraction around which MESA saved; unused here)
    mn, Xc_save = read_model_number_and_Xc_save(storm_summary_path)
    # The real central hydrogen fraction of this MESA model and more
    age, Xc, logT, logL = mn_lookup[mn]

    with h5py.File(storm_summary_path, "r") as storm_summary :
        ls_ss = storm_summary['ls'][:]
        npgs_ss = storm_summary['n_pgs'][:]
        ms_ss = storm_summary["ms"][:]
        freqs_ss = storm_summary["freqs_deformed"][:]
        reliable_ss = storm_summary["track_reliable"][:]

        lookup = {
            (l, m, n_pg): i
            for i, (l, m, n_pg) in enumerate(zip(ls_ss, ms_ss, npgs_ss))
        }

        f_K = storm_summary.attrs["f_K"]
        frots_rel = np.array(storm_summary["frots_rel"])
        if not len(frots_rel_to_include) == 0 :
            # Get indices that are in the frots_rel_to_include
            I = np.isin(frots_rel, frots_rel_to_include)
        else :
            # Get all the indices
            I = np.full(len(frots_rel), True, dtype=bool) 

        reliables = np.empty(len(ls_track))
        # freqs has the shape (K x F)
        freqs = np.empty((len(ls_track), np.sum(I)))
        for k, (l, m, n_pg) in enumerate(zip(ls_track, ms_track, n_pgs_track)) :
            #index in storm summary
            iss = lookup.get((l, m, n_pg))
            if iss is not None :
                reliable = bool(reliable_ss[iss])
                reliables[k] = reliable
                if reliable :
                    freqs[k,:] = freqs_ss[iss, I]
                else :
                    freqs[k,:] = np.nan
            else :
                freqs[k,:] = np.nan
                reliable = False
            reliables[k] = reliable

    if add_beta :
        import pygyre as pyg
        betas = np.empty(len(ls_track))
        gyre_output_dir = f"{MESA_model_dir}/gyre_summaries/"
        gyre_model_path = f"{gyre_output_dir}/gyre_Xc{Xc_save:.5f}_mn{mn}.HDF".format(Xc_save=Xc_save, mn=mn)
        if not os.path.isfile(gyre_model_path) :
            betas[:] = np.nan
            print("Missing GYRE file : ", gyre_model_path)
            track = {"age": age, "mn": mn, "Xc": Xc, "Xc_save": Xc_save, "f_K": f_K,
               "freqs": freqs, "track_reliable": reliable,
               "log_Teff":logT, "log_L":logL, "betas":betas}
            return track
                
        gyre_output = pyg.read_output(gyre_model_path)

        for k, (l, n_pg) in enumerate(zip(ls_track, n_pgs_track)) :
            if l==1 and n_pg==0 :
                betas[k] = np.nan
            elif l == 0 :
                betas[k] = np.nan
            elif n_pg >= 4 :
                betas[k] = 1.0
            else :
                beta_data = np.real(gyre_output[(gyre_output['l']==l) & \
                                    (gyre_output['n_pg']==n_pg)]['beta']).data
                if len(beta_data) == 1 :
                    betas[k] = beta_data[0]
                else :
                    betas[k] = np.nan

    track = {"age": age, "mn": mn, "Xc": Xc, "Xc_save": Xc_save, "f_K": f_K,
               "freqs": freqs, "track_reliable": reliable,
               "log_Teff":logT, "log_L":logL}
    if add_beta :
        track["betas"] = betas
    return track


def read_MESA_history_column(history_file, lookup_header, column_name, 
                             dtype=float, Xc_min=None, Xc_max=None, flip=False) :
    """
    Read the given column from the given MESA history file

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    history_file: string
        Path to the history file
    lookup_header: dict
        A table to find the column number matching each history column name.
        Keys are strings and values are integers. 
    column_name: string
        Name of the column in the history columns header
    dtype: numpy.dtype
        Data type of the column to load in
        Defaults to float
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
    flip: bool
        Whether to flip the history column. If True, order from ZAMS to TAMS or
        with increasing age. If False, order from TAMS to ZAMS or with
        increasing Xc.
        Defaults to False

    ------- Returns -------
    column: numpy.array of given dtypes
        The selected history column, for those rows with Xc between Xc_min and
        Xc_max from small to large Xc
    """
    column = np.loadtxt(history_file, skiprows=6,
                        usecols=[lookup_header[column_name]],
                        dtype=dtype)
    if (Xc_min is not None) or (Xc_max is not None) :
        Xc = np.loadtxt(history_file, skiprows=6,
                        usecols=[lookup_header["center_h1"]],
                        dtype=dtype)
        if Xc_min is not None :
            I_toKeep = [Xc > Xc_min][0]
            column = column[I_toKeep]
            Xc = Xc[I_toKeep]
        if Xc_min is not None :
            I_toKeep = [Xc < Xc_max][0]
            column = column[I_toKeep]
            Xc = Xc[I_toKeep]
    # If requested, flipping orders such that it is ordered from smallest to
    # largest Xc
    if flip :
        return np.flip(column)
    else :
        return column


def read_MESA_history_columns(history_file, lookup_header, column_names, 
                              Xc_min=None, Xc_max=None, flip=False) :
    """
    Read the given column from the given MESA history file

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    history_file: string
        Path to the history file
    lookup_header: dict
        A table to find the column number matching each history column name.
        Keys are strings and values are integers. 
    column_names: list of strings
        Names of the column in the history columns header
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
    flip: bool
        Whether to flip the history column. If True, order from ZAMS to TAMS or
        with increasing age. If False, order from TAMS to ZAMS or with
        increasing Xc.
        Defaults to False

    ------- Returns -------
    column: numpy.array of given dtypes
        The selected history column, for those rows with Xc between Xc_min and
        Xc_max from small to large Xc
    """
    usecols = [lookup_header[column_name] for column_name in column_names]
    if (Xc_min is not None) or (Xc_max is not None) :
        Xcs = np.flip(np.loadtxt(history_file, skiprows=6,
                                 usecols=[lookup_header["center_h1"]],
                                 dtype=np.float32))
        if Xc_min is None :
            Xc_min = np.min(Xcs)
        if Xc_max is None :
            Xc_max = np.max(Xcs)
        imin, imax = np.searchsorted(Xcs, (Xc_min, Xc_max))
        # Bear in mind that the Xcs are flipped, so imax takes the role of where
        # to start reading the file and imin defines the endpoint!
        columns = np.loadtxt(history_file, skiprows=max(5+len(Xcs)-imax, 6),
                             max_rows=len(Xcs)-imin+1, usecols=usecols,
                             unpack=True)
    else :
        columns = np.loadtxt(history_file, skiprows=6, usecols=usecols,
                             unpack=True)

    # If requested, flipping orders such that it is ordered from smallest to
    # largest Xc
    if flip :
        return np.flip(columns, axis=1)
    else :
        return columns


def read_MESA_Xc_initial(history_file, lookup_header) :
    """
    Read the central hydrogen mass fraction of the first model in the given MESA 
    history file

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    history_file: string
        Path to the history file
    lookup_header: dict
        A table to find the column number matching each history column name.
        Keys are strings and values are integers. 

    ------- Returns -------
    Xcini: float
        Initial central hydrogen mass fraction at ZAMS
    """
    Xcini = np.loadtxt(history_file, skiprows=6, max_rows=1, 
                       usecols=[lookup_header["center_h1"]])
    return float(Xcini)


def read_model_parameters(parameter_file, header=True, delimiter=",") :
    """
    Read the file describing the parameters used in all the MESA models,
    specifically the initial metallicities Z, initial masses Mini, overshooting
    parameters fov and logarithm of the mixing strength at the base of the
    envelope logD. The rotation rates are read in elsewhere from the StORM
    summaries. Note that the parameter grid need not be rectangular.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    parameter_file: string
        Path to the .csv file containing all the grid parameters. This file is
        assumed to have a header like the one in the provided example input
        files.
    header: bool
        Whether this file has a header. If not, the order of the parameters is
        assumed to be Zini, Mini, logD, fov.
        Defaults to True
    delimiter: string
        The character(s) used to separate the values in the file.
        Defaults to ","

    ------- Returns -------
    Zinis: numpy.array of floats
        Initial metallicity of each MESA-GYRE run
    Minis: numpy.array of floats
        Initial mass in solar masses of each MESA-GYRE run
    logDs: numpy.array of floats
        Base 10 logarithm of the mixing strength at the base of the radiative
        envelope of each MESA-GYRE run
    fovs: numpy.array of floats
        Overshooting parameter of each MESA-GYRE run
    Each of these arrays is off the same length, namely the number of lines in
    the parameter file (minus header)
    """
    if header:
        header = np.loadtxt(parameter_file, max_rows=1, dtype=str,
                            delimiter=delimiter)
        parameters = np.loadtxt(parameter_file, skiprows=1, dtype=float,
                                unpack=True, delimiter=delimiter, ndmin=2)
        try :
            Zinis = parameters[(header == "Z") | (header == "Zini")][0]
        except :
            raise KeyError("read_model_parameters: parameter file does not " \
                           "have a header or header does not contain one of " \
                           "'Z' or 'Zini' as expected.")
        try :
            Minis = parameters[(header == "M") | (header == "Mini")][0]
        except :
            raise KeyError("read_model_parameters: parameter file does not " \
                           "have a header or header does not contain one of " \
                           "'M' or 'Mini' as expected.")
        try :
            fovs = parameters[(header == "fov") | (header == "f_ov")][0]
        except :
            raise KeyError("read_model_parameters: parameter file does not " \
                           "have a header or header does not contain one of " \
                           "'fov' or 'f_ov' as expected.")
        try :
            logDs = parameters[(header == "log_D") | (header == "logD") | \
                               (header == "logDmix") | (header == "log_Dmix") | \
                               (header == "log_D_mix")][0]
        except :
            raise KeyError("read_model_parameters: parameter file does not " \
                           "have a header or header does not contain one of " \
                           "'logD', 'log_D', 'logDmix', 'log_Dmix' or " \
                           "'log_D_mix' as expected.")
    else :
        Zinis, Minis, logDs, fovs = np.loadtxt(parameter_file, unpack=True,
                                               delimiter=delimiter, ndmin=2)
    return Zinis, Minis, logDs, fovs


def read_storm_tracks(MESA_output_dir, Zini, Mini, logD, fov, 
                      frots_rel_to_include=[], Xc_min=None, Xc_max=None,
                      ls=[0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2],
                      n_pgs=[1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3],
                      add_beta=False) :
    """
    Read in and summarise all the STORM output of a MESA run with the given
    parameters. This collects the predicted frequencies of the given
    combinations of degree l and radial order n_pgs at every output step. All
    output is collected into a dictionary, described below.

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    MESA_output_dir: string
        The full or relative path to the top folder containing all the output of
        MESA and StORM.
    Zini: float
        The initial metallicity of the MESA run to be read in.
    Mini: float
        The initial mass of the MESA run to be read in.
    logD: float
        The mixing strength at the base of the envelope of the MESA run to be
        read in.
    fov: float
        The overshooting parameter of the MESA run to be read in.
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
    ls: list of ints
        Degrees of the predicted modes to be read in
        Must be the same length as n_pgs
    n_pgs: list of ints
        Radial orders, within the Eckart-Scuflaire-Osaki-Takata scheme, of the
        predicted modes to be read in
        Must be the same length as ls
    add_beta: bool
        Whether to include the beta_nl constant from GYRE for each multiplet in
        the model
        Setting this to True increases loading time considerably
        Defaults to False

    ------- Returns -------
    all_tracks: list of dicts containing
        "ages": numpy.array of floats
            The age in years of the saved models
        "Xcs": numpy.array of floats
            The actual central hydrogen fractions of the saved models
            array length equals M
        "Xc_saves": numpy.array of floats
            The central hydrogen fractions around which MESA saved models
            array length equals M
        "Xc_min": float
            The minimum allowed central hydrogen fraction. Any MESA-StORM output
            below this limit is ignored
        "Xc_max": float
            The maximum allowed central hydrogen fraction. Any MESA-StORM output
            above this limit is ignored
        "mns": numpy.array of ints
            Model numbers at which MESA saved
            array length equals M
        "log_Teffs": numpy.array of floats
            base-10 logarithms of effective temperature in K
            array length equals M
        "log_Ls": numpy.array of floats
            base-10 logarithms of luminosity in solar luminosities
            array length equals M
        "frots": numpy.array of floats
            the rotation rates in cycles per day
            array length equals F
        "frots_rel": numpy.array of floats
            the rotation rates in relative fraction of the Keplerian critical
            rotation rate
            array length equals F
        "ls": numpy.array of ints
            Degrees of the modes
            array length equals K
        "ms": numpy.array of ints
            Degrees of the modes
            array length equals K
        "n_pgs": numpy.array of ints
            Radial orders of the modes
            array length equals K
        "freqs": numpy.array of numpy.arrays of numpy.array of floats
            Contains an array for each mode identity (l,m,n_pgs), rotation rate,
            and output step.
            array dimensions equals (K, F, M)
        "track_reliable": numpy.array of numpy.arrays of bools
            Whether the summarising algorithm did not detect any discrepancies
            with this each track
            array dimensions equals (K, M)
        "f_Ks": numpy.array of floats
            The Keplerian critical rotation rate of each model in cycles per day
            array length equals M
        "MESA_model_dir": string
            Path to the directory containing the output of this particular MESA
            run
        "histfile": string
            Path to the history file of this MESA model
        "lookup_histfile": dict
            A lookup dictionary to find the column number matching the header 
            name of a column in the history file
            Keys are strings, values are keys
        "betas": numpy array of floats
            The beta_nl of each mode
            Only included if add_beta is True
    Xcini: float
        The initial central hydrogen mass fraction at ZAMS
    with K the number of mode identities read in and M the number of output
    steps of the model and F the number of considered rotation rates.
    Each array over time (those of length M) is sorted by Xc
    """
    assert len(ls) == len(n_pgs)

    ls_track = []
    ms_track = []
    n_pgs_track = []
    for l, n_pg in zip(ls, n_pgs) :
        for m in range(-l, l+1) :
            ls_track.append(l)
            ms_track.append(m)
            n_pgs_track.append(n_pg)

    # Setting a number of paths
    folder = "Z{Z:.3f}_M{M:.2f}_logD{logD:.1f}_fov{fov:.3f}".format(Z=Zini,M=Mini,logD=logD,fov=fov)
    MESA_model_dir = "{MESA_output_dir}/{folder}/".format(MESA_output_dir=MESA_output_dir,folder=folder)
    history_file = "{MESA_model_dir}/{folder}.hist".format(MESA_model_dir=MESA_model_dir,folder=folder)
    # Building the lookup dictionary for the column names of the history file
    history_columnNames = np.loadtxt(history_file, skiprows=5, max_rows=1, dtype=str)
    lookup_histfile = {
        history_columnNames[i]: i
        for i in range(len(history_columnNames))
    }

    Xcini = read_MESA_Xc_initial(history_file, lookup_histfile)
    mn_hist,Xc_hist,age_hist,log_Teff_hist,log_L_hist = read_MESA_history_columns(history_file, lookup_histfile, 
                                                                                  ["model_number","center_h1","star_age","log_Teff","log_L"],
                                                                                  Xc_min=Xc_min,
                                                                                  Xc_max=Xc_max)
    mn_lookup = {
        mn: (age, Xc, logT, logL)
        for mn, age, Xc, logT, logL in zip(mn_hist, age_hist, Xc_hist, log_Teff_hist, log_L_hist)
    }

    storm_summary_paths = glob.glob("{MESA_model_dir}/storm_summaries/*.hdf5".format(MESA_model_dir=MESA_model_dir))
    # Ignore entries that lie outside the requested range of central
    # hydrogen fraction. Note that entries equal to the limits are allowed
    mask = []
    for storm_summary_path in storm_summary_paths :
        # Get the model number and central hydrogen fraction at which MESA was meant to save
        mn, Xc_save = read_model_number_and_Xc_save(storm_summary_path)
        mask.append((Xc_min is None or Xc_save > Xc_min) and (Xc_max is None or Xc_save < Xc_max))
    storm_summary_paths = [x for x, keep in zip(storm_summary_paths, mask) if keep]

    K = len(ls_track) # Number of predicted tracks
    M = len(storm_summary_paths) # Number of summaries
    F = len(frots_rel_to_include) # Number of considered relative rotation rates

    # Read the number of rotation rates from one file, assuming they all 
    # use the same frots_rel. 
    if F == 0 :
        with h5py.File(storm_summary_paths[0], "r") as storm_summary :
            frots_rel = np.array(storm_summary["frots_rel"])
            F = len(frots_rel)
    else :
        frots_rel = np.array(frots_rel_to_include)

    lookup_ID = build_lookup_ID(ls_track, ms_track, n_pgs_track)
    lookup_frot = {
        frots_rel[i]: i
        for i in range(len(frots_rel))
    }

    all_tracks = {"ages":np.empty(M), "mns":np.empty(M),
                  "log_Teffs":np.empty(M), "log_Ls":np.empty(M),
                  "Xcs":np.empty(M), "Xc_saves":np.empty(M),
                  "Xc_min":Xc_min, "Xc_max":Xc_max,
                  "MESA_model_dir":MESA_model_dir,
                  "histfile":history_file,
                  "lookup_histfile":lookup_histfile,
                  "ms":np.array(ms_track),
                  "ls":np.array(ls_track),
                  "n_pgs":np.array(n_pgs_track),
                  "lookup_ID":lookup_ID,
                  "freqs":np.empty((K,F,M)),
                  "track_reliable":np.empty((K,M)),
                  "f_Ks":np.empty(M),
                  "frots_rel":frots_rel, 
                  "lookup_frot":lookup_frot
                  }
    if add_beta :
        all_tracks["betas"] = np.empty((K,M))

    tracks = [
        process_summary_file(
            path,
            ls_track,
            ms_track,
            n_pgs_track,
            frots_rel_to_include,
            mn_lookup,
            add_beta,
            MESA_model_dir
        )
        for path in storm_summary_paths
    ]

    for j, track in enumerate(tracks):
        all_tracks["ages"][j] = track["age"]
        all_tracks["mns"][j] = track["mn"]
        all_tracks["log_Teffs"][j] = track["log_Teff"]
        all_tracks["log_Ls"][j] = track["log_L"]
        all_tracks["Xcs"][j] = track["Xc"]
        all_tracks["Xc_saves"][j] = track["Xc_save"]
        all_tracks["freqs"][:,:,j] = track["freqs"]
        all_tracks["track_reliable"][:,j] = track["track_reliable"]
        all_tracks["f_Ks"][j] = track["f_K"]
        if add_beta :
            all_tracks["betas"][:,j] = track["betas"]

    # for j,storm_summary_path in enumerate(storm_summary_paths) :
    # Turn all time independent lists into Numpy arrays and sort by age
    I_sort = np.argsort(all_tracks["ages"])
    all_tracks["ages"] = all_tracks["ages"][I_sort]
    all_tracks["Xcs"] = all_tracks["Xcs"][I_sort]
    all_tracks["Xc_saves"] = all_tracks["Xc_saves"][I_sort]
    all_tracks["mns"] = all_tracks["mns"][I_sort]
    all_tracks["log_Teffs"] = all_tracks["log_Teffs"][I_sort]
    all_tracks["log_Ls"] = all_tracks["log_Ls"][I_sort]
    all_tracks["f_Ks"] = all_tracks["f_Ks"][I_sort]
    all_tracks["freqs"][:,:] = all_tracks["freqs"][:,:,I_sort]
    all_tracks["track_reliable"][:] = all_tracks["track_reliable"][:,I_sort]
    if add_beta :
        all_tracks["betas"][:] = all_tracks["betas"][:,I_sort]

    return (all_tracks, Xcini)


def read_model_number_and_Xc_save(storm_summary_path) :
    """
    Get the model number and central hydrogen fraction at which MESA should have
    saved from the name of the STORM summary file

    Author: Mathijs Vanrespaille

    ------- Parameters -------
    storm_summary_path: string
        The full or relative path to the storm summary file.
        This file's name is expected to be formatted as
        "storm_summary_Xc{Xc}_mn{mn}.hdf5" with Xc the central hydrogen
        fraction, mn the model number and frot the rotation rate in c/d.

    ------- Returns -------
    mn: int
        Model number
    Xc: float
        Central hydrogen fraction around which MESA saved
    """
    filename = os.path.basename(storm_summary_path)
    Xc = float(substring(filename, "c", "_mn"))
    mn = int(substring(filename, "_mn", ".hdf5"))
    return mn, Xc


##### OUTPUT ###################################################################


def write_model(model, save_name, save_dir,
                attrs=["n_pg_main", "age", "Xc", "frot_best", "Xc_min",
                       "Xc_max", "log_Teff", "log_L", "M", "MESA_model_dir",
                       "histfile"],
                dsets=["ls", "ms", "n_pgs", "freqs"]) :
    """
    Write a .hdf file of the given model with the given name in the given
    location.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    model: dict
        The dictionary describing the particular model as created by
        interpolate_track_to_model
    save_name: string
        Name of the file to write. Will be appended with .hdf
        Note: Overwrites a file with the same
    save_dir: string
        Path to the directory where to write the file. Directory will be created
        if it does not exist yet.
    attrs: list of string
        The keys in the model dictionary to be writen out as attributes. Each
        key must be in model and contain a single value.
        Defaults to ["n_pg_main", "age", "Xc", "frot_best", "Xc_min", "Xc_max",
                     "log_Teff", "log_L", "M", "MESA_model_dir", "histifle"]
    dsets: list of string
        The keys in the model dictionary to be writen out as datasets. Each key
        must be in model and contain a list of value.
        Defaults to ["ls", "ms", "n_pgs", "freqs"]
    """
    if (type(model) == float) :
        print("write_model :  provided model is a NaN!")
        return

    if not os.path.isdir(save_dir) :
        print("Making new directory to write_model: ", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    with h5py.File(f"{save_dir}/{save_name}.hdf", "w") as g :
        for attr in attrs :
            g.attrs[attr] = model[attr]
        for dset in dsets :
            g.create_dataset(dset, data=model[dset])


def write_results(grid, observation, results, statistical_model, save_dir) :
    """
    Write all the output. For further details of what is included in these
    files, examine the accompanying examples or the three functions
    write_statistical_model, write_results_summary, and write_model

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    observation: dict
        The dictionary describing all the observations of a star as defined in
        create_observation_dict
    results: dict
        The dictionary containing all the results to be written out.
        Expected to contain at least "Xcs_cross", "ages_cross", "frots_best",
        "chi2s", "log_Rs", "log_gs", "log_Rs" and "log_Rs"
    statistical_model: dict
        The exp(-chi^2/2) averaged model of all optimal models within the
        2-sigma error ellipse on the HRD as created by make_statistical_model
    save_dir: string
        Path to the directory where to write the file
    """
    if not os.path.isdir(save_dir) :
        os.makedirs(save_dir, exist_ok=True)
    write_statistical_model(statistical_model,
                            f"{observation['name']}_statisticalModel_{observation['save_suffix']}",
                            save_dir)
    write_results_summary(grid, results,
                          f"{observation['name']}_resultsSummarised_{observation['save_suffix']}",
                          save_dir)
    write_model(results["best_model"],
                f"{observation['name']}_bestModel_{observation['save_suffix']}",
                save_dir)
    return


def write_results_summary(grid, results, save_name, save_dir,
                          exclude_splitting_from_chi2=False) :
    """
    Write .csv file containing the parameters for each evolutionary track's
    optimal model. Users are encouraged to expand these according to their
    needs.

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    grid: dict
        The dictionary summarising all the relevant MESA-StORM(-GYRE) grid
        output as created by initialise_grid
    results: dict
        The dictionary containing all the results to be written out.
        Expected to contain at least "Xcs_cross", "ages_cross", "frots_best",
        "chi2s", "log_Rs", "log_gs", "log_Rs" and "log_Rs"
    save_name: string
        Name of the file to write. Will be appended with .csv
        Note: Overwrites a file with the same
    save_dir: string
        Path to the directory where to write the file
        Note: Make sure this path already exists!
    exclude_splitting_from_chi2: bool
        Whether to not include rotational splitting terms in the chi2 sum
        Defaults to False
    """
    G = 6.67e-11 # in SI
    Msol = 1.989e30 # in kg
    Rsol = 6.957e8 # in m
    day = 86400 # in s
    models = results["models"]
    header = "chi2,chi2_reduced,spectro_compatible,Z,Mini,logD,fov,Xc,Xc_prime," \
             "age,log_age,frot,frot_rel,fK,log_Teff,log_L,M,R,log_R,vrot,log_g," \
             "Mcc,Mcc_over_M,Pi0"
    arrays = [results["chi2s"], results["chi2s_reduced"],
              results["spectro_compatible"], grid["Zinis"], grid["Minis"],
              grid["logDs"], grid["fovs"], grid["Xcinis"], 
              results["Xcs_cross"], results["ages_cross"], 
              results["frots_best"], results["log_Rs"],
              results["log_gs"], results["Ms_cc"], results["Pi0s"]]
    if exclude_splitting_from_chi2 :
        header += ",chi2_withSplitting,chi2_withSplitting_reduced"
    all_lines = [header]
    for i,(chi2,chi2_reduced,spectro_compatible,Zini,Mini,logD,fov,Xcini,Xc,age,
           frot,log_R,log_g,Mcc,Pi0) in enumerate(np.array(arrays).T) :
        model = models[i]
        log_Teff, log_L, M = model["log_Teff"], model["log_L"], model["M"]
        R = 10**log_R
        f_K = np.sqrt(G*M*Msol/(R*Rsol)**3) * day
        new_line = f"{chi2},{chi2_reduced},{spectro_compatible},{Zini},{Mini}," \
                   f"{logD},{fov},{Xc},{Xc/Xcini},{age},{np.log10(age)},{frot}," \
                   f"{frot/f_K},{f_K},{log_Teff},{log_L},{M},{R},{log_R}," \
                   f"{2*np.pi*frot/day*10**log_R*Rsol/1e3},{log_g},{Mcc}," \
                   f"{Mcc/M},{Pi0}"
        if exclude_splitting_from_chi2 :
            new_line += f",{results['chi2_withSplitting'][i]}," \
                        f"{results['chi2_withSplitting_reduced'][i]}"
        all_lines.append(f"\n{new_line}")

    if not os.path.isdir(save_dir) :
        print("Making new directory to write results summary: ", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    f = open(f"{save_dir}/{save_name}.csv", "w")
    f.writelines(all_lines)
    f.close()


def write_statistical_model(statistical_model, save_name, save_dir) :
    """
    Write .csv file of all the information in the given statistical model. This
    simply writes out all the values in the statistcal model dictinoary under
    columns named per the keys

    Author: Mathijs Vanrespaille

    ------- Parameters -------

    statistical_model:
        The exp(-chi^2/2) averaged model of all optimal models within the
        2-sigma error ellipse on the HRD as created by make_statistical_model
    save_name: string
        Name of the file to write. Will be appended with .csv
        Note: Overwrites a file with the same
    save_dir: string
        Path to the directory where to write the file
        Note: Make sure this path already exists!

    """
    all_lines = []
    for key in np.sort(list(statistical_model.keys())) :
        if type(statistical_model[key]) == str :
            all_lines.append(f'{key},"{statistical_model[key]}"\n')
        else :
            all_lines.append(f"{key},{statistical_model[key]}\n")

    if not os.path.isdir(save_dir) :
        print("Making new directory to write statistical model: ", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    f = open(f"{save_dir}/{save_name}.csv", "w")
    f.writelines(all_lines)
    f.close()
