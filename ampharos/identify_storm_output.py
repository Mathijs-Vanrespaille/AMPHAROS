import h5py
import numpy as np
import glob
import scipy.interpolate as scint
import itertools
import sys
import copy


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


import scipy.optimize as scopt
def linear(x, a, b) :
    return a*x + b


def main() :
    # Where to store the summaries this code produces
    storm_summary_dir = sys.argv[1] #f"{MESA_top_dir}/{folder}/storm/summaries/"
    # Where to find all the StORM output we have to summarise
    storm_output_dir = sys.argv[2] # f"{MESA_top_dir}/{folder}/storm/output/"

    # SETTINGS
    fmin_min = 2. # Smallest possible lower bound of the frequency range
    fmax = 15.
    ftol_normal = 0.05 # tolerance on the difference between predicted and deformed frequencies in c/d of existing track
    ftol_recent = 0.1 # tolerance on the difference between predicted and deformed frequencies in c/d of track with not enough points for full extrapolation yet
    ftol_new = 0.25 # tolerance on the difference between predicted and deformed frequencies in c/d of fully new track
    ftol_track = 0.50 # tolerance on the difference between freq and the edges of the frequency range wherein we search for new or lost freqs
    Nstart = 5
    markers = {0:"o", 1:"^", 2:"p"}
    # MESA_top_dir = "/home/mathijsv/forward_modelling/MESA_output_fullGrid"
    # folder = "Z0.014_M10.23_logD3.0_fov0.020"

    f_rot_min = 0.00
    f_rot_max = 0.405 # A little bit larger than the actual max to ensure numpy.arange doesn't cut off the maximum
    f_rot_step = 0.01
    f_rots = np.arange(0, 0.405, f_rot_step) # Make sure these are sorted from smallest to largest, starting with 0

    files = glob.glob(f"{storm_output_dir}/*_frot0.01.HDF")
    Xcs = [substring(file, "c", "_mn") for file in files]
    mns = [substring(file, "_mn", "_frot") for file in files]

    # start = time.perf_counter()

    for Xc, mn in zip(Xcs, mns) :
        try :
            # Preparing variables and lists to hold output
            ms_summary = []
            ls_summary = []
            n_pgs_summary = []
            versions_summary = []
            freqs_deformed_summary = []
            reliable_summary = []
            track_reliable_summary = []

            for m in [-2,-1,0,1,2] :

                # print("Xc = ", Xc, "mn = ", mn, "m = ", m)
                ###### PHASE 0 #################################

                # Key : three-element tuple (l, n_pg, version)
                # Value : a list of deformed frequencies at the rotation rates in f_rots. NaN if this ID is not included at that rotation frequency
                freq_tracks = {}

                # Key : three-element tuple (l, n_pg, version)
                # Value : list of booleans describing whether anything suspect occured to the track,
                # typically meaning its matched deformed frequency fell outside the tolerance of a prediction
                reliable = {}

                # DEPRECATED
                # Key : two-element tuple (l, n_pg)
                # Value : boolean describing whether this ID has already shown up at this f_rot
                seen = {}

                # Key : two-element tuple (l, n_pg)
                # Value : int describing how many versions of this ID are around
                versions = {}

                # Key : various strings, namely "fmin", "ls", "n_pgs", "versions", "freqs_deformed", "reliable", "all_reliable", "track_reliable"
                # Value : all the output I should need from this analysis for my forward modelling,
                #         namely the deformed frequencies, their IDs and whether I can trust them
                # summary = {}


                ###### PHASE 1 #################################


                f_rot = 0.00
                storm_output = h5py.File(f"{storm_output_dir}/storm_Xc{Xc}_mn{mn}_frot{f_rot:.2f}.HDF".format(f_rot=f_rot), "r")
                ms = np.array(storm_output["azimuthal-order"])
                freqs = np.array(storm_output["frequency"])[np.where(ms == m)]
                n_pgs = np.array(storm_output["radial-order"])[np.where(ms == m)]
                ls = np.array(storm_output["degree"])[np.where(ms == m)]
                # Sorting everything by original frequency
                K = np.argsort(freqs)
                freqs = freqs[K]
                n_pgs = n_pgs[K]
                ls = ls[K]

                # Dynamic frequency
                f_K = storm_output["model"].attrs["dynamical-frequency"]*86400/(2*np.pi)

                # Grabbing, sorting deformed frequencies and cutting out any frequency outside the frequency range of 2 to 15 c/d
                freqs_deformed = np.array([thing[0] for thing in storm_output["deformation"][str(m)]["frequency"]])
                freqs_deformed = np.sort(freqs_deformed)

                # Adding each frequency to its corresponding track and counting repetitions of IDs
                for l, n_pg, freq in zip(ls, n_pgs, freqs_deformed) :
                    if (l, n_pg) not in versions.keys() :
                        versions[(l,n_pg)] = 1
                    else :
                        versions[(l, n_pg)] += 1
                    ID = (l, n_pg, versions[l,n_pg])
                    freq_tracks[ID] = [freq]
                    reliable[ID] = [True] # Track is good until we spot some issue


                ###### PHASE 2 #################################


                for i in range(1,Nstart) : # index over the f_rots
                    for ID in seen.keys() :
                        seen[ID] = 0
                    f_rot = f_rots[i]
                    fmin = max(fmin_min, m*f_rot*f_K*1.01)
                    reliable_this_step = {}

                    # Reading the undeformed frequencies and their identities to match the deformed frequencies to
                    storm_output = h5py.File(f"{storm_output_dir}/storm_Xc{Xc}_mn{mn}_frot{f_rot:.2f}.HDF".format(f_rot=f_rot), "r")
                    ms = np.array(storm_output["azimuthal-order"])
                    freqs = np.array(storm_output["frequency"])[np.where(ms == m)]
                    n_pgs = np.array(storm_output["radial-order"])[np.where(ms == m)]
                    ls = np.array(storm_output["degree"])[np.where(ms == m)]
                    # Sorting everything by original frequency
                    K = np.argsort(freqs)
                    freqs = freqs[K]
                    n_pgs = n_pgs[K]
                    ls = ls[K]

                    # Grabbing, sorting deformed frequencies and cutting out any frequency outside the frequency range of 2 to 15 c/d
                    freqs_deformed = np.array([thing[0] for thing in storm_output["deformation"][str(m)]["frequency"]])
                    freqs_deformed = np.sort(freqs_deformed)

                    # freqs_deformed can inlcude extra signals at low frequencies. Only keep as many as there are original frequencies
                    freqs_deformed = freqs_deformed[-len(freqs):]

                    for l, n_pg, freq, freq_deformed in zip(ls, n_pgs, freqs, freqs_deformed) :
                        # expected frequency in the previous f_rot step
                        f_exp = freq_deformed - m*f_rot_step*f_K
                        f_diff = 99999999.
                        for key in freq_tracks.keys() :
                            if key[0] == l :
                                if not np.isnan(freq_tracks[key][i-1]) and abs(freq_tracks[key][i-1] - f_exp) < f_diff :
                                    f_diff = abs(freq_tracks[key][i-1] - f_exp)
                                    ID_nearest = key
                        # Only accept the matching frequency if it is within the tolerance
                        if f_diff < ftol_normal :
                            ID = ID_nearest
                        # If it doesn't match an existing frequency track and is near the boundaries, it is presumably
                        # a new frequency entering our frequency range
                        elif freq < fmin+ftol_normal or freq > fmax-ftol_normal :
                            if (l,n_pg) not in versions.keys() :
                                versions[(l,n_pg)] = 0
                            ID = (l, n_pg, versions[(l,n_pg)]+1)
                            freq_tracks[ID] = [np.nan for j in range(i)]
                            reliable[ID] = [True for j in range(i)]
                            reliable_this_step[ID] = True
                            versions[(l,n_pg)] += 1
        #                     print(f"New ID near boundary : {ID} at f_rot = {f_rot}")
                        # Otherwise, we stick with into a new ID with the version number increased and throw a warning
                        # as this is not expected to just happen in the middle of the frequency ranges
                        else :
                            if (l,n_pg) not in versions.keys() :
                                versions[(l,n_pg)] = 0
                            ID = (l, n_pg, versions[(l,n_pg)]+1)
                            freq_tracks[ID] = [np.nan for j in range(i)]
                            reliable[ID] = [False for j in range(i)]
                            reliable_this_step[ID] = False
                            versions[(l,n_pg)] += 1
        #                     warnings.warn(f"WARNING unexpected new ID {ID} at f_rot = {f_rot}")

                        freq_tracks[ID].append(freq_deformed)

                    # If an ID did not show up, its length will be one smaller than the rest, and we append an np.nan
                    # to indicate to future steps it did not show up
                    for key in freq_tracks.keys() :
                        if len(freq_tracks[key]) < i+1 :
        #                     print(f"ID gone missing : {ID} at f_rot = {f_rot}")
                            freq_tracks[key].append(np.nan)

                    # Updating reliability tracks
                    for ID in freq_tracks.keys() :
                        if ID in reliable_this_step.keys() :
                            reliable[ID].append(reliable_this_step[ID])
                        else :
                            # If this step did not report anything notable, repeat the previous reliability
                            reliable[ID].append(reliable[ID][-1])


                ###### PHASE 3 #################################


                for i in range(Nstart, len(f_rots)) : # index over the f_rots
                    f_rot = f_rots[i]
                    fmin = max(fmin_min, m*f_rot*f_K*1.01)
                    reliable_this_step = {}


                    # Collecting the undeformed frequencies and their l and n_pg from the previous step
                    freqs_old = freqs
                    ls_old = ls
                    n_pgs_old = n_pgs

                    # Reading the undeformed frequencies and their identities to match the deformed frequencies to
                    storm_output = h5py.File(f"{storm_output_dir}/storm_Xc{Xc}_mn{mn}_frot{f_rot:.2f}.HDF".format(f_rot=f_rot), "r")
                    ms = np.array(storm_output["azimuthal-order"])
                    freqs = np.array(storm_output["frequency"])[np.where(ms == m)]
                    ls = np.array(storm_output["degree"])[np.where(ms == m)]
                    n_pgs = np.array(storm_output["radial-order"])[np.where(ms == m)]
                    # Sorting everything by original frequency
                    K = np.argsort(freqs)
                    freqs = freqs[K]
                    n_pgs = n_pgs[K]
                    ls = ls[K]

                    # Grabbing, sorting deformed frequencies and cutting out any frequency outside the frequency range of 2 to 15 c/d
                    freqs_deformed = np.array([thing[0] for thing in storm_output["deformation"][str(m)]["frequency"]])
                    freqs_deformed = np.sort(freqs_deformed)

                    # freqs_deformed can inlcude extra signals at low frequencies. Only keep as many as there are original frequencies
                    freqs_deformed = freqs_deformed[-len(freqs):]

                    freqs_predicted = {} # Normal tracks
                    freqs_recent_predicted = {} # tracks with not enough points for an extrapolation.
                    freqs_new_predicted = {} # new tracks

                    # Looking for new frequency tracks to take into account
                    # NOTE TO SELF
                    # So far I assumed that new frequencies popping into our range would always be the
                    # largest or smallest. So far, that's held true. HOWEVER, I am not confident that
                    # it is necessarily linked to the smallest deformed frequency. Therefore, setting
                    # the predicted frequency equal to the smallest deformed frequency is a dangerous
                    # idea. Instead, I intend to introduce an all new method of predicting the first
                    # mode, namely by using the similar slopes of all modes with the same degree.
                    # I first need to poke those a bit though, in particular to test if the slope
                    # is related to the (initial) frequency.
                    # So this idea doesn't look like it's going to work out as there is significant
                    # variation in the slope of the frequency tracks, especially at low frequencies.
                    # Instead, I intend to make these new frequencies lower priority targets. In effect,
                    # we first search for and match all the frequencies that already have an actual
                    # track. Then around the end we check if the remaining unclaimed frequency(ies)
                    # seem a decent fit to the new frequency with a major tolerance of ftol_new
                    # Our initial prediction for the frequency is generated as follows:
                    #   First we seek the difference between the freq and deformed freq of the
                    #     previous step for those tracks with the same degree that are non-NaN
                    #   Next, fit a linear to the relation freq to (freq_deformed-freq)
                    #   Evaluate that fit at the freq of this new mode
                    #   Add that evaluation to the freq as our initial estimate\
                    # Note that frequency tracks that leave and re-enter are also considered new and
                    # their previous track is erased
                    for l, n_pg, freq in zip(ls, n_pgs, freqs) :
                        if ((freq < fmin+ftol_track or freq > fmax-ftol_track) and (l,n_pg) not in versions.keys()) or \
                        ((l,n_pg) not in zip(ls_old,n_pgs_old) and (l,n_pg) in versions.keys()) :
        #                     if (l,n_pg) not in versions.keys() :
        #                         versions[(l,n_pg)] = 1
        #                         ID_new = (l, n_pg, 1)
        #                     else :
        #                         versions[(l,n_pg)] += 1
        #                         ID_new = (l, n_pg, versions[(l, n_pg)])
        #                     freq_tracks[ID] = [np.nan for j in range(i)]
                            #   First we seek the difference between the freq and deformed freq of the
                            #     previous step if that is not NaN
                            versions[(l,n_pg)] = 1
                            ID_new = (l, n_pg, 1)
                            freqs_l = []
                            freqs_deformed_l = []
                            for ID in freq_tracks.keys() :
                                if ID[0] != l or (l,ID[1]) not in zip(ls_old,n_pgs_old) or versions[(ID[0], ID[1])] != 1 \
                                or np.isnan(freq_tracks[ID][-1]) or not reliable[ID][-1] :
                                    continue
                                freqs_l.append(freqs_old[(ls_old==l) & (n_pgs_old==ID[1])][0])
                                freqs_deformed_l.append(freq_tracks[(l,ID[1],1)][-1])
                            freqs_l = np.array(freqs_l)
                            freqs_deformed_l = np.array(freqs_deformed_l)
                            if len(freqs_l) > 2 :
                                #   Next, fit a linear to the relation freq to (freq_deformed-freq)
                                popt, pcov = scopt.curve_fit(linear, freqs_l, freqs_deformed_l - freqs_l, p0=(0.,0.))
                                #   Evaluate that fit at the freq of this new mode
                                freq_diff_fit = linear(freq, popt[0], popt[1])
                            else :
                                freq_diff_fit = 0.
                            # The predicted frequency is the freq plus the predicted offset from the fit
                            freqs_new_predicted[ID_new] = freq + freq_diff_fit
        #                     print(f"New or returning ID near boundary : {ID_new} at f_rot = {f_rot}")

                    # Adding predictions for the next steps
                    for ID in freq_tracks.keys() :
                        if ID in freqs_predicted.keys() or ID in freqs_new_predicted.keys() :
                            continue
                        # if the last value is NaN, the ID seems to have left our frequency range.
                        # Just add another NaN.
                        if np.isnan(freq_tracks[ID][-1]) :
                            freqs_predicted[ID] = np.nan
                        # Check there are enough non-NaN values
                        elif (i - np.sum(np.isnan(freq_tracks[ID]))) < 3 :
                            # Not enough non-NaN values !
                            # Instead of doing an interpolation, simply predict the next value based on m and the rotation rate
                            # From the initial tests, it became clear that this approach is not precise enough.
                            # As an alternative, I am trying a prediction similar to new tracks above where we
                            # look at all other identical degree tracks and fit their slopes.
                            # The hope is this will improve the predictions enough to continue going as is,
                            # but if this is not so, I will have to treat these predicted frequencies as
                            # second rate predictions, similar to the new tracks
                            freqs_deformed_l = []
                            freqs_predicted_l = []
                            for ID_track in freqs_predicted.keys() :
                                if ID_track[0] != ID[0] or versions[(ID_track[0], ID_track[1])] != 1 \
                                or np.isnan(freq_tracks[ID_track][-1]) or not reliable[ID_track][-1] :
                                    continue
                                freqs_deformed_l.append(freq_tracks[ID_track][-1])
                                freqs_predicted_l.append(freqs_predicted[ID_track])
                            freqs_deformed_l = np.array(freqs_deformed_l)
                            freqs_predicted_l = np.array(freqs_predicted_l)
                            if len(freqs_deformed_l) > 2 :
                                #   Next, fit a linear to the relation freq to (freq_deformed-freq)
                                popt, pcov = scopt.curve_fit(linear, freqs_deformed_l, freqs_predicted_l-freqs_deformed_l, p0=(m*f_rot_step*f_K,0.))
                                #   Evaluate that fit at the freq of this new mode
                                freq_diff_fit = linear(freq_tracks[ID][-1], popt[0], popt[1])
                            else :
                                freq_diff_fit = m*f_rot_step*f_K
                            freqs_recent_predicted[ID] = freq_tracks[ID][-1] + freq_diff_fit
                        else :
                            L = ~np.isnan(freq_tracks[ID]) # indices of non-NaN frequency track entries
                            spline = scint.CubicSpline(f_rots[:i][L], np.array(freq_tracks[ID])[L], extrapolate=True)
                            freqs_predicted[ID] = spline(f_rot)

                    # Look for returning tracks, which I count as recents :


                    # Check if we have more predictions than deformations. If so, look for any
                    # tracks that may have left the frequency range
                    # Changing to always look for frequencies that have left the range for cases
                    # when one track leaves and a new one enters in the same step
                    # I now have an issue with
        #             if len(freqs_deformed) < np.sum(~np.isnan(list(freqs_predicted.values()))) :
                    removed = {} # (l, n_pg) as key, how many of these have already been killed off
                    for l,n_pg,freq in zip(ls_old, n_pgs_old, freqs_old) :
                        if not (freq < fmin+ftol_track or freq > fmax-ftol_track) :
                            continue
                        N_ID_match = np.sum((ls==l) & (n_pgs==n_pg))
                        N_ID_match_old = np.sum((ls_old==l) & (n_pgs_old==n_pg))
                        # A frequency ID has gone missing!
                        if N_ID_match < N_ID_match_old :
                            if (l, n_pg) not in removed.keys() :
                                removed[(l, n_pg)] = 0
                            if (N_ID_match_old - N_ID_match) <= removed[(l, n_pg)] :
                                continue
        #                     print(f"A track has gone missing: l = {l}, n_pg = {n_pg} at f_rot = {f_rot}")
        #                         freqs_match = freqs_old[(ls==l) & (n_pgs==n_pg)]

                            # Originally, I killed off whatever track had the larger or smaller freq
                            # based on the azimuthal order m, expecting that tracks fly off the edges
                            # HOWEVER, when the
                            if m < 0 :
                                # Go through the predicted IDs and set the one with the smallest freq to NaN
                                ID_match = []
                                freqs_match = []
                                for ID in list(freqs_predicted.keys()) + list(freqs_recent_predicted.keys()) :
                                    if ID[0]==l and ID[1]==n_pg and \
                                    ((ID in freqs_predicted.keys() and not np.isnan(freqs_predicted[ID])) or \
                                     (ID in freqs_recent_predicted.keys() and not np.isnan(freqs_recent_predicted[ID]))):
                                        ID_match.append(ID)
                                        if ID in freqs_recent_predicted.keys() :
                                            freqs_match.append(freqs_recent_predicted[ID])
                                        elif ID in freqs_predicted.keys() :
                                            freqs_match.append(freqs_predicted[ID])
                                ID_toRemove = ID_match[np.argmin(freqs_match)]
                                if ID_toRemove in freqs_recent_predicted.keys() :
                                    freqs_recent_predicted[ID_toRemove] = np.nan
                                else :
                                    freqs_predicted[ID_toRemove] = np.nan
                            elif m > 0 :
                                # Go through the predicted IDs and set the one with the largest freq to NaN
                                ID_match = []
                                freqs_match = []
                                for ID in list(freqs_predicted.keys()) + list(freqs_recent_predicted.keys()) :
                                    if ID[0]==l and ID[1]==n_pg :
                                        ID_match.append(ID)
                                        if ID in freqs_recent_predicted.keys() :
                                            freqs_match.append(freqs_recent_predicted[ID])
                                        elif ID in freqs_predicted.keys() :
                                            freqs_match.append(freqs_predicted[ID])
                                ID_toRemove = ID_match[np.argmax(freqs_match)]
                                if ID_toRemove in freqs_recent_predicted.keys() :
                                    freqs_recent_predicted[ID_toRemove] = np.nan
                                else :
                                    freqs_predicted[ID_toRemove] = np.nan
                            removed[(l, n_pg)] += 1
                    # Match the predictions to the actual deformed frequencies and add to the tracks
                    # The index of the deformed frequency to which the predicted frequency with the ID
                    # as key is matched.
                    # STEP 1 : match each prediction to the closest deformed frequency
                    J_match = {}
                    for ID in freqs_predicted.keys() :
                        if np.isnan(freqs_predicted[ID]) :
                            continue
                        J_match[ID] = np.argmin(abs(freqs_predicted[ID]-freqs_deformed))

                    # STEP 2 : check for duplicates
                    flipped = {} # The IDs that were matched to index of the deformed frequency
                    for j in range(len(freqs_deformed)) :
                        flipped[j] = []
                    for ID,j in J_match.items():
                        flipped[j].append(ID)

                    # Go over the flipped dictionary and check the length of the list of IDs
                    # matched to each deformed frequency index
                    # We define three categories of deformed frequency indices j:
                    #     - unique_match: exactly one ID matched to this index
                    #     - no_match: no bitches, no matches. We need to find the prediction that should match this
                    #     - multiple_match: several IDs are matched to this index and need to be redistributed
                    no_match, unique_match, multiple_match = [], [], []
                    for j in range(len(freqs_deformed)) :
                        if len(flipped[j]) == 0 :
                            no_match.append(j)
                        elif len(flipped[j])==1 :
                            unique_match.append(j)
                        else :
                            multiple_match.append(j)
                    # First simple check: Is the double match at the edge and if so, is one of the predictions
                    # (almost) outside of the frequency range? If so and if the number of deformed frequencies
                    # is smaller than the number of predictions, that is a clear indication that the track
                    # has simply moved out of our range and should thus be set to nan from here on out.
                    for j in multiple_match :
                        for ID in flipped[j] :
                            if len(freqs_deformed) < np.sum(~np.isnan(list(freqs_predicted.values()))) and (freqs_predicted[ID] < fmin+ftol_normal or freqs_predicted[ID] > fmax-ftol_normal) :
                                freqs_predicted[ID] = np.nan
                                del J_match[ID]
                                flipped[j].remove(ID)
                                if len(flipped[j]) == 1 :
                                    multiple_match.remove(j)
                                    unique_match.append(j)
                                if len(flipped[j]) == 0 :
                                    multiple_match.remove(j)
                                    no_match.append(j)
                    # If not all matches are unique, we need to resolve the conflict
                    if len(multiple_match) > 0 or len(no_match) > np.sum(~np.isnan(list(freqs_new_predicted.values()))) + np.sum(~np.isnan(list(freqs_recent_predicted.values()))) :
                        # Foregoing the idea of checking separately at the boundaries. That should have been taken
                        # care of in the predictions in step 1.
                        # Instead, I only redistribute the IDs based on sorting the unclaimed deformed frequencies
                        # and predicted ones and matching them in order. I keep track of which
                        # matches were altered from their initial guess like this in case I ever want to backtrack
                        # and try the other option.

                        # Collect all unclaimed indices and the corresponding deformed frequencies
                        J_unclaimed = np.append(no_match, multiple_match).astype(int)
                        freqs_deformed_unclaimed = freqs_deformed[J_unclaimed]
                        # Collect all unmatched IDs and the corresponding predicted frequencies
                        IDs_unclaimed = []
                        for j in J_unclaimed :
                            for ID in flipped[j] :
                                IDs_unclaimed.append(ID)
                        freqs_predicted_unclaimed = np.array([freqs_predicted[ID] for ID in IDs_unclaimed])
                        # Sort both remaining freqs lists and IDs
                        J_unclaimed_sorted = J_unclaimed[np.argsort(freqs_deformed_unclaimed)]
                        freqs_deformed_unclaimed_sorted = np.sort(freqs_deformed_unclaimed)
                        H = np.argsort(freqs_predicted_unclaimed)
                        IDs_unclaimed_sorted = []
                        for h in H :
                            IDs_unclaimed_sorted.append(IDs_unclaimed[h])
                        freqs_predicted_unclaimed_sorted = np.sort(freqs_predicted_unclaimed)
                        assert len(freqs_deformed_unclaimed_sorted) ==  np.sum(~np.isnan(freqs_predicted_unclaimed_sorted)) + np.sum(~np.isnan(list(freqs_new_predicted.values()))) + np.sum(~np.isnan(list(freqs_recent_predicted.values())))
                        # Match both by minimising the chi2 distance
                        chi2_best = sys.float_info.max
                        # Initialising pd, a list linking the indices p of the sorted unclaimed predicted frequencies to
                        # the indices d of the sorted unclaimed deformed frequencies
                        Np = len(freqs_predicted_unclaimed)
                        Nd = len(freqs_deformed_unclaimed)
                        pd = [p for p in range(Np)]
                        while pd[0] <= Nd-Np :
                            chi2 = np.sum((freqs_predicted_unclaimed_sorted - freqs_deformed_unclaimed_sorted[pd])**2)
                            if chi2 < chi2_best :
                                chi2_best = chi2
                                pd_best = copy.deepcopy(pd)
                            p = -1 # Always keep this negative! Let it decrease to walk through pd from right to left
                            pd[p] += 1
                            while p > -Np :
                                if pd[p] > Nd + p :
                                    p -= 1
                                    pd[p] += 1
                                    for r,q in enumerate(range(Np+p, Np)) :
                                        pd[q] = pd[p] + r
                                else :
                                    break
                        # Check if they are within a reasonable tolerance. If they are, update :
                        #   J_match, flipped, no_match, unique_match and multiple_match.
                        # Add backtracking here?
                        # First clear the sorted out IDs from the original J_match and flipped
                        for ID in IDs_unclaimed_sorted :
                            j_old = J_match[ID]
                            del J_match[ID]
                            flipped[j_old].remove(ID)

                        diffs = freqs_deformed_unclaimed_sorted[pd_best] - freqs_predicted_unclaimed_sorted
                        for p, (ID, diffs) in enumerate(zip(IDs_unclaimed_sorted, diffs)) :
                            j = J_unclaimed_sorted[pd_best[p]]
                            J_match[ID] = j
                            flipped[j].append(ID)
                            # If not, shout WEEWOO
                            if not abs(diffs) < ftol_normal :
        #                         print(i)
        #                         print("WEEWOO WEEWOO!!!! Bad assignment for multiple matched existing track")
                                reliable_this_step[ID] = False

                        no_match, unique_match, multiple_match = [], [], []
                        for j in range(len(freqs_deformed)) :
                            if len(flipped[j]) == 0 :
                                no_match.append(j)
                            elif len(flipped[j])==1 :
                                unique_match.append(j)
                            else :
                                multiple_match.append(j)

                        assert len(multiple_match) == 0
                        assert len(no_match) == np.sum(~np.isnan(list(freqs_new_predicted.values()))) + np.sum(~np.isnan(list(freqs_recent_predicted.values())))

                    diffs = []
                    J_sorted = np.array(no_match)[np.argsort(freqs_deformed[no_match])] # Should be identical to just no_match, but I'm not taking any chances
                    freqs_new_predicted_values = list(freqs_new_predicted.values()) + list(freqs_recent_predicted.values())
                    IDs_new_predicted = list(freqs_new_predicted.keys()) + list(freqs_recent_predicted.keys())
                    freqs_new_predicted_noNaN = []
                    IDs_new_predicted_noNaN = []
                    for freq, ID in zip(freqs_new_predicted_values, IDs_new_predicted) :
                        if not np.isnan(freq) :
                            freqs_new_predicted_noNaN.append(freq)
                            IDs_new_predicted_noNaN.append(ID)
                    ftols = []
                    for ID in IDs_new_predicted_noNaN :
                        if ID in freqs_new_predicted.keys() :
                            ftols.append(ftol_new)
                        elif ID in freqs_recent_predicted.keys() :
                            ftols.append(ftol_recent)
                    Js = list(itertools.permutations(J_sorted))
                    chi2_best = 999999999.
                    for J in Js :
                        chi2 = 0
                        for k,j in enumerate(J) :
                            chi2 += ((freqs_deformed[j] - freqs_new_predicted_noNaN[k]) / ftols[k])**2
                        if chi2 < chi2_best :
                            chi2_best = chi2
                            J_best = copy.deepcopy(J)

                    for j, ID, freq, ftol in zip(J_best, IDs_new_predicted_noNaN, freqs_new_predicted_noNaN, ftols) :
                        if ID not in freq_tracks.keys() :
                            freq_tracks[ID] = [np.nan for j in range(i)]
                            reliable[ID] = [True for j in range(i)]
                        J_match[ID] = j
                        if not freq - freqs_deformed[j] <= ftol :
        #                     print(i)
        #                     print("WEEWOO WEEWOO!!!! Bad assignment to remaining frequencies for new track")
                            reliable_this_step[ID] = False

                    for ID in freq_tracks.keys() :
                        if ID in freqs_predicted.keys() and np.isnan(freqs_predicted[ID]) :
                            freq_tracks[ID].append(np.nan)
                        elif ID in freqs_recent_predicted.keys() and np.isnan(freqs_recent_predicted[ID]) :
                            freq_tracks[ID].append(np.nan)
                        else :
                            freq_tracks[ID].append(freqs_deformed[J_match[ID]])

                    # Check if the normal tracks are within their reasonable tolerances
                    for ID in freqs_predicted.keys() :
                        if np.isnan(freqs_predicted[ID]) or np.isnan(freq_tracks[ID][-1]) :
                            continue
                        if abs(freqs_predicted[ID] - freq_tracks[ID][-1]) > ftol_normal :
                            reliable_this_step[ID] = False

                    # Check whether the tracks with identical degree are sorted by radial order
                    # orders_per_degree is a dict with l as its key and lists of n_pgs as their values
                    # Similarly, freqs_per_degree iuses l as its key and the freqs of each n_pg as its values
                    # fullIDs_per_degree also uses l as its key and contains the full ID for admin purposes
                    orders_per_degree = {0:[], 1:[], 2:[]}
                    fullIDs_per_degree = {0:[], 1:[], 2:[]}
                    freqs_per_degree = {0:[], 1:[], 2:[]}
                    for ID in freq_tracks.keys() :
                        if np.isnan(freq_tracks[ID][-1]) :
                            continue
                        orders_per_degree[ID[0]].append(ID[1])
                        fullIDs_per_degree[ID[0]].append(ID)
                        freqs_per_degree[ID[0]].append(freq_tracks[ID][-1])
                    order_indices_per_degree = {}
                    freq_indices_per_degree = {}
                    for l in [0,1,2] :
                        orders_of_degree = orders_per_degree[l]
                        freqs_of_degree = freqs_per_degree[l]
                        Q = np.argsort(orders_of_degree)
                        R = np.argsort(freqs_of_degree)
                        for q,r in zip(Q,R) :
                            ID = fullIDs_per_degree[l][q]
                            if q != r :
                                reliable_this_step[ID] = False

                    # Updating reliability tracks
                    for ID in freq_tracks.keys() :
                        if ID in reliable_this_step.keys() :
                            reliable[ID].append(reliable_this_step[ID])
                        else :
                            # If this step did not report anything notable, repeat the previous reliability
                            reliable[ID].append(reliable[ID][-1])

                    # Verify whether every track has been updated
                    for ID in freq_tracks.keys() :
                        assert len(freq_tracks[ID]) == i+1
                        assert ID in reliable.keys()
                        assert len(reliable[ID]) == i+1

        #         fig, ax = plt.subplots(1,1,figsize=(16,5))

        #         for ID in freq_tracks.keys() :
        #             l, n_pg, version = ID
        #             freqs = freq_tracks[ID]
        #             colour = {True:"blue", False:"brown"}
        #             ax.plot(freqs, f_rots[:len(freqs)], marker=markers[l], color=colour[np.all(reliable[ID])], alpha=0.6, zorder=1)

        #         for f_rot in np.arange(0, 0.405, 0.01) :
        #             storm_output = h5py.File(f"{storm_output_dir}/storm_Xc{Xc}_mn{mn}_frot{f_rot:.2f}.HDF".format(f_rot=f_rot), "r")

        #             freqs_all = np.array(storm_output["frequency"])
        #             ms_all = np.array(storm_output["azimuthal-order"])
        #             freqs = freqs_all[np.where(ms_all == m)]

        #             # Grabbing, sorting deformed frequencies and cutting out any frequency outside the frequency range of 2 to 15 c/d
        #             freqs_deformed_plot = np.array([thing[0] for thing in storm_output["deformation"][str(m)]["frequency"]])
        #             freqs_deformed_plot = np.sort(freqs_deformed_plot)[-len(freqs):]
        #             ax.scatter(freqs_deformed_plot, np.ones(len(freqs_deformed_plot))*f_rot,
        #                        marker="|", color='red', alpha=0.6, zorder=2)
        #             ax.set_title(f"m = {m}  ;  Xc = {Xc}  ;  mn = {mn}")
        #         plt.show()
        #         plt.close()

                # Collecting the output for this azimuthal order and storing it in all_summaries

                for ID in freq_tracks.keys() :
                    ms_summary.append(m)
                    ls_summary.append(ID[0])
                    n_pgs_summary.append(ID[1])
                    versions_summary.append(ID[2])
                    freqs_deformed_summary.append(freq_tracks[ID])
                    reliable_summary.append(reliable[ID])
                    track_reliable_summary.append(np.all(reliable[ID]))

        #         current = time.perf_counter()
        #         print(f"TIME SO FAR : {current - start}")

            # Collecting various other output
            # all_summaries["f_K"] = f_K
            # all_summaries["frots"] = f_rots * f_K
            # all_summaries["frots_rel"] = f_rots
            # all_summaries["fmins"] = np.array([max(fmin_min, m*f_rot*f_K*1.01) for f_rot in f_rots])
            # all_summaries["ms"] = [-2,-1,0,1,2]


            # Writing all the output to a file
            filename = f"storm_summary_Xc{Xc}_mn{mn}.hdf5"
            with h5py.File(f"{storm_summary_dir}/{filename}", "w") as f:
                f.attrs["f_K"] = f_K
                f.create_dataset("frots", dtype="f", data=f_rots * f_K)
                f.create_dataset("frots_rel", dtype="f", data=f_rots)
        #         f.create_dataset("fmins", dtype="f", data=[max(fmin_min, m*f_rot*f_K*1.01) for f_rot in f_rots])
                f.create_dataset("ms", dtype="i", data=np.array(ms_summary))
                f.create_dataset("ls", dtype="i", data=np.array(ls_summary))
                f.create_dataset("n_pgs", dtype="i", data=np.array(n_pgs_summary))
                f.create_dataset("versions", dtype="i", data=np.array(versions_summary))
                f.create_dataset("track_reliable", dtype="b", data=np.array(track_reliable_summary))
                f.create_dataset("reliable", dtype="b", data=np.array(reliable_summary))
                f.create_dataset("freqs_deformed", dtype="f", data=np.array(freqs_deformed_summary))
                f.create_dataset("fmins", dtype="f", data=np.array(np.array([max(fmin_min, m*f_rot*f_K*1.01) for f_rot in f_rots])))
            # end = time.perf_counter()

            # print(f"TOTAL TIME : {end - start}")
            # print(f"Finished Xc = {Xc}, mn = {mn}")

        except Exception as error:
            print(f"Something went wrong at Xc = {Xc}, mn = {mn}, f_rot = {f_rot} : ", error)


if __name__ == "__main__":
    main()
