#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:52:45 2023

@author: dario
"""

import numpy as np
from scipy.interpolate import interp1d
import gdr3bcg.bcg as bcg
import mesa_reader as mr


def addIS(ax, dm=7.54):
    '''
    Add the instability strip borders to a observational CMD

    Parameters
    ----------
    ax : matplotlib axis
        The axis with the CMD you want the IS borders added to.
    dm : float, optional
        Distance modulus of the cluster. Set to 0 for absolute magnitude CMD.
        The default is 7.54.

    Returns
    -------
    None

    '''

    tab = bcg.BolometryTable()  # initiate

    # define functions

    def Mbol(logL):
        '''Get bolometric magnitude based on log(L/L_sun)'''
        M_bol_sun = 4.74
        return -2.5 * logL + M_bol_sun

    def BC(Teff):
        '''Get bolometric correction for Teff, assuming solar metallicity'''
        return [tab.computeBc([T, 4.0, 0.0, 0]) for T in Teff]

    def bp_rp(Teff):
        """Transform Teff to bp_rp baased on Mamajek's table"""
        bp_rp = [-0.087, -0.037, 0.005, 0.068, 0.11, 0.166, 0.194, 0.222,
                 0.263, 0.32, 0.327, 0.377, 0.434, 0.49, 0.518, 0.546, 0.587,
                 0.64]
        T = [10400, 9700, 9300, 8800, 8600, 8250, 8100, 7910, 7760, 7590, 7400,
             7220, 7020, 6820, 6750, 6670, 6550, 6350]

        func = interp1d(T, bp_rp)

        return func(Teff)

    # gDor Drupet (2005)

    Teff_up = np.array([7500, 7100, 6890, 6720])
    Teff_low = np.array([7100, 6780, 6620, 6590])

    Lum_up = np.array([0.775, 1.075, 1.25, 1.35])
    Lum_low = np.array([0.525, 0.75833, 1.2, 1.35])

    M_bol_up = Mbol(Lum_up)
    M_bol_low = Mbol(Lum_low)

    Mabs_up = M_bol_up - BC(Teff_up)
    Mabs_low = M_bol_low - BC(Teff_low)

    bp_rp_up = bp_rp(Teff_up)
    bp_rp_low = bp_rp(Teff_low)

    ax.plot(bp_rp_up, Mabs_up + dm, c="0.4", ls="dotted")
    ax.plot(bp_rp_low, Mabs_low + dm, c="0.4", ls="dotted")

    # dScu Murphey (2019)

    Teff_u = np.array([9300, 8500])
    Teff_l = np.array([7500, 6500])

    L_u = -1e-3 * Teff_u + 10.5
    L_l = -8.11e-4 * Teff_l + 6.672

    Mb_u = Mbol(L_u)
    Mb_l = Mbol(L_l)

    Mabs_u = Mb_u - BC(Teff_u)
    Mabs_l = Mb_l - BC(Teff_l)

    bp_rp_u = bp_rp(Teff_u)
    bp_rp_l = bp_rp(Teff_l)

    ax.plot(bp_rp_u, Mabs_u + dm, c="0.4", ls="dashed")
    ax.plot(bp_rp_l, Mabs_l + dm, c="0.4", ls="dashed")


def addBCepIS(ax):

    teff = [4.4114,  4.4163, 4.4184, 4.4234, 4.4309, 4.4350, 4.4346, 4.4139,
            4.3856, 4.3636, 4.3536, 4.3453, 4.3403]

    lum = [3.8346, 3.9697, 4.1691, 4.3442, 4.4660, 4.5724, 4.6477, 4.6300,
           4.4505, 4.2001, 4.0406, 3.8013, 3.3848]

    ax.plot(teff, lum, c='k', ls='dashed')


def add_IS_siemen(ax,
                  datadir='/Users/dario/PyhtonPackages/myhelpers/instability_strips/',
                  zorder=0, Zini='0.014', rot='0.0', alpha=0.2, y_axis='logL',
                  plot_label=False, IStype='both', p_mode_col='k', fill=False,
                  p_mode_ls='solid', g_mode_col='k', g_mode_ls='dashed', lw=.5):
    '''
    Function to plot Siemen Burssens' instability_strips given a certain axis.
    Written by Siemen Burssens

    @ ax --> axis on which to plot the strips
    @ datadir --> location of the .txt files containing the boundaries
    @ spherical --> spherical degrees
    @ zorder = zorder (matplotlib def.)
    @ alpha = alpha (matplotlib def.)


    '''
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


def plot_tracks(ax, grid_dir='/STER/siemenb/OB_strips/final_grid/',
                evo_tracks=[6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 32.0, 40.0],
                Zini=0.014, fov=0.020, ldm=1.0, xaxis='log_Teff', yaxis='log_L',
                alpha=1, lw=.5, ls='solid', color='0.6', zorder=0,
                x_offset=-0.01, y_offset=-0.03, text=True, fontsize=8,
                plot_ZAMS=True, zams_color='0.6', zams_ls='solid', zams_lw=.5):
    '''
    Add tracks to HRD.
    '''

    mesa_files = []

    for mass in evo_tracks:
        name = f'Z{Zini:.3f}_M{mass:.2f}_logD{ldm:.2f}_aov0.000_fov{fov:.3f}'
        modeldir = f'LOGS_{name}/'
        histname = f'{name}.hist'
        mesa_files.append(grid_dir + modeldir + histname)

    ZAMS = []

    def find_ZAMS(trackdata):
        #Hinit = trackdata.data('center_h1')[0]
        #print(np.argmin(trackdata.data('log_R')))
        # np.where(Hinit - trackdata.data('center_h1') > 1e-4)[0][0]

        #TAMS_idx = find_TAMS(trackdata)
        #print(np.argmin(np.abs(trackdata.data('log_Lnuc')[:TAMS_idx] - trackdata.data('log_L')[:TAMS_idx] - 0.9)))
        return np.argmin(trackdata.data('log_R'))


    def find_TAMS(trackdata):
        return np.where(trackdata.data('center_h1') < 1e-4)[0][0]

    for i, track in enumerate(mesa_files):
        trackdata = mr.MesaData(file_name=track, file_type='log')
        ZAMS_idx = find_ZAMS(trackdata)
        TAMS_idx = find_TAMS(trackdata)

        mass = evo_tracks[i]
        if mass < 14:
            add_idx = 43
        elif mass < 20:
            add_idx = 55
        elif mass < 25:
            add_idx = 65
        else:
            add_idx = 67

        xvals = trackdata.data(xaxis)[ZAMS_idx:TAMS_idx + add_idx]
        yvals = trackdata.data(yaxis)[ZAMS_idx:TAMS_idx + add_idx]

        ZAMS.append(np.array([xvals[0], yvals[0]]))

        ax.plot(xvals, yvals, lw=lw, c=color, alpha=alpha, zorder=zorder, ls=ls)

        if text:
            mass_label = f'${evo_tracks[i]:.0f}\,M_\odot$'
            ax.text(xvals[-1] + x_offset, yvals[-1] + y_offset, mass_label,
                    clip_on=True, c=color, fontsize=fontsize, zorder=5)

    if plot_ZAMS:
        ZAMS = np.array(ZAMS).T

        ax.plot(ZAMS[0], ZAMS[1], lw=zams_lw, c=zams_color, alpha=alpha, zorder=zorder,
                ls=zams_ls, label='ZAMS')
