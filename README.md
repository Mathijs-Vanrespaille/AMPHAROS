This tutorial will teach you the basics of Asteroseismic Modelling of Hot Pulsators Accounting for Rotation using Oscillation code StORM (AMPHAROS), the forward modelling code presented in Vanrespaille et al. (submitted). The user is expected to have read Section 4 of that paper to understand the default forward modelling methodology. Also beneficial to read are Section 3 for a description of the stellar models and oscillation computations and Section 5 and Appendix B for the alternative methods included in AMPHAROS.

# Installation

## Requirements

To run AMPHAROS, the following packages are required. The code is not expected to strongly depend on the specific versions used. Nonetheless, we list the versions used to test the code and collect the results presented in Vanrespaille et al. (submitted) for debugging purposes:

 - h5py (3.12.1)
 - matplotlib (3.5.1)
 - NumPy (1.26.4)
 - pandas (1.4.2)
 - Python (3.9.12)
 - SciPy (1.13.1)
 - Optional: pygyre (1.4)


## Install instructions

To install the AMPHAROS Python package, create and navigate into an appropriate folder, for example `~/ampharos/`.
Download the [GitHub repository]{https://github.com/Mathijs-Vanrespaille/AMPHAROS} and unzip it there. Alternatively, clone the repository using

```bash
git clone https://github.com/Mathijs-Vanrespaille/AMPHAROS
```

Now create an environment to run AMPHAROS in and perform a local pip install. If you use anaconda (or miniconda), run the following:

```bash
conda create -n ampharos python=3.9
conda activate ampharos
python -m pip install -e ~/ampharos/
```

You have successfully installed AMPHAROS, congratulations! To verify it has been installed correctly, you can test the installation by trying out the example walkthrough detailed below.


# Input for the code

The user has to provide three things to the code:

 - The MESA-StORM (and optionally GYRE) model grid output
 - A parameter file describing at what values for the free parameters to look for model output
 - The observational input

 We now briefly describe what each of these should contain and where one can obtain them.


### MESA-StORM(-GYRE) grid

The model output is expected to be structured as follows:

```
{MESA_top_dir}
│
└───Z{Z1}_M{M1}_logD{logD1}_fov{fov1}
│   │   Z{Z1}_M{M1}_logD{logD1}_fov{fov1}.hist
│   │
│   └───storm_summaries
│   │   │   storm_summary_Xc{Xc11}_mn{mn11}.hdf5
│   │   │   storm_summary_Xc{Xc11}_mn{mn11}.hdf5
│   │   │   ...
│   │   
│   └───gyre_summaries
│       │   gyre_Xc{Xc11}_mn{mn11}.HDF
│       │   gyre_Xc{Xc12}_mn{mn12}.HDF
│       │   ...
│   
└───Z{Z2}_M{M2}_logD{logD2}_fov{fov2}
│   │   Z{Z2}_M{M2}_logD{logD2}_fov{fov2}.hist
│   │
│   └───storm_summaries
│   │   │   storm_summary_Xc{Xc21}_mn{mn21}.hdf5
│   │   │   storm_summary_Xc{Xc21}_mn{mn21}.hdf5
│   │   │   ...
│   │   
│   └───gyre_summaries
│       │   gyre_Xc{Xc21}_mn{mn21}.HDF
│       │   gyre_Xc{Xc22}_mn{mn22}.HDF
│       │   ...
│       
└───...
```

Note that the `gyre_summaries` are optional and are only necessary when using the alternative modelling method employing the first-order rotational treatment, described in Appendix B of Vanrespaille et al. (submitted).  

Herein *Zi* is the initial metallicity of model *i* with three decimal places, *Mi* the initial mass in solar masses with two decimal places, *logDi* the mixing coefficient in cgs at the base of the envelope with one decimal place, *fovi* the exponential overshoot parameter in local pressure scale with three decimal places, *Xcij* the central hydrogen mass fraction at which output step *j* in model *i* is saved and *mnij* is the corresponding MESA model number. *MESA_top_dir* is the top level directory containing all the model output, which the user needs to provide to the code.
The files ending in .hist are the history files created by MESA. Files ending in .HDF in the `gyre_summaries` subdirectories are the summary files created by GYRE. Files ending in .hdf5 are custom summaries of the frequencies predicted by StORM at the different rotation rates for a given input model. These have been created with the Python scripts provided in the StORM setup subdirectory in the Zenodo repository (link to be added).

Due to the size of the model output the code requires, the full model output is not included in the GitHub, though a small subgrid is provided for the example walkthrough below. The remaining models and StORM output in this grid can be generated with the MESA work directory and StORM setup in the Zenodo repository accompanying Vanrespaille et al. (submitted) (link to be provided). All the model output used in Vanrespaille et al. (submitted) are also available at the KU Leuven Research Data Repository (link to be added). Note that this contains far more than what AMPHAROS requires.


### Parameter file

This is a basic ascii file describing at which model parameters to look for output and subsequently try to fit the observations to. By default, the code assumes this is a comma-separated values (csv) file, including a header containing the names of the columns. This file must contain at least four columns, namely:

 - Initial metal mass fraction. The name of this column must be "Z" or "Zini".
 - Initial stellar mass in solar mass. The name of this column must be "M" or "Mini".
 - Mixing coefficient at the base of the envelope in cgs. The name of this column must be "log_D", "logD", "logDmix", "log_Dmix", or "log_D_mix".
 - Overshoot parameter in units of local pressure scale height. The name of this column must be "f_ov" or "fov".

Note that adding more columns is permitted; AMPHAROS will not read those in.

An short example of such a file can be found in this GitHub repository at `input/tutorial/grid_test.csv`. The parameter file used in Vanrespaille et al. (submitted) can be found at `input/grid_params/grid_varying_logD.csv`.


### Observation file

This is a basic ascii file describing the input observations for the code. By default, AMPHAROS assumes this is a tab-separated values (tsv) file, including a header containing the names of the columns. This file must contain at least four columns plus 6 per observed identified modes. The columns that must always be included are:

 - "name": User-chosen name of the star used to define where the modelling output and plots will be saved.
 - "save_suffix": User-chosen suffix used in the filenames in which the modelling output and plots will be saved. Useful to differentiate between different analyses of the same star.
 - "log_Teff": Base-10 logarithm of the observed effective temperature in kelvin.
 - "log_Teff_err": Uncertainty on the base-10 logarithm of the observed effective temperature in kelvin.
 - "log_L": Base-10 logarithm of the observed luminosity in solar luminosity.
 - "log_L_err": Uncertainty on the base-10 logarithm of the observed luminosity in solar luminosity.
 - "Nm_ID": Number of identified modes.
 - "Ni_ID": Number of identified radial modes or rotationally split multiplets.
 - "n_pg_main": Radial order of the fixed mode (referred to as the 'main mode' in the code). This column can be left empty.
 - "all_f": list of frequencies of all observed signals, including unidentified ones, in cycles per day.
 - "all_a": list of amplitudes of all observed signals, including unidentified ones. Only used in visualisation so the unit does not really matter.

Next, for each identified mode with index *j* varying between 1 and N_f, the following must be provided:
 - "i{j}": An index indicating to which multiplet this mode belongs. Varies between 1 and Ni_ID.
 - "l{j}": Degree
 - "m{j}": Azimuthal order
 - "f{j}": Frequency in cycles per day
 - "f_err{j}": Uncertainty on the frequency in cycles per day
 - "a{j}": Amplitude. Only used in visualisation so the unit does not really matter.

Note that adding more columns is permitted, to add comments for instance, which AMPHAROS will ignore.

Examples of such a file can be found in this GitHub repository at `input/tutorial/obs_test_radial.tsv`, `input/tutorial/obs_test_nonRadial.tsv`, and `input/tutorial/obs_test_nonRadial_withVariations.tsv`. The observation files used in Vanrespaille et al. (submitted) can be found in the subdirectory `input/observations/`.


# Example Walkthrough

In this section, we provide some examples with increasing complexity on how to use the code using the default methodology described in Section 4 of Vanrespaille et al. (submitted).


### Modelling two stars with identified radial modes

Let us begin with a simple case of two stars with a known radial mode and one rotationally split multiplet. These are described in `input/tutorial/obs_test_radial.tsv`.

In order to use the code, open a Python terminal, Jupyter notebook, or something similar in the work directory provided in this repository. Then import the AMPHAROS using

```python
import ampharos as amphy
```

Next, you need to load in the observations and the model grid. As this can take a significant amount of time (> 1h when loading in 1505 models), we load in a much smaller subgrid for teaching purposes described in `input/tutorial/grid_test.csv`. Say the models are stored in `/path/to/model_grid/`, where you will need to replace `/path/to/` with the relevant absolute path. For this test, you can use the small subgrid provided in the GitHub repository. Assuming you installed AMPHAROS in `~/ampharos/` run the following in your Python terminal:

```python
MESA_top_dir = "~/ampharos/model_grid/"
grid, obs_radial = amphy.initialise_all("input/tutorial/obs_test_radial.tsv", "input/tutorial/grid_test.csv", MESA_top_dir)
```

Wait until the subgrid is loaded in, which should take roughly a minute. In the meantime, you should see two stars' telephone numbers printed, followed by the message "Finished reading {k}/27" 27 times. As the radial orders of these stars' modes are not known a priori, we try three different radial orders for the fixed radial mode. Once the grid is loaded in, you can start the analysis using

```python
amphy.analyse_observations(grid, obs_radial, n_pgs=[1,2,3])
```

On the laptop used to test this tutorial, this analysis took roughly two minutes. In the meantime, you should see the names of the two stars being printed to the screen. It is also likely that some warnings pop up. You can safely ignore these for now, or consult the **common warning messages** section below.

Once the analysis is complete, you should see two new subdirectories, `plots/` and `results/` have been created. Take a few minutes to explore these subdirectories. Of particular importance are `plots/HD117704/picking_npg/` and `plots/TIC465541764/picking_npg/`, which contain figures designed to help you figure out what the most likely radial order of the fixed mode is. You need to appraise and manually select the most likely radial order identifications. Try that yourself! Vanrespaille et al. (submitted) identified the most likely radial mode of HD 117704 to be $ n_pg=1 $ and that of TIC 465541764 to be $ n_pg=2 $. Do you agree?


### Modelling stars with a known radial order for the fixed mode

Now that we known the radial orders of the fixed modes of HD 117704 and TIC 465541764, we can tell AMPHAROS to only analyse them using only those radial orders provided in the "n_pg_main" column in the observation file. To do so, run the following:

```python
amphy.analyse_observations(grid, obs_radial, n_pgs=[])
```

Herein we left the argument `n_pgs` deliberately empty, which tells AMPHAROS to look for the "n_pg_main" column in the observation file. However, note that the name of HD 117704 is not printed to the screen. That means it has been ignored in the analysis. If you open the file `input/tutorial/obs_test_radial.tsv`, you will see that the column "n_pg_main" is left empty. Consequently, AMPHAROS does not know what radial order to use because we also did not tell it which radial orders to try out. Therefore, it simply skipped this target. To alleviate this, complete the column with the radial order you found above and run the following again:

```python
mf.analyse_observations(grid, obs_radial, n_pgs=[])
```

Now both names should get printed to the screen. You should take stock of how long this computation took. It should take only roughly a third of the time of the analysis when you provided `n_pgs=[1,2,3]`.


### Modelling a star without a known radial mode

Now how does one analyse a star without an identified radial mode to use to fix the age. We first need to read the observation file that star (HD 340239) is in, though we do not need to re-load the grid. To do so, use the following command:

```python
obs_nonradial = amphy.read_observations("input/tutorial/obs_test_nonRadial.tsv", sort_by_l=False, sort_by_freq=True)
```

Let's take a moment to explain what these arguments `sort_by_l` and `sort_by_freq` mean. Internally, AMPHAROS creates lists of various properties of the zonal modes in the observation files, notably their degree and frequency. The first mode in these lists is used as the fixed mode. By default, it will sort these lists by the degree such that a radial mode, i.e. one with $ l=0 $, comes first. This is not helpful when no radial mode is identified. Instead, as explained by Section 4.2 of Vanrespaille et al. (submitted), it is best to use the zonal mode with the highest frequency as the fixed mode. That's why we tell AMPHAROS not to sort by degree and instead to sort by frequency, which will place the largest frequency first.

Now that this is loaded in, we can analyse the star, trying out different possibilities for the radial order. However, now we have to also consider that the quintuplet ($ l=2 $) we use as the fixed mode of this star might be a g-mode, meaning $ n_pg<0 $, or an f-mode, meaning $ n_pg=0 $. Therefore, analyse this star using:

```python
amphy.analyse_observations(grid, obs_nonradial, n_pgs=[-2,-1,0,1,2,3])
```

You should see a variety of warnings and messages pop up, the meaning of which you can consult below. Look at the results and figures created for this target. For what radial order(s) did the code not create any output? For what radial order(s) does the code generate bizarre and thus unreliable figures (be sure to check the Hertzsprung-Russell diagram in particular) ?

Vanrespaille et al. (submitted) took this star's most likely radial order for the fixed mode to be $ n_pg=0 $, meaning it is due to an f-mode. Do you agree?


### Constraining differential rotation

As described in Section 7. of Vanrespaille et al. (submitted), differential rotation can be measured by analysing a star with several rotationally-split multiplets several times. Each time, we only model the rotational splitting of one multiplet, such that the rotation frequency is optimised to that one multiplet alone. To that end, we prepared the observation file for HD 340239 `input/tutorial/obs_test_nonRadial_withVariations.tsv` with three different variations: once with all identified signals, once with one complete multiplet and the other multiplet's zonal mode and once vice versa. Note that each line has a unique "save_suffix".

Let's analyse these different variations now. Herein, we'll only use the identified radial order of the star provided in the observation file:

```python
obs_nonradial_variations = amphy.read_observations("input/tutorial/obs_test_nonRadial_withVariations.tsv", sort_by_l=False, sort_by_freq=True)
amphy.analyse_observations(grid, obs_nonradial_variations, n_pgs=[])
```
Once completed, navigate to the star's results folder at `results/HD340239/n_pg_main0/`. You should see three different copies of each file with different suffixes in their names. A similar structure should be seen in the relevant plots folder.


## Alternative modelling approaches

As described in Section 5 and Appendix B of Vanrespaille et al. (submitted), AMPHAROS features some alternative modelling approaches. These are to analyse the rotational splitting with the first-order treatment of GYRE and to neglect the rotational splitting in the selection of the best model and computation of the statical parameter estimates (though the rotational splitting is still used in the optimisation of the rotation frequency for each evolutionary track). We now briefly show how to employ these alternative methods:


### Neglecting rotational splitting in the cost function

If you haven't done so already, read in the test grid and observations using:

```python
import ampharos as amphy
MESA_top_dir = "/path/to/MESA_output_fullGrid/"
grid, obs_radial = amphy.initialise_all("input/tutorial/obs_test_radial.tsv", "input/tutorial/grid_test.csv", MESA_top_dir)
```

To drop the rotational splitting from the cost function and thus the selection of the best model and construction of the statistical model, simply add the `exclude_splitting_from_chi2` flag to the `analyse_observations` function:

```python
amphy.analyse_observations(grid, obs_radial, n_pgs=[], exclude_splitting_from_chi2=True)
```

Take a look at the `results/` and `plots/` subdirectories. You should see that new folders have been created with the suffix `_chi2SplittingExcluded`.


### First-order rotation treatment

**Requires pygyre be installed**

To use GYRE's first-order rotation treatment rather than StORM's second-order treatment, you need to read in some additional information from GYRE when loading in the grid. As this slows down the grid loading somewhat and requires extra model output not all users may have, this is turned off by default. Load the grid in with that additional information using:

```python
import ampharos as amphy
MESA_top_dir = "/path/to/MESA_output_fullGrid/"
grid, obs_radial = amphy.initialise_all("input/tutorial/obs_test_radial.tsv", "input/tutorial/grid_test.csv", MESA_top_dir, add_beta=True)
```

Then perform the actual analysis with:

```python
amphy.analyse_observations(grid, obs_radial, n_pgs=[], use_GYRE=True)
```

You should find that this modelling is completed much faster than when using StORM. Take a look at the `results/` and `plots/` subdirectories. You should see that new folders have been created with the suffix `_GYRE`.

Note that you can also turn off the rotational splitting's contributions to the cost function by setting `exclude_splitting_from_chi2=True`. Also note that you can still analyse the star with second-order effects by setting `use_GYRE=False` with the same grid already loaded in.


## Optimising your grid input

The functions `initialise_all` features a few more optional parameters that have not been discussed in the walkthrough above yet. These let the user optimise what parts of the model grid and oscillation computations to include or ignore, hence trading parameter space coverage for computational speed. We briefly explain the options here and provide recommended values for the asteroseismic forward modelling of beta Cephei pulsators:

 - `Xc_min`: Smallest central hydrogen mass fraction to include in the grid. StORM summaries for an Xc smaller than this minimum are not loaded in and extrapolating beyond this minimum in the age fixing is not permitted. If None, no filtering based on the lower limit of the central hydrogen mass fraction is done. The default value is 0.0001. In most applications, unless one is modelling a highly evolved pulsator, this can be set higher, e.g. to 0.01.
 - `Xc_max`: Largest central hydrogen mass fraction to include in the grid. StORM summaries for an Xc greater than this maximum are not loaded in and extrapolating beyond this maximum in the age fixing is not permitted. If None, no filtering based on the upper limit of the central hydrogen mass fraction is done. The default value is 0.7012. As beta Cephei stars are primarily expected to occur in the second half of the main sequence, it is usually safe to lower this upper limit, e.g. to 0.5.
 - `frots_rel_to_include`: Which rotation rates relative to the Keplerian critical rotation rate to extract from the StORM summaries. These relative rotation rates must be included in all summaries, otherwise the grid initialisation will fail. Leave empty to include all frots_rel available in the StORM summaries. The default value is []. These can be used to limit the number of rotation rates for which the age needs to be optimised. For instance, if one has a slowly rotating beta Cephei star, say with observed rotation splitting on the order of a few 10$ ^{-2} $ cycles per day (d$ ^{-1} $), one might provide `frots_rel_to_include = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]`. Herein, bear in mind that the typical Keplerian critical rotation frequency of a beta Cephei pulsator is around 1 or 2 cycles per day (d$ ^{-1} $).
 - `ls_to_include`: The degrees of the modes to be read in. One entry for each combination of (l,n_pg). Must be of the same length as n_pgs_to_include. The default value is [0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2].
 - `n_pgs_to_include`: The radial orders of the modes to be read in and to try fitting the observations against. These are defined from the Eckart-Scuflaire-Osaki scheme, except for l=1 modes, which use the Takata scheme. One entry for each combination of (l,n_pg). This list must be of the same length as ls_to_include. The default value is [1,2,3,4,-2,-1,1,2,3,4,-2,-1,0,1,2,3,4].


# Common warning messages

We show some examples of commonly encountered warning messages and what to do about them.

1. **RuntimeWarning: All-NaN slice encountered**

This warning occurs when a particular mode is not present in the StORM summaries around the fixed age in an evolutionary track. Usually that means that the code is looking for a mode that falls outside the frequency range used in the oscillation computations, which is commonly encountered when the assumed radial order of the fixed mode is wrong or parts of the parameter space provided in the parameter file are too different from the star's true values. It is normal for this warning to show up a few times in a modelling run and can be safely ignored. If it occurs many times in a run, this likely indicates the range of radial orders considered for the main mode is unnecessarily wide or the target falls outside the grid's parameter range.


2. **make_plots_to_pick_npg : No models compatible with log_Teff and log_L!**

This message appears when no evolutionary track's optimised model lies within the observed 2$ \sigma $ error ellipse in the Hertzsprung-Russell diagram. This usually indicates the radial order just tried out is wrong or that the parameter space provided through the parameter file does not cover the star's observed position on the Hertzsprung-Russell diagram. Alternatively, the provided base-10 logs of the effective temperature in K or luminosity in solar luminosity might be mistaken or their uncertainties underestimated.

When this message appears, the code will create what plots and output it can, but these should be considered unreliable.


3. **All chi2 are NaN! No useable model was found!** OR **No models to use in the statistical model provided!** OR **write_model :  provided model is a NaN!**

When searching for the best model between the optimised models of each all evolutionary tracks, no optimised models were found. This occurs when no evolutionary track's predicted frequency for the fixed mode matches the observed at any age. Usually that means the considered radial order for the fixed mode is wrong. Alternatively, the provided frequency of the fixed mode could be mistaken. Make sure it is expressed in cycles per day (d$ ^{-1} $)!

When one or more of these warnings show up (they usually appear together), no plots or output will be created for this fixed mode's radial order.


4. **Missing GYRE file**

This message can appear when loading in the grid with `add_beta=True` when no GYRE output was found at a particular age when StORM output was found. In that case, the $\beta = 1 - C_{nl}$ is left empty for that given model and we count on interpolation of $\beta$ from the surrounding output. These usually occur near the terminal-age main sequence ($X_c < 0.05$), so unless you model a highly evolved star and are missing so many GYRE summaries that interpolation becomes unreliable, this message can be safely ignored.


# Modifying the code, figures, or output

Users are welcome to adapt AMPHAROS for their own purposes. The code is spread over four Python files:
 - `__init__.py`: The top level functions that the user is expected to interact with the most are contained here. Here the user can control what plots are created, how the rotation frequency is optimised, how the statistical model is computed, which history results to add to the output, etc.
 - `visualisation.py`: This contains the functions used to create figures. Most functions already have a number of arguments to help the user optimise the figures, though creating more functions or further altering the functionality there is encouraged as well.
 - `io_functions.py`: These functions help read in the model grid and observations and create the output of the code.
 - `support_functions.py`: These functions primarily exist to aid other functions in the other three files, so changes here will propagate. Be cautious when altering these functions!


# Contact

Questions, comments, and suggestions are very much welcome and can be sent to AMPHAROS's author, Mathijs Vanrespaille (mathijs.vanrespaille@kuleuven.be)


# How to cite this code

If you use this forward modelling code in your publications, please cite the following paper, which describes the methodology and first scientific application:

(Link to NASA ADS once available)

```
INSERT COPYABLE .BIB ENTRY FROM THE ADS
```


# Licence

This software is distributed under a GNU General Public License v3.0 (GNUgplv3). The terms and conditions are detailed in the file `COPYING`. To briefly summarise, this means you may copy, modify, and re-distribute the software as long as any modifications are tracked and described upon re-distribution. The re-distributed version of the code must use the same licence.
