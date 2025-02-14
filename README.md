# Brain network organization shapes rhythmic universals in music

## Overview

This repository contains the code and data for the paper "Brain network organization shapes rhythmic universals in music".

*Note.* The behavioural finger-tapping data was collected using a custom experiment server built around the Python package REPP (Anglada-Tort et al., 2022).
The code for running the experiment server is available upon request from the corresponding authors.

## Installation instructions

### System requirements
To use these scripts, users will have to have the following softwares installed on their system. The software versions as used in our analysis are given
in parentheses.

#### General software
- **MATLAB** (v. R2016b): A platform for numerical computation, visualization, and programming. Installation instructions for Mac, Windows, and Linux
are available at [MathWorks Installation Guide](https://it.mathworks.com/help/install/ug/install-products-with-internet-connection.html)
- **R** (v. 4.2.2): A programming language and free software environment for statistical computing and graphics. Download the latest R installer for your
operating system (Windows, Mac, or Linux) from the official CRAN mirrors at (https://cran.r-project.org/mirrors.html).
Run the installer and follow the on-screen instructions to complete the setup.
- **Python** (v. 3.9.6): A general-purpose programming language. Installation instructions can be found on the [Python website](https://www.python.org/).

#### Resting-state fMRI analysis
- **Conn Toolbox** (v. 2022a): An open-source, MATLAB-based software for analyzing functional connectivity Magnetic Resonance Imaging (fcMRI).
Installation instructions for Mac, Windows, and Linux can be found at [Conn Installation Guide](https://web.conn-toolbox.org/resources/conn-installation)
- **SPM12** (RRID:SCR_007037): A software package designed for the analysis of brain imaging data sequences.
Installation instructions can be found [here](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)

#### Diffusion analysis
- **MRtrix3** (v. 3.0.2_RC3): A toolbox for performing diffusion MRI analyses. Installation instructions can be found [here](https://www.mrtrix.org/).
- **FSL** (v. 5.0.9): A library of analysis tools for fMRI, MRI and DTI brain imaging data. Installation instructions can be found [here](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FSL.html).

#### Behavioural analysis and statistics
All required _R_ libraries will be installed automatically when running the .R or .Rmd scripts. The scripts will check whether the libraries are installed,
and if not will install them accordingly. Note that for the _R Markdown_ statistics documents, it is also possible to open the ``.html`` files to view the
code, the results, and any produced plots without running any software.

All Python packages and version numbers are included in the ``python_requirements.txt`` file. It is recommended to first create a [virtual environment](https://docs.python.org/3/library/venv.html), and
then install these requirements using ``pip install -r python_requirements.txt`` in a terminal run from the folder where the requirements file has been stored.

## Folder structure
This repository is organized as follows:
```
├── data
│   ├── behavioural
│   │   ├── processed
│   │   └── raw
│   │       ├── data_participants_info
│   │       └── data_participants_tapping
│   ├── brain
│   │   ├── adjacency
│   │   ├── between_network_connections
│   │   ├── correlations
│   │   ├── fixel
│   │   └── mappings
│   └── pilot
│       ├── processed
│       └── raw
│           ├── data_participants_info
│           └── data_participants_tapping
├── plots
│   ├── behavioural_measures
│   ├── connectograms
│   ├── connectomes
│   ├── correlations_behav_conn
│   ├── fixel
│   └── network_connections
├── results
│   ├── fixel_based
│   └── network_assignment
└── scripts
    ├── behavioural_measures_calculate
    ├── behavioural_measures_stats
    ├── cpm
    │   └── utilities
    ├── fixel_based
    ├── gold_msi_plots
    ├── network_assignment
    └── plots
```

The data that are included in this repository concern the raw and processed behavioural data, as well as processed MRI data. Raw MRI data
can be requested from the corresponding author.

## Refernces

```
Anglada-Tort, M., Harrison, P. M. C., & Jacoby, N. (2022). REPP: A robust cross-platform solution for online sensorimotor synchronization experiments. Behavior Research Methods. https://doi.org/10.3758/s13428-021-01722-2
```