# IVIM MRI Data Processing

This repository contains Python scripts for processing Intravoxel Incoherent Motion (IVIM) Magnetic Resonance Imaging (MRI) data. The main script is `IVIM-Run_IVIM_batches_on_SLURM.py`.

## IVIM-Run_IVIM_batches_on_SLURM.py

This script performs the following tasks:

1. **IVIM Data Processing**: It uses the `ivim_tool` function to process IVIM data. This function loads the IVIM image data, preprocesses it (denoising and masking), performs motion correction, fits the IVIM model, extracts parameter maps, and saves these maps as NIfTI files.

2. **SLURM Batch Script Creation**: It uses the `create_slurm_batch_scripts` function to create SLURM batch scripts for parallel processing of the IVIM data on a SLURM-managed high-performance computing cluster. The batch scripts unload the current Python module, load Python 3.9.5, find the IVIM dataset files, and run the `ivim_tool` function on each file.

To use this script, set the number of batches and the path to the IVIM dataset at the bottom of the script, and then run the script. It will generate the SLURM batch scripts, which you can then submit to your SLURM cluster for processing.

## Contact

This project is no longer active. Interested parties can contact the [RIIPL Lab](https://the-riipl-lab.github.io/) for more details and data.
- [DIPY](https://workshop.dipy.org/documentation/1.6.0./examples_built/reconst_ivim/)
- [Intravoxel incoherent motion magnetic resonance imaging: basic principles and clinical applications.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7757509/)