import os

# ---- Begin ivim_tool.py ----
def ivim_tool(nii_img, bval_file, bvec_file, mask_file):
    from dipy.io.image import load_nifti, save_nifti
    from dipy.io.gradients import read_bvals_bvecs
    import dipy.reconst.ivim as ivim
    from dipy.segment.mask import median_otsu
    from dipy.denoise.nlmeans import nlmeans
    from dipy.core.gradients import gradient_table
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.align import motion_correction
    from dipy import __version__

    print("Running dipy version", __version__)

    # Check if the output file already exists, if yes, skip the processing
    output_file = os.path.splitext(nii_img)[0] + '-perfusion_fraction_map.nii.gz'
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipping processing.")
        return

    fn = os.path.splitext(nii_img)[0]
    moco = str(fn + "_moco.nii.gz")

    # Step 1: Load the IVM image data
    data, affine = load_nifti(nii_img)
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)

    # Step 2: Preprocessing: denoising and masking
    sigma = estimate_sigma(data, N=32)  # head coil array!
    data_denoised = nlmeans(
        data, sigma=sigma, patch_radius=1, block_radius=2, rician=True
    )

    # Step 3: Mask the data
    mask, affine = load_nifti(mask_file)

    # Step 3: Motion Correct the data
    data_corrected = moco
    if os.path.isfile(data_corrected):
        data_corrected, affine = load_nifti(data_corrected)
    else:
        data_corrected, reg_affines = motion_correction(data, gtab, affine)
        save_nifti(moco, data_corrected.get_fdata(), data_corrected.affine)

    # Step 4: IVIM model fitting
    ivim_model = ivim.IvimModelVP(gtab, maxiter=10)
    ivim_fit = ivim_model.fit(data_corrected, mask)

    # Step 5: Extract parameter maps
    d_map = ivim_fit.D
    d_star_map = ivim_fit.D_star
    perfusion_fraction_map = ivim_fit.perfusion_fraction
    S0_prediction = ivim_fit.S0_predicted

    # Step 6: Save the parameter maps as NIfTI files
    save_nifti(str(fn + '-S0_prediction.nii.gz'), S0_prediction, affine)
    save_nifti(str(fn + '-d_map.nii.gz'), d_map, affine)
    save_nifti(str(fn + '-d_star_map.nii.gz'), d_star_map, affine)
    save_nifti(str(fn + '-perfusion_fraction_map.nii.gz'), perfusion_fraction_map, affine)
# ---- End ivim_tool.py ----

# ---- Begin create_slurm_batch_scripts ----
def create_slurm_batch_scripts(base_script_name, num_batches, ivim_dataset_path):
    for batch_index in range(num_batches):
        script_name = f"{base_script_name}_batch{batch_index}.slurm"

        with open(script_name, "w") as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write("#SBATCH --job-name=ivim_processing\n")
            script_file.write("#SBATCH --output=ivim_%A_%a.out\n")
            script_file.write("#SBATCH --error=ivim_%A_%a.err\n")
            script_file.write("#SBATCH --nodes=1\n")
            script_file.write("#SBATCH --ntasks=10\n")
            script_file.write("#SBATCH --cpus-per-task=3\n")
            script_file.write("#SBATCH --mem-per-cpu=2048\n")
            script_file.write("#SBATCH --time=18:00:00\n")
            script_file.write("\n")
            script_file.write("module unload python\n")
            script_file.write("module load python/3.9.5\n")
            script_file.write("\n")
            script_file.write(f'IVIM_DATASET="{ivim_dataset_path}"\n')
            script_file.write("\n")
            script_file.write("FILES=($(find $IVIM_DATASET -type f -name '*_moco.nii.gz'))\n")
            script_file.write("\n")
            script_file.write(f"INDEX=$(($SLURM_ARRAY_TASK_ID * {batch_index + 1}))\n")
            script_file.write("\n")

            for i in range(10):
                script_file.write(f'FILE=${{FILES[INDEX + {i}]}}\n')
                script_file.write("BASE_NAME=$(basename $FILE _moco.nii.gz)\n")
                script_file.write("\n")
                script_file.write("OUTPUT_FILE=${BASE_NAME}-perfusion_fraction_map.nii.gz\n")
                script_file.write("if [ -f $OUTPUT_FILE ]; then\n")
                script_file.write("    echo Output file $OUTPUT_FILE already exists, skipping\n")
                script_file.write("    continue\n")
                script_file.write("fi\n")
                script_file.write("\n")
                script_file.write(f"python ivim_tool.py ${{BASE_NAME}}_moco.nii.gz ${{BASE_NAME}}.bval ${{BASE_NAME}}.bvec ${{BASE_NAME}}_brain_mask.nii.gz &\n")
                script_file.write("\n")

            script_file.write("wait\n")
# ---- End create_slurm_batch_scripts ----

# Set the number of batches and IVIM dataset path
num_batches = 15
ivim_dataset_path = ""

# Generate SLURM batch scripts
create_slurm_batch_scripts("run_ivim", num_batches, ivim_dataset_path)
