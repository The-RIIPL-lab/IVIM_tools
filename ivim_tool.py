import os
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.ivim as ivim
from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from dipy.core.gradients import gradient_table
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.align import motion_correction
from dipy import __version__
import sys
print("Running dipy version", __version__)

nii_img = sys.argv[1]
bval_file = sys.argv[2]
bvec_file = sys.argv[3]
mask_file = sys.argv[4]
fn=os.path.splitext(nii_img)[0]
moco=str(fn+"_moco.nii.gz")
print("nifti input: {}\n bval: {}\n bvec: {}\n moco: {}".format(
   nii_img, bval_file, bvec_file, moco
))

# Step 1: Load the IVM image data
data, affine = load_nifti(nii_img)
bvals, bvecs = read_bvals_bvecs(bval_file,bvec_file)
gtab = gradient_table(bvals, bvecs, b0_threshold=0)

# Step 2: Preprocessing: denoising and masking
sigma = estimate_sigma(data, N=32) # head coil array!
data_denoised = nlmeans(data, sigma=sigma,  patch_radius=1, block_radius=2, rician=True)

# Step 3: Mask the data
#maskdata, mask = median_otsu(data_denoised, median_radius=4, numpass=2, vol_idx=[1,6,11,15])
mask,affine=load_nifti(mask_file)

# Step 3: Motion Correct the data
data_corrected = moco
if os.path.isfile(data_corrected):
    data_corrected, affine = load_nifti(data_corrected)
else:
    data_corrected, reg_affines = motion_correction(data, gtab, affine)
    save_nifti(moco, data_corrected.get_fdata(), data_corrected.affine)
    

# Step 4: IVIM model fitting)
ivim_model = ivim.IvimModelVP(gtab, maxiter=10)
ivim_fit = ivim_model.fit(data_corrected, mask)

# Step 5: Extract parameter maps
#f_map = ivim_fit.f
d_map = ivim_fit.D
d_star_map = ivim_fit.D_star
perfusion_fraction_map = ivim_fit.perfusion_fraction
S0_prediction = ivim_fit.S0_predicted

# Step 6: Save the parameter maps as NIfTI files
save_nifti(str(fn+'-S0_prediction.nii.gz'), S0_prediction, affine)
save_nifti(str(fn+'-d_map.nii.gz'), d_map, affine)
save_nifti(str(fn+'-d_star_map.nii.gz'), d_star_map, affine)
save_nifti(str(fn+'-perfusion_fraction_map.nii.gz'), perfusion_fraction_map, affine)
