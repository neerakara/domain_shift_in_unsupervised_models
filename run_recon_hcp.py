# ============================================================
# load test data, undersampled it and feed it to the recon funcion.
# ============================================================
import numpy as np
import pickle
import vaerecon
import utils
from US_pattern import US_pattern
from MR_image_data import MR_image_data
import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--vol', type=int, default=5)
parser.add_argument('--sli', type=int, default=150) 
parser.add_argument('--usfact', type=float, default=4) # undersampling factor 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=0) 
parser.add_argument('--num_iter', type = int, default = 202) 
parser.add_argument('--n1', type = int, default = 5) 
parser.add_argument('--n2', type = int, default = 5)
parser.add_argument('--gamma', type = float, default = 1.0) 
args=parser.parse_args()

# ======================================
# hyper-parameters
# ======================================
regtype = 'reg2_dc'
reg = 0.1
dcprojiter = 10
chunks40 = True
n1 = args.n1 # number of updates of the normalization module
n2 = args.n2 # number of updates of the image
R = args.usfact # define the undersampling factor
gamma = args.gamma # simulated contrast change
vol = args.vol
sli = args.sli

# =============================
# make results directory if it does not exist
# =============================     
basefolder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/'
results_folder = basefolder + 'code/v2.0/results/hcp/subject' + str(vol) + '/'
results_folder = results_folder + 'us' + str(R) + '_sli' + str(sli) + '_regtype_' + regtype + '_reglambda_' + str(reg) + '_n1_' + str(n1) +  '_n2_' + str(n2) + '_gamma_' + str(gamma) + '/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ndims = 28
lat_dim = 60
mode = 'MRIunproc' # 'Melanie_BFC'

# ==================================
# create an instance of the undersampling pattern class
# ==================================
USp = US_pattern()

# ==================================
# make a dataset to load images
# there are 40 chunks in total, the 39th chunk is used for testing.
# ==================================
noise = 0
MRi = MR_image_data(dirname = basefolder + 'data/',
                    imgSize = [260, 311, 260],
                    testchunks = [39],
                    noiseinvstd = noise)

# ==================================
# cast the undersampling factor to an integer if not already an integer
# ==================================
if np.floor(R) == R: 
     R = int(R)

# ===============================
# Add subject name to log
# ===============================
logging.info('============================================================')
logging.info('DATASET: HCP')
logging.info('SUBJECT:' + str(args.vol))
logging.info('SLICE:' + str(args.sli))
logging.info('UNDERSAMPLING FACTOR:' + str(args.usfact))
logging.info('CONTRAST GAMMA:' + str(args.gamma))
logging.info('============================================================')

# =======================================================================================
# RECON
# =======================================================================================

# =============================
# get the original (fully-sampled) image. In the HCP dataset, we only have the magnitude image.
# =============================
orim = MRi.center_crop(MRi.get_image(39, vol, sli),
                       252,
                       308,
                       252,
                       308)

# =============================
# magnitude image (for the HCP dataset, orim and orima are the same)
# =============================
orima = np.abs(orim)

# =============================
# SYNTHETIC phase
# =============================
nx, ny = (252, 308)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
synthph = np.transpose(np.pi*(xv+yv)-np.pi) # synthetic phase

# synthph = 1.5
# orim = orima*np.exp(1j * synthph * (orima>0.1))
orim = orima.copy()

# =============================
# change contrast of image
# =============================
orim_new_contrast = orim ** gamma
# normalize intensity range after gamma transformation
if gamma != 1.0:
     orim_new_contrast = (orim_new_contrast - np.min(orim)) / (np.max(orim) - np.min(orim))
orim = orim_new_contrast

# ======================================
# undersampling pattern. either load or generate.
# ======================================
try:
     uspat = np.load(basefolder + 'data/uspats/uspat_us' + str(R) + '_vol' + str(vol) + '_sli' + str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp = US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(orim.shape, R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder + 'data/uspats/uspat_us' + str(R) + '_vol' + str(vol) + '_sli' + str(sli), uspat)

# ======================================
# apply undersampled FT on the image, this returns the undersampled k space.
# undersampled k space (usksp) = undersampling_mask * FT(image)
# ======================================
usksp = utils.UFT(orim, uspat, normalize=False) / np.percentile(np.abs(utils.tUFT(utils.UFT(orim, uspat, normalize=False),
                                                                                  uspat, normalize=False).flatten()), 99)
usksp = usksp[:, :, np.newaxis]

# =============================
# if continue run...
# =============================
if args.contrun == 0:
     continue_recon = ''
else:
     continue_recon = results_folder + 'results'
     
# =============================
# do the reconstruction
# =============================
rec_vae = vaerecon.vaerecon(usksp, # undersampled k space
                            sensmaps = np.ones_like(usksp), # coil sensitivity map
                            dcprojiter = dcprojiter,
                            lat_dim = lat_dim,
                            patchsize = ndims,
                            contRec = continue_recon,
                            parfact = 25,
                            num_iter = args.num_iter,
                            regiter = 10,
                            reglmb = reg,
                            regtype = regtype,
                            half = True,
                            mode = mode,
                            chunks40 = chunks40,
                            n1 = n1,
                            n2 = n2,
                            log_dir = results_folder)

# =============================   
# write results to disk
# =============================   
pickle.dump(rec_vae[0], open(results_folder + 'results', 'wb'))

# =============================   
# This will be the final reconstructed image
# =============================   
lastiter = int((np.floor(rec_vae.shape[1] / 13) - 2) * 13)
maprecon = rec_vae[:, lastiter].reshape([252, 308]) # this is the final reconstructed image