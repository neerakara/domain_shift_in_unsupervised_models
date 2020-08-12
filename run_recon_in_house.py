import numpy as np
import pickle
import vaerecon
from US_pattern import US_pattern
import argparse
import h5py
import utils

parser = argparse.ArgumentParser(prog = 'PROG')
parser.add_argument('--sli', type = int, default = 3) 
parser.add_argument('--base', default = "sess_02_07_2018/CK/")
parser.add_argument('--usfact', type = float, default = 2) 
parser.add_argument('--contrun', type = int, default = 0) 
args=parser.parse_args()

basefolder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/data/'
ndims = 28
lat_dim = 60
mode = 'MRIunproc' #'Melanie_BFC'
USp = US_pattern()
usfact = args.usfact
if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)

# =======================================================================================
# RECON
# =======================================================================================
dirbase = basefolder + 'the_h5_files/' + args.base
sli = args.sli
ddr = np.array((h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')['DS1']))
ddi = np.array((h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')['DS1']))
dd = ddr + 1j*ddi
dd = np.transpose(dd)
# dd=np.transpose(dd,axes=[1,0,2])
dd = np.rot90(dd, 3)
dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd, axes=[0,1]), axes=[0,1]), axes=[0,1])

espsi = np.array((h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')['DS1']))
espsr = np.array((h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')['DS1']))
esps = espsr+1j*espsi
esps = np.transpose(esps)
# esps=np.transpose(esps,axes=[1,0,2])
esps = np.rot90(esps,3)
esps = np.fft.fftshift(esps,axes=[0,1])

sensmaps = esps.copy()
sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])
sensmaps = np.rot90(np.rot90(sensmaps))
sensmaps = sensmaps / np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

dd = np.rot90(np.rot90(dd))
ddimc = utils.tFT_with_sensmaps(dd, sensmaps)
dd = dd / np.percentile(np.abs(ddimc).flatten(), 99)
ddimc = utils.tFT_with_sensmaps(dd, sensmaps)
        
# =============================
# define the undersampling factor
# =============================
R = usfact

# ======================================
# undersampling pattern. either load or generate.
# ======================================
try:
     uspat = np.load(basefolder + 'uspats/uspat_realim_us' + str(R) + '_base_' + dirbase[-19:-1].replace("/","_") + '_sli' + str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     uspat = USp.generate_opt_US_pattern_1D(dd.shape[0:2], R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder + 'uspats/uspat_realim_us' + str(R) + '_base_' + dirbase[-19:-1].replace("/","_") + '_sli' + str(sli), uspat)
     print("Generated a new u.s. pattern file")
          
usksp = dd*np.tile(uspat[:,:,np.newaxis], [1, 1, dd.shape[2]])

num_iter = 102
regtype = 'reg2'
reg = 0 # no phase regulization!
dcprojiter = 10
onlydciter = 10 # do firist iterations only SENSE reconstruction
chunks40 = True
mode = 'MRIunproc'
  
# =============================
# do the reconstruction
# =============================          
rec_vae = vaerecon.vaerecon(usksp,
                            sensmaps = sensmaps,
                            dcprojiter = 10,
                            onlydciter = 10,
                            lat_dim = lat_dim,
                            patchsize = ndims,
                            contRec = '',
                            parfact = 25,
                            num_iter = 302,
                            regiter = 10,
                            reglmb = reg,
                            regtype = regtype,
                            half = True,
                            mode = mode,
                            chunks40 = chunks40)

pickle.dump(rec_vae[0], open(basefolder + 'MAPestimation/rec_meas_us' + str(R) + '_sli' + str(sli) + '_regtype_' + regtype + '_dcprojiter_' + str(dcprojiter) ,'wb'))