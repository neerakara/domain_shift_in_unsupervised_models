import pickle
import numpy as np
import utils
import h5py
import MR_image_data

# basefolder
basefolder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/'

# parameters
dataset = 'in_house'

if dataset == 'in_house':    
    reglambda = 0
    regtype = 'reg2'
    dcprojiter = 10
    
    subject = 'NK/'
    R = 3
    sli = 3 
    n1 = 5 # number of updates of the normalization module
    n2 = 5 # number of updates of the image
    
    # read original image
    dirbase = basefolder + 'data/the_h5_files/' + subject
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
    dd = utils.tFT_with_sensmaps(dd, sensmaps)
    
    orig_image = np.abs(dd)
    
elif dataset == 'hcp':
    reglambda = 0.1
    regtype = 'reg2_dc'
    dcprojiter = 10

    R = 4    
    vol = 5
    sli = 150 
    subject = 'subject' + str(vol) + '/'

    # read original image    
    MRi = MR_image_data(dirname = basefolder + 'data/',
                        imgSize = [260, 311, 260],
                        testchunks = [39],
                        noiseinvstd = 0)
    
    orim = MRi.center_crop(MRi.get_image(39, vol, sli),
                           252,
                           308,
                           252,
                           308)

    orig_image = np.abs(orim)
    
# load pickle
results_folder = basefolder + 'code/v2.0/results/' + dataset + '/' + subject
recon_suffix = 'us' + str(R) + '_sli' + str(sli) + '_regtype_' + regtype + '_reglambda_' + str(reglambda) + '_n1_' + str(n1) +  '_n2_' + str(n2)  + '_accum_grads_norm_k_1'
rec = pickle.load(open(results_folder + recon_suffix, 'rb'))

# save figure
rec1 = np.reshape(rec, [orig_image.shape[0], orig_image.shape[1], rec.shape[-1]])
lastiter = int((np.floor(rec1.shape[-1] / 13) - 2) * 13)
rec1 = rec1[:, :, :lastiter]
utils.save_recon_results(np.abs(rec1), 
                         orig_image,
                         results_folder + recon_suffix + '.tiff')