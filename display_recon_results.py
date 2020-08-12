import pickle
import numpy as np
import utils

# basefolder
basefolder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/'

# parameters
dataset = 'hcp'

if dataset == 'in_house':    
    imgsizex = 237
    imgsizey = 256
    reglambda = 0
    regtype = 'reg2'
    dcprojiter = 10
    
    subject = 'sess_02_07_2018/CK/'
    R = 2
    sli = 3 
    
elif dataset == 'hcp':
    imgsizex = 252
    imgsizey = 308
    reglambda = 0.1
    regtype = 'reg2_dc'
    dcprojiter = 10
    
    subject = 'subject5/'
    R = 4
    sli = 150 
    
# load pickle
results_folder = basefolder + 'code/v2.0/results/' + dataset + '/' + subject
rec_suffix = 'us' + str(R) + '_sli' + str(sli) + '_regtype_' + regtype + '_reglambda_' + str(reglambda) + '_dcprojiter_' + str(dcprojiter)
rec = pickle.load(open(results_folder + rec_suffix, 'rb'))

# save figure
rec1 = np.reshape(rec, [imgsizex, imgsizey, rec.shape[-1]])
utils.save_recon_results(np.abs(rec1), results_folder + rec_suffix + '.png')