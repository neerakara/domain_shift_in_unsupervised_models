# ===============================================================================================================
	# FOR NEERAV:
# ===============================================================================================================
	# 1. I would do your optimization here as x* = max_phi ELBO(N_phi(|x|)) with x as the current estimate of the image, i.e. recs[:,it]
	# then reform the complex image as x_new = x* * np.exp(1i*p.angle(x)) and do the rest of the optimization.
	#
	# 2. the second output here is the value of ELBO: ftot, f_lik, f_dc = feval(recs[:,it+1]).
	# I called f_lik for some other reasons, but it is the ELBO actually.
	#
	# 3. The confusing part is that the VAE works with patches, but you want to do the N_phi on the whole image.
# I am not sure, but the only possible way seems to be to implement the patching operation in tensorflow,
# if you want to be able to pass the gradients through it.
	# Alternatively, you can use the full image size VAE, then it becomes nearly trivial...
# ===============================================================================================================
from __future__ import division
from __future__ import print_function
import numpy as np
from Patcher import Patcher
from definevae import definevae
import scipy.optimize as sop
import SimpleITK as sitk
import utils
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')     

## direk precision optimize etmek daha da iyi olabilir. 
def vaerecon(us_ksp_r2, # undersampled k space
             sensmaps,
             dcprojiter,
             onlydciter = 0,
             lat_dim = 60,
             patchsize = 28,
             contRec = '',
             parfact = 10,
             num_iter = 302,
             rescaled = False,
             half = False,
             regiter = 15,
             reglmb = 0.05,
             regtype = 'TV',
             usemeth = 1,
             stepsize = 1e-4,
             optScale = False,
             mode = [],
             chunks40 = False,
             Melmodels = '',
             N4BFcorr = False,
             z_multip = 1.0,
             n1 = 5,
             n2 = 5):
     
     logging.info('xxxxxxxxxxxxxxxxxxx contRec is ' + contRec)
     logging.info('xxxxxxxxxxxxxxxxxxx parfact is ' + str(parfact) )
     
     # ==============================================================================
     # set parameters
     # ==============================================================================
     np.random.seed(seed = 1)     
     imsizer = us_ksp_r2.shape[0] #252#256#252
     imrizec = us_ksp_r2.shape[1] #308#256#308
     nsampl = 50 #0
          
     # ==============================================================================
     # make a network and a patcher to use later
     # ==============================================================================     
     # =================================
     # get the output from the VAE
     # funop: ELBO
     # grd0: gradient of the ELBO wrt the image
     # grd_p_x_z0, grd_p_z0, grd_q_z_x0: different gradients
     # =================================
     vae_outputs = definevae(lat_dim = lat_dim,
                             patchsize = patchsize,
                             batchsize = parfact*nsampl,
                             rescaled = rescaled,
                             half = half,
                             mode = mode,
                             chunks40 = chunks40,
                             Melmodels = Melmodels,
                             use_normalizer = bool(n1>0))
     
     x_rec, x_inp, funop, grd0, sess = vae_outputs[0:5]
     grd_p_x_z0, grd_p_z0, grd_q_z_x0, grd20 = vae_outputs[5:9] # these are not being used in the code now.
     y_out, y_out_prec, z_std_multip, op_q_z_x = vae_outputs[9:13] # these are not being used in the code now.
     mu, std = vae_outputs[13:15] # these are not being used in the code now.
     grd_q_zpl_x_az0, op_q_zpl_x, z_pl = vae_outputs[15:18] # these are not being used in the code now.
     z = vae_outputs[18:19] # these are not being used in the code now.
     # Most of these outputs were being used in the function likelihood_grad_meth3, which is no longer being used.
     norm_accum_grads_zero_op, norm_accum_grads_op, norm_accum_grads_mean_op, num_accum_steps_pl, norm_update_op, x_norm = vae_outputs[19:25]
     
     # =================================
     # used to go from image to patches and back
     # =================================
     Ptchr = Patcher(imsize = [imsizer, imrizec],
                     patchsize = patchsize,
                     step = int(patchsize/2),
                     nopartials = True,
                     contatedges = True)
     
     nopatches = len(Ptchr.genpatchsizes)
     logging.info("There will be in total " + str(nopatches) + " patches.")
     
     # =================================
     # functions for data consistency
     # =================================
     def dconst(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          return np.linalg.norm(utils.UFT_with_sensmaps(us, uspat, sensmaps) - data)**2
     
     def dconst_grad(us):
          #inp: [nx, ny]
          #out: [nx, ny]
          return 2 * utils.tUFT_with_sensmaps(utils.UFT_with_sensmaps(us, uspat, sensmaps) - data, uspat, sensmaps)

     # =================================  
     # function for running the normalization op on a batch of patches
     # =================================  
     def update_normalizer(im,
                           sess,
                           accum_gradients_zero_op,
                           accum_gradients_op,
                           accum_gradients_mean_op,
                           num_accum_steps_pl,
                           update_normalizer_op):
         
          # make patches
          ptchs = Ptchr.im2patches(np.reshape(im, [imsizer, imrizec]))
          ptchs = np.array(ptchs)          
          ptchs = np.abs(ptchs)
          
          # zero accumulated gradients
          sess.run(accum_gradients_zero_op)
          num_accumulation_steps = 0
          
          # convert image to patches
          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1])
          # extra indices that have to be padded to ensure we have complete batches for the VAE
          extraind = int(np.ceil(ptchs.shape[0] / parfact) * parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs, ((0, extraind), (0,0)), mode = 'edge')
          
          # send batches of patches and accumulate gradients
          for ix in range(int(np.ceil(ptchs.shape[0] / parfact))):
              batch_of_patches = ptchs[parfact * ix : parfact * ix + parfact, :]
              sess.run(accum_gradients_op, feed_dict = {x_rec: np.tile(batch_of_patches, (nsampl, 1)), z_std_multip: z_multip})
              num_accumulation_steps = num_accumulation_steps + 1
              
          # run op to mean gradients accumulated so far
          sess.run(accum_gradients_mean_op, feed_dict = {num_accum_steps_pl: num_accumulation_steps})
          
          # update normalizer parameters according to the mean gradient.
          # The stuff passed in the feed_dict does not matter in this line, but something needs to be passed. So passing the last batch of patches.
          sess.run(update_normalizer_op, feed_dict = {x_rec: np.tile(batch_of_patches, (nsampl, 1)), z_std_multip: z_multip})
          
          return 0     
     
     # =================================
     # functions for computing the ELBO and its derivatives
     # =================================     
     def likelihood(us):
          # inp: [parfact,ps*ps]
          # out: parfact
          us = np.abs(us)
          funeval = funop.eval(feed_dict={x_rec: np.tile(us, (nsampl,1)), z_std_multip: z_multip})
          # funeval: [500x1]
          funeval = np.array(np.split(funeval, nsampl, axis=0)) # [nsampl x parfact x 1]
          return np.mean(funeval, axis=0).astype(np.float64)
     
     def likelihood_grad(us):
          # inp: [parfact, ps*ps]
          # out: [parfact, ps*ps]
          usc = us.copy()
          usabs = np.abs(us)
          grd0eval = grd0.eval(feed_dict = {x_rec: np.tile(usabs, (nsampl, 1)), z_std_multip: z_multip}) # grd0eval: [500x784]
          grd0eval = np.array(np.split(grd0eval, nsampl, axis=0)) # [nsampl x parfact x 784]
          grd0m = np.mean(grd0eval,axis=0) # [parfact,784]
          # correction for the complex image
          grd0m = usc/np.abs(usc)*grd0m
          return grd0m #.astype(np.float64)
          
     def likelihood_grad_patches(ptchs):
          # inp: [np, ps, ps] 
          # out: [np, ps, ps] 
          # takes set of patches as input and returns a set of their grad.s 
          # both grads are in the positive direction
          
          shape_orig = ptchs.shape
          ptchs = np.reshape(ptchs, [ptchs.shape[0], -1] )
          grds = np.zeros([int(np.ceil(ptchs.shape[0]/parfact)*parfact), np.prod(ptchs.shape[1:])], dtype = np.complex64)
          
          # extra indices that have to be padded to ensure we have complete batches for the VAE
          extraind = int(np.ceil(ptchs.shape[0] / parfact) * parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs, ((0, extraind), (0,0)), mode='edge')
          
          for ix in range(int(np.ceil(ptchs.shape[0] / parfact))):
              grds[parfact*ix : parfact*ix+parfact, :] = likelihood_grad(ptchs[parfact*ix : parfact*ix+parfact, :]) 

          grds = grds[0:shape_orig[0],:]

          return np.reshape(grds, shape_orig)
     
     def likelihood_patches(ptchs):
          # inp: [np, ps, ps] 
          # out: 1
          
          fvls = np.zeros([int(np.ceil(ptchs.shape[0] / parfact) * parfact) ])
          extraind = int(np.ceil(ptchs.shape[0] / parfact) * parfact) - ptchs.shape[0]
          ptchs = np.pad(ptchs, [(0,extraind), (0,0), (0,0)], mode='edge')
          
          for ix in range(int(np.ceil(ptchs.shape[0]/parfact))):
               fvls[parfact*ix : parfact*ix+parfact] = likelihood(np.reshape(ptchs[parfact*ix : parfact*ix+parfact,:,:], [parfact,-1]))
               
          fvls = fvls[0:ptchs.shape[0]]
               
          return np.mean(fvls)

     # =================================
     # returns the gradient of the ELBO wrt the image 
     # =================================     
     def full_gradient(image):
          #inp: [nx*nx, 1]
          #out: [nx, ny], [nx, ny]
          #returns both gradients in the respective positive direction.
          #i.e. must 
          
          # convert image to patches
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs = np.array(ptchs)          
          
          # compute gradient of the ELBO wrt each patch
          grd_lik = likelihood_grad_patches(ptchs)
          grd_lik = (-1)* Ptchr.patches2im(grd_lik)
          
          grd_dconst = dconst_grad(np.reshape(image, [imsizer,imrizec]))
          
          return grd_lik + grd_dconst, grd_lik, grd_dconst
     
     def full_funceval(image):
          #inp: [nx*nx, 1]
          #out: [1], [1], [1]
          
          tmpimg = np.reshape(image, [imsizer,imrizec])
          dc = dconst(tmpimg)     
          ptchs = Ptchr.im2patches(np.reshape(image, [imsizer,imrizec]))
          ptchs = np.array(ptchs)          
          lik = (-1)*likelihood_patches(np.abs(ptchs))    
          
          return lik + dc, lik, dc    
     
     # =================================
     # phase projection functions
     # =================================     
     def tv_proj(phs,mu=0.125,lmb=2,IT=225):
          phs = fb_tv_proj(phs,mu=mu,lmb=lmb,IT=IT)          
          return phs
     
     def fgrad(im):
          imr_x = np.roll(im,shift=-1,axis=0)
          imr_y = np.roll(im,shift=-1,axis=1)
          grd_x = imr_x - im
          grd_y = imr_y - im          
          return np.array((grd_x, grd_y))
     
     def fdivg(im):
          imr_x = np.roll(np.squeeze(im[0,:,:]),shift=1,axis=0)
          imr_y = np.roll(np.squeeze(im[1,:,:]),shift=1,axis=1)
          grd_x = np.squeeze(im[0,:,:]) - imr_x
          grd_y = np.squeeze(im[1,:,:]) - imr_y          
          return grd_x + grd_y
     
     def f_st(u,lmb):          
          uabs = np.squeeze(np.sqrt(np.sum(u*np.conjugate(u),axis=0)))          
          tmp = 1 - lmb/uabs
          tmp[np.abs(tmp) < 0] = 0             
          uu = u*np.tile(tmp[np.newaxis,:,:],[u.shape[0],1,1])          
          return uu
       
     def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
          sz = im.shape
          us=np.zeros((2,sz[0],sz[1],IT))
          us[:,:,:,0] = u0          
          for it in range(IT-1):               
               # grad descent step:
               tmp1 = im - fdivg(us[:,:,:,it])
               tmp2 = mu*fgrad(tmp1)
               tmp3 = us[:,:,:,it] - tmp2                 
               # thresholding step:
               us[:,:,:,it+1] = tmp3 - f_st(tmp3, lmb=lmb)                    
          return im - fdivg(us[:,:,:,it+1]) 
          
     def tikh_proj(usph, niter=100, alpha=0.05):          
          ims = np.zeros((imsizer, imrizec, niter))
          ims[:,:,0] = usph.copy()
          for ix in range(niter-1):
              ims[:,:,ix+1] = ims[:,:,ix] + alpha*2*fdivg(fgrad(ims[:,:,ix]))
          return ims[:,:,-1]
     
     def reg2_proj(usph, niter=100, alpha=0.05):          
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          usph = usph + np.pi
          ims = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())

          for ix in range(niter-1):
              ims[:,:,ix+1] = ims[:,:,ix] +alpha*reg2grd(ims[:,:,ix].flatten()).reshape([252,308]) # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              regval = reg2eval(ims[:,:,ix+1].flatten())
          
          return ims[:,:,-1] - np.pi    
     
     def reg2_dcproj(usph, magim, bfestim, niter=100, alpha_reg=0.05, alpha_dc=0.05):
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          # usph = usph+np.pi          
          ims = np.zeros((imsizer,imrizec,niter))
          grds_reg = np.zeros((imsizer,imrizec,niter))
          grds_dc = np.zeros((imsizer,imrizec,niter))
          ims[:,:,0]=usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          
          for ix in range(niter-1):
               
              grd_reg = reg2grd(ims[:,:,ix].flatten()).reshape([252,308])  # *alpha*np.real(1j*np.exp(-1j*ims[:,:,ix])*    fdivg(fgrad(np.exp(  1j* ims[:,:,ix]    )))     )
              grds_reg[:,:,ix]  = grd_reg
              grd_dc = reg2_dcgrd(ims[:,:,ix].flatten() , magim, bfestim).reshape([252,308])
              grds_dc[:,:,ix]  = grd_dc
              
              ims[:,:,ix+1] = ims[:,:,ix] + alpha_reg*grd_reg  - alpha_dc*grd_dc
              regval = reg2eval(ims[:,:,ix+1].flatten())
              f_dc = dconst(magim*np.exp(1j*ims[:,:,ix+1])*bfestim)
              
              logging.info("norm grad reg: " + str(np.linalg.norm(grd_reg)))
              logging.info("norm grad dc: " + str(np.linalg.norm(grd_dc)))
              logging.info("regval: " + str(regval))
              logging.info("fdc: (*1e9) {0:.6f}".format(f_dc/1e9))
          
          return ims[:,:,-1] #-np.pi    
     
     def reg2eval(im):
          # takes in 1d, returns scalar
          im = im.reshape([252,308])
          phs = np.exp(1j*im)
          return np.linalg.norm(fgrad(phs).flatten())
     
     def reg2grd(im):
          # takes in 1d, returns 1d
          im = im.reshape([252,308])
          return -2*np.real(1j*np.exp(-1j*im) * fdivg(fgrad(np.exp(1j * im)))).flatten()
     
     def reg2_dcgrd(phim, magim, bfestim):   
          # takes in 1d, returns 1d
          phim = phim.reshape([252,308])
          magim = magim.reshape([252,308])
          tmp = utils.UFT_with_sensmaps(bfestim * np.exp(1j * phim) * magim, uspat, sensmaps) - data
          return -2 * np.real(1j * np.exp(-1j*phim) * magim * bfestim * utils.tUFT_with_sensmaps(tmp, uspat, sensmaps)).flatten()
     
     def reg2_proj_ls(usph, niter=100):
          # from  Separate Magnitude and Phase Regularization via Compressed Sensing,  Feng Zhao
          # with line search         
          usph = usph + np.pi
          ims = np.zeros((imsizer, imrizec, niter))
          ims[:,:,0] = usph.copy()
          regval = reg2eval(ims[:,:,0].flatten())
          logging.info(regval)
          for ix in range(niter-1):               
              currgrd = reg2grd(ims[:,:,ix].flatten())     
              res = sop.minimize_scalar(lambda alpha: reg2eval(ims[:,:,ix].flatten() + alpha * currgrd   ), method='Golden')
              alphaopt = res.x
              logging.info("optimal alpha: " + str(alphaopt) )               
              ims[:,:,ix+1] = ims[:,:,ix] + alphaopt*currgrd.reshape([252,308])
              regval = reg2eval(ims[:,:,ix+1].flatten())
              logging.info("regval: " + str(regval))             
          return ims[:,:,-1]-np.pi 

     def N4corrf(im):
          phasetmp = np.angle(im)
          ddimcabs = np.abs(im)
          inputImage = sitk.GetImageFromArray(ddimcabs, isVector=False)
          corrector = sitk.N4BiasFieldCorrectionImageFilter();
          inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
          output = corrector.Execute(inputImage)
          N4biasfree_output = sitk.GetArrayFromImage(output)          
          n4biasfield = ddimcabs/(N4biasfree_output+1e-9)
          
          if np.isreal(im).all():
               return n4biasfield, N4biasfree_output 
          else:
               return n4biasfield, N4biasfree_output*np.exp(1j*phasetmp)
     
     # ===============================
     # make the data
     # ===============================     
     uspat = np.abs(us_ksp_r2) > 0
     uspat = uspat[:,:,0]
     data = us_ksp_r2
     trpat = np.zeros_like(uspat)
     trpat[:, 120:136] = 1
          
     logging.info(uspat)
     
     # ===================================== 
     # initialize counters
     # ===================================== 
     numiter = num_iter
     multip = 0 #0.1
     alphas = stepsize*np.ones(numiter) # np.logspace(-4,-4,numiter)
     
     # ===================================== 
     # functions for POCS
     # ===================================== 
     def feval(im):
          return full_funceval(im)
     
     def geval(im):
          t1, t2, t3 = full_gradient(im)
          return np.reshape(t1,[-1]), np.reshape(t2,[-1]), np.reshape(t3,[-1])
     
     # =====================================
     # initialize data
     # =====================================
     recs = np.zeros((imsizer*imrizec, numiter+30), dtype=complex) 
     phaseregvals = []
     n4biasfields = []
     
     # =====================================
     # first image in the 'recs' list is the initial undersampled image
     # =====================================
     recs[:, 0] = utils.tUFT_with_sensmaps(data, uspat, sensmaps).flatten().copy() 
     
     if N4BFcorr:
          n4bf, N4bf_image = N4corrf( np.reshape(recs[:,0],[imsizer,imrizec]) )
          recs[:,0] = N4bf_image.flatten()
     else:
          n4bf = 1
               
     # =====================================
     # If the optimization is to be continued from a previous run,
     # load the final image from the previous optimization as the first image for the current optimization.
     # =====================================
     logging.info('contRec is ' + contRec)
     if contRec != '':
          try:
               logging.info('Reading from a previous pickle file ' + contRec)
               rr = pickle.load(open(contRec, 'rb'))
               recs[:, 0] = rr[:, -1]
               logging.info('Initialized to the previous recon from pickle: ' + contRec)
          except:
               logging.info('Reading from a previous numpy file ' + contRec)
               rr = np.load(contRec)
               recs[:, 0] = rr[:, -1]
               logging.info('Initialized to the previous recon from numpy: ' + contRec)
     
     # =====================================
     # main loop
     # we don't do GD, we do POCS (projection ontol convex sets)
     #     o. N1 times gradient updates for min_|phi| -ELBO(|x|)
     #     a. N2 times gradient updates for min_|x| -ELBO(|x|)
     #     b. do data consistency projection into a set of x such that || Ex - y ||_2^2 = 0.
     #     c. N gradient updates for the phase image ps: min_px
     # =====================================
     for it in range(0, numiter-1, 13):         
        
          alpha = alphas[it]
          
          # ===============================================
          # first do N1 times magnitude prior iterations wrt normalization module
          # ===============================================      
          recstmp = recs[:, it].copy()          
          for ix in range(n1):
          
               update_normalizer(recstmp,
                                 sess,
                                 norm_accum_grads_zero_op,
                                 norm_accum_grads_op,
                                 norm_accum_grads_mean_op,
                                 num_accum_steps_pl,
                                 norm_update_op)
                              
               ftot, f_lik, f_dc = feval(recstmp)
          
               if N4BFcorr:
                    f_dc = dconst(recstmp.reshape([imsizer, imrizec]) * n4bf)
               
               gtot, g_lik, g_dc = geval(recstmp) # gradient evaluation: total, wrt_vae, wrt_data_consistency
               
               logging.info("updating normalization module...")
               logging.info("iteration number: " + str(it+ix))
               logging.info("f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + " f_dc (1e6)= " + str(f_dc/1e6))
               logging.info("|g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )

               # recstmp will not change in this loop. Only the normalization module will.
               recs[:, it+ix+1] = recstmp.copy()
               
          # ===============================================
          # now do N2 times magnitude prior iterations wrt the image itself
          # ===============================================      
          recstmp = recs[:, it + n1].copy()          
          for ix in range(n2):
          
               ftot, f_lik, f_dc = feval(recstmp)
          
               if N4BFcorr:
                    f_dc = dconst(recstmp.reshape([imsizer, imrizec]) * n4bf)
               
               gtot, g_lik, g_dc = geval(recstmp) # gradient evaluation : total, wrt_vae, wrt_data_consistency
               
               logging.info("updating image...")
               logging.info("iteration number: " + str(it+n1+ix))
               logging.info("f_tot= " + str(ftot) + " f_lik= " + str(f_lik) + " f_dc (1e6)= " + str(f_dc/1e6))
               logging.info("|g_lik|= " + str(np.linalg.norm(g_lik)) + " |g_dc|= " + str(np.linalg.norm(g_dc)) )
     
               # recstmp will change in this loop. The normalization module will stay fixed.          
               recstmp = recstmp - alpha * g_lik # g_lik
               recs[:, it + ix + n1 + 1] = recstmp.copy()
     
          # =============================================== 
          # Now do a  DC projection.... ACTUALLY, skip the DC projection for now.
          # ===============================================
          recs[:, it + n1 + n2 + 1] = recs[:, it + n1 + n2] 
           
          # ===============================================
          # now do a phase projection
          # ===============================================
          tmpa = np.abs(np.reshape(recs[:, it + n1 + n2 + 1], [imsizer, imrizec]))
          tmpp = np.angle(np.reshape(recs[:, it + n1 + n2 + 1], [imsizer, imrizec]))
          tmpatv = tmpa.copy().flatten()
           
          if reglmb == 0:
               logging.info("skipping phase proj")
               tmpptv = tmpp.copy().flatten()
               
          else:
               if regtype == 'TV': # Total variation
                    tmpptv = tv_proj(tmpp, mu=0.125,lmb=reglmb,IT=regiter).flatten() # 0.1, 15

               elif regtype == 'reg2': # Tikhonov
                    tmpptv = reg2_proj(tmpp, alpha=reglmb, niter=100).flatten() # 0.1, 15
                    regval = reg2eval(tmpp)
                    phaseregvals.append(regval)
                    logging.info("KCT-dbg: phase reg value is " + str(regval))
                
               elif regtype == 'reg2_dc': # Tikhonov, with additional constraint from data consistency
                    tmpptv = reg2_dcproj(tmpp, tmpa, n4bf, alpha_reg=reglmb, alpha_dc=reglmb, niter=100).flatten()
                
               elif regtype == 'abs':
                    tmpptv=np.zeros_like(tmpp).flatten()
               
               elif regtype == 'reg2_ls':
                    tmpptv = reg2_proj_ls(tmpp, niter=regiter).flatten() #0.1, 15
                    regval = reg2eval(tmpp)
                    phaseregvals.append(regval)
                    logging.info("KCT-dbg: phase reg value is " + str(regval))
                
               else:
                    logging.info("hey mistake!!!!!!!!!!")
           
          # recombine magnitude and updated phase.
          recs[:, it + n1 + n2 + 2] = tmpatv*np.exp(1j*tmpptv)

          # ===============================================      
          # now do a data consistency projection
          # take the measured part of the k space from the measured data and the remaining part of the k space from the updated image.
          # ===============================================
          if not N4BFcorr:  
               tmp1 = utils.UFT_with_sensmaps(np.reshape(recs[:, it + n1 + n2 + 2], [imsizer,imrizec]), (1-uspat), sensmaps)
               tmp2 = utils.UFT_with_sensmaps(np.reshape(recs[:, it + n1 + n2 + 2], [imsizer,imrizec]), (uspat), sensmaps)
               tmp3 = data * uspat[:,:,np.newaxis]
               
               tmp = tmp1 + multip * tmp2 + (1 - multip) * tmp3
               recs[:, it + n1 + n2 + 3] = utils.tFT_with_sensmaps(tmp, sensmaps).flatten()
               
               ftot, f_lik, f_dc = feval(recs[:, it + n1 + n2 + 3])
               logging.info('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc/np.linalg.norm(data)**2))
               
          elif N4BFcorr:               
               
               n4bf_prev = n4bf.copy()
               
               imgtmp = np.reshape(recs[:, it + n1 + n2 + 2], [imsizer,imrizec]) # biasfree
               
               imgtmp_bf = imgtmp * n4bf_prev # img with bf
               
               n4bf, N4bf_image = N4corrf(imgtmp_bf) # correct the bf, this correction is supposed to be better now.
               
               imgtmp_new = imgtmp * n4bf
               
               n4biasfields.append(n4bf)
               
               tmp1 = utils.UFT_with_sensmaps(imgtmp_new, (1-uspat), sensmaps)
               tmp3 = data * uspat[:, :, np.newaxis]
               tmp = tmp1 + (1 - multip) * tmp3 # multip=0 by default
               recs[:, it + n1 + n2 + 3] = (utils.tFT_with_sensmaps(tmp, sensmaps) / n4bf).flatten()
               
               ftot, f_lik, f_dc = feval(recs[:, it + n1 + n2 + 3])

               if N4BFcorr:
                     f_dc = dconst(recs[:, it + n1 + n2 + 3].reshape([imsizer,imrizec])*n4bf)
               
               logging.info('f_dc (1e6): ' + str(f_dc/1e6) + '  perc: ' + str(100*f_dc / np.linalg.norm(data) ** 2))
              
     return recs, 0, phaseregvals, n4biasfields