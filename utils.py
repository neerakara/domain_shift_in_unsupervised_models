import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# ==================================
# fourier transform functions
# ==================================
# fft2 computes the 2-dimensional discrete Fourier Transform
# fftshift shifts the zero-frequency component to the center of the spectrum.
# ifft2 and ifftshift are inverse of fft2 and fftshift respectively.

# ======================
# compute ft
# ======================
def FT(x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.fftshift(np.fft.fft2(x, axes=(0,1)), axes=(0,1)) / np.sqrt(252*308) # size of HCP images
     else:
          return np.fft.fftshift(np.fft.fft2(x, axes=(0,1)), axes=(0,1)) 

# ======================
# compute inverse ft
# ======================
def tFT(x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.ifft2(np.fft.ifftshift(x, axes=(0,1)), axes=(0,1)) * np.sqrt(252*308)
     else:
          return np.fft.ifft2(np.fft.ifftshift(x, axes=(0,1)), axes=(0,1))   

# ======================
# compute FFT and multiply with the undersampling mask
# ======================
def UFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]     
     return uspat*FT(x, normalize)

def tUFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     return  tFT(uspat*x, normalize)

# ======================
# ======================
def calc_rmse_percentage(rec, imorig):
     return 100 * np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))))
 
def calc_rmse(rec, imorig):
     return np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))))

# ======================
# FT functions with sensmaps
# ======================
# encoding operation of MR: multiplication with the sensitivity map, followed by FT
def FT_with_sensmaps(x, sensmaps):
     #inp: [nx, ny]
     #out: [nx, ny, ns]
     return np.fft.fftshift(np.fft.fft2(sensmaps*np.tile(x[:,:,np.newaxis], [1,1,sensmaps.shape[2]]), axes=(0,1)), axes=(0,1)) #  / np.sqrt(x.shape[0]*x.shape[1])

def tFT_with_sensmaps(x, sensmaps):
     #inp: [nx, ny, ns]
     #out: [nx, ny]    
     temp = np.fft.ifft2(np.fft.ifftshift(x, axes=(0,1)), axes=(0,1))
     return np.sum(temp * np.conjugate(sensmaps), axis=2) / np.sum(sensmaps * np.conjugate(sensmaps), axis=2) #  * np.sqrt(x.shape[0]*x.shape[1])

def UFT_with_sensmaps(x, uspat, sensmaps):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny, ns]     
     return np.tile(uspat[:,:,np.newaxis], [1, 1, sensmaps.shape[2]]) * FT_with_sensmaps(x, sensmaps)

def tUFT_with_sensmaps(x, uspat, sensmaps):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     tmp1 = np.tile(uspat[:,:,np.newaxis], [1, 1, sensmaps.shape[2]])
     return tFT_with_sensmaps(tmp1*x, sensmaps)
 
# ==========================================================
# ==========================================================       
def save_recon_results(rec,
                       orig,
                       savepath):
    
    num_images = rec.shape[-1]
    ids = np.arange(0, num_images, num_images // 8)
    nc = len(ids)
    nr = 3
            
    plt.figure(figsize=[5*nc, 5*nr])
    for c in range(nc): 
        
        r = np.rot90(rec[:,:,ids[c]], k=1)
        o = np.rot90(orig, k=1)
        e = r-o
        
        rmse_percent = np.round(calc_rmse_percentage(r*np.linalg.norm(o)/np.linalg.norm(r),o), 3)
        rmse = np.round(calc_rmse(r*np.linalg.norm(o)/np.linalg.norm(r),o), 3)
        error_str = 'rmse: ' + str(rmse) + ', ' + str(rmse_percent) + '%'
        
        plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(r, cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('iteration' + str(ids[c]))        
        plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(o, cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('orig')
        plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(e, cmap='gray'); plt.clim([-1.0,1.0]); plt.colorbar(); plt.title('error, ' + error_str)
        
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def save_single_image(image,
                      savepath):
        
    plt.figure(figsize=[5,5])            
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(savepath, bbox_inches='tight', dpi=50)
    plt.close()