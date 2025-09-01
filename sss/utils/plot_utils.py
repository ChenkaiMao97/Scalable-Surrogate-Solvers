import numpy as np
import matplotlib 
matplotlib.use('agg')
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm

import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")

try:
	import matplotlib.font_manager as font_manager
	font_dirs = ['/home/chenkaim/fonts/Microsoft_Aptos_Fonts']
	font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
	for font_file in font_files:
		font_manager.fontManager.addfont(font_file)
except:
	pass

plt.rcParams.update({
    'font.size': 16,               # Default font size
    'font.family': 'Aptos',        # Default font family
    # 'font.sans-serif': ['Arial'],  # Specific font for sans-serif
    'axes.titlesize': 24,             # Font size for axes titles
    'axes.labelsize': 20,             # Font size for x and y labels
    'xtick.labelsize': 20,            # Font size for x-axis tick labels
    'ytick.labelsize': 20,            # Font size for y-axis tick labels
    'legend.fontsize': 18,            # Font size for legend
    'figure.titlesize': 28,
    'figure.figsize': (8, 6),
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.0,
})

def MAE_loss(a,b=None):
	if b is None:
		return torch.mean(torch.abs(a))
	else:
		return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def MSE_loss(a,b=None):
	if b is None:
		return torch.mean(torch.abs(a)**2)
	else:
		return torch.mean(torch.abs(a-b)**2)/torch.mean(torch.abs(b)**2)


def plot_helper(data, row, column, titles, path, cmaps=None, center_zero=None, log_scale=None, plot_colobar=True):
	# data is a list of 2d data to be plot
	tot_sub_plot = len(data)
	assert tot_sub_plot <= row*column
	fig = plt.figure(figsize=(column*8, row*8))
	gs = fig.add_gridspec(row, column, hspace=0.4, wspace=0.4)
	
	for i in range(tot_sub_plot):
		plot_data = data[i]
		if isinstance(plot_data, torch.Tensor):
			plot_data = plot_data.detach().cpu().numpy()
		ax = fig.add_subplot(gs[i//column, i%column])
		norm_plot = LogNorm(vmin=1e-15, vmax=1e-5) if (log_scale is not None and log_scale[i]) else None
		if cmaps is not None:
			im = ax.imshow(plot_data, cmap=cmaps[i], norm=norm_plot)
		else:
			im = ax.imshow(plot_data, norm=norm_plot)
		if center_zero is not None:
			if center_zero[i]:
				vmax = np.max(np.abs(plot_data))
				im.set_clim(-vmax, vmax)
		ax.set_xticks([])
		ax.set_yticks([])
		if len(titles) > 0:
			ax.set_title(f"{titles[i]}")
		
		# Create fixed-size colorbar axes
		cax_width = 0.01  # Width of colorbar as fraction of figure width
		cax_spacing = 1e-2  # Space between plot and colorbar
		
		if plot_colobar:
			# Get the position of the subplot
			pos = ax.get_position()
			cax = fig.add_axes([pos.x1 + cax_spacing, pos.y0, cax_width, pos.height])
			cbar = plt.colorbar(im, cax=cax)
			if log_scale is not None and log_scale[i]:
				pass
			else:
				cbar.formatter.set_scientific(True)
				cbar.formatter.set_powerlimits((0,0))
			cbar.ax.yaxis.set_offset_position('left')
			cbar.ax.yaxis.get_offset_text().set_fontsize(36)
			cbar.ax.tick_params(labelsize=36)
			cbar.update_ticks()
	
	plt.tight_layout()
	plt.savefig(path, transparent=True, dpi=100, pad_inches=0.2)
	plt.close()

def setup_plot_data(data, src, pml_th=40, sx_f=None, sy_f=None, eps_max=8):
	colored_yee = np.zeros((data.shape[0], data.shape[1],3))
	# Si_color = np.array([245,112,108], dtype=np.uint8)
	air_color = np.array([249,232,215], dtype=np.uint8)
	pml_color = np.array([255, 185, 0], dtype=np.uint8)
	src_color = np.array([52, 181, 168], dtype=np.uint8)

	top_mat_color = np.array([30, 30, 30], dtype=np.uint8)
	exceed_mat_color = np.array([0, 0, 0], dtype=np.uint8)
	data = np.asarray(data)

	colored_yee = air_color+((data[:,:,None]-1)/(eps_max-1)*(top_mat_color.astype(np.float32)-air_color.astype(np.float32))).astype(np.uint8)
	colored_yee = (data[:,:,None]>eps_max)*exceed_mat_color + (data[:,:,None]<=eps_max)*colored_yee 

	pml = np.zeros((data.shape[0], data.shape[1]))
	if sx_f is not None:
		pml[np.abs(sx_f)>1+1e-6] = 1
		pml[np.abs(sy_f)>1+1e-6] = 1
	else:
		pml[:pml_th, :] = 1
		pml[data.shape[0]-pml_th:, :] = 1
		pml[:,:pml_th] = 1
		pml[:,data.shape[1]-pml_th:] = 1

	pml_alpha = 0.5  # Adjust this value to control transparency (0.0 to 1.0)
	colored_yee = np.where(pml[:,:,None] > 0.5, 
						   (pml_alpha * pml_color + (1 - pml_alpha) * colored_yee).astype(np.uint8), 
						   colored_yee)

	thickened_src = np.abs(src[:,:,None])>1e-8
	thickened_src = thickened_src + np.roll(thickened_src, 1, axis=0) + np.roll(thickened_src, -1, axis=0) + \
	np.roll(thickened_src, 1, axis=1) + np.roll(thickened_src, -1, axis=1)

	colored_yee = (thickened_src>1e-8)*src_color + (thickened_src<1e-8)*colored_yee 

	return colored_yee

def setup_bc_plot(top_bc, bottom_bc, left_bc, right_bc, bc_thickness=1):
	_, sy, _ = top_bc.shape
	sx, _, _ = left_bc.shape

	bc_plot = np.zeros((sx, sy))
	bc_plot[:bc_thickness,:] = top_bc[None, 0, :, 0] # plot real part of top bc
	bc_plot[-bc_thickness:,:] = bottom_bc[None, 0, :, 0] # plot real part of bottom bc
	bc_plot[:,:bc_thickness] = left_bc[:, None, 0, 0] # plot real part of left bc
	bc_plot[:,-bc_thickness:] = right_bc[:, None, 0, 0] # plot real part of right bc

	return bc_plot

def plot_iter_algo(algo_name, idx, num_iter, plot_iter, history, errors, residues, cropped_eps, cropped_source_RI, cropped_Sx_f, cropped_Sy_f, cropped_Sx_b, cropped_Sy_b, cropped_field_RI, residue_gt, this_wl, this_dL, c_shift, save_folder='image_outputs', plot_colobar=True):
	colored_yee = setup_plot_data(cropped_eps[0,:,:], torch.view_as_complex(cropped_source_RI[0,:,:]), sx_f=cropped_Sx_f[0,:,:], sy_f=cropped_Sy_f[0,:,:])
	plot_data = [colored_yee,cropped_source_RI[0,:,:,0]+cropped_source_RI[0,:,:,1],cropped_Sx_f[0,:,:].imag+cropped_Sy_f[0,:,:].imag,cropped_Sx_b[0,:,:].imag+cropped_Sy_b[0,:,:].imag,cropped_field_RI[0,:,:,0], residue_gt[0,:,:,0]]

	print(this_wl, this_dL)
	mean_size = np.mean(cropped_eps[0,:,:].numpy())**.5*this_dL*64/this_wl
	plot_title = [f'eps, max:{np.max(cropped_eps[0,:,:].numpy()):.2f}\nmean size:{mean_size:.2f}', 'source', 'Sx_f+Sy_f', 'Sx_b+Sy_b', 'gt', 'residue_gt']
	cmaps = [None, None, None, None, 'seismic', 'Reds']
	row, column = 3, 2
	plot_helper(plot_data,row,column,plot_title,f"{save_folder}/{algo_name}_setup_{idx}.png", cmaps=cmaps, plot_colobar=plot_colobar)

	plot_data = [f[0,:,:,0] for f in history] + [e[0,:,:,0] for e in errors] + [torch.log(torch.abs(torch.view_as_complex(r[0,:,:,:]))+1e-6)/torch.log(torch.tensor(10)[None, None]) for r in residues] 
	plot_title = ['init', *[f'iter {i*plot_iter}' for i in range(1,num_iter//plot_iter+1)]] + [f'iter {i*plot_iter}\nL1 loss: {MAE_loss(history[i], cropped_field_RI):.1e}' for i in range(num_iter//plot_iter+1)] + [f'residue' for i in range(num_iter//plot_iter+1)]
	cmaps = ['seismic']*len(history) + ['seismic']*len(errors) + ['Reds']*len(residues)
	center_zero = [True]*len(history) + [True]*len(errors) + [False]*len(residues)
	log_scale = [False]*len(history) + [False]*len(errors) + [True]*len(residues)
	row, column = 3, num_iter//plot_iter+1
	plot_helper(plot_data,row,column,plot_title,f"{save_folder}/{algo_name}_output_{idx}_wl_{this_wl:.2e}_dL_{this_dL:.2e}_cshift_{c_shift}.png", cmaps=cmaps, center_zero=center_zero, log_scale=log_scale, plot_colobar=plot_colobar)


# plot helper for 2d test cases:
# def plot_2d(data, fname=None, stride = 1, my_cmap = plt.cm.binary, cm_zero_center=True, title=None):
#     """Plot 2D slices of volumetric data.
    
#     Args:
#         data: 2D numpy array of shape (sx, sy)
#         fname: Output filename
#     """
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)
    
#     # Get data dimensions
#     sx, sy = data.shape
#     if cm_zero_center:
#         vm = max(np.max(data), -np.min(data))
#         norm = Normalize(vmin=-vm, vmax=vm)
#     else:
#         norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        
#     # Plot the data
#     ax.imshow(data, cmap=my_cmap, norm=norm)
    
#     # Add colorbar
#     fig.colorbar(ax.imshow(data, cmap=my_cmap, norm=norm), ax=ax, shrink=0.5, aspect=5)
    
#     if title:
#         plt.title(title)
        
#     plt.savefig(fname, dpi=100, bbox_inches='tight')
#     plt.close()
