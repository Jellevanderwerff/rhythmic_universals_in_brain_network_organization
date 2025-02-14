from nltools import Brain_Data
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt

desikan_killiany = Brain_Data('https://github.com/neurodata/neuroparc/raw/master/atlases/label/Human/Desikan_space-MNI152NLin6_res-1x1x1.nii.gz').to_nifti()
orthoslicer = plotting.plot_roi(desikan_killiany, title='Desikan-Killiany', cmap='Grays', colorbar=True, display_mode='x', draw_cross=False)
orthoslicer.savefig('desikan_gray.pdf')
orthoslicer = plotting.plot_roi(desikan_killiany, title='Desikan-Killiany', colorbar=True, display_mode='x', draw_cross=False)
orthoslicer.savefig('desikan_colour.pdf')
orthoslicer = plotting.plot_roi(desikan_killiany, title='Desikan-Killiany', cmap='Grays', colorbar=True, display_mode='x', draw_cross=False, view_type='contours')
orthoslicer.savefig('desikan_contours.pdf')
