
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


NUM_COLORS = 10
values = range(NUM_COLORS)

contrast_colormap = plt.get_cmap('Accent') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=contrast_colormap)
#print scalarMap.get_clim()

for idx in range(NUM_COLORS):
    colorVal = scalarMap.to_rgba(values[idx])
    print colorVal

