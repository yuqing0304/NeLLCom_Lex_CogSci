import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ast
import colorspacious as cs

def cielab_to_rgb(lab):
    """Convert CIELAB to sRGB (0-1 range)."""
    rgb = cs.cspace_convert(lab, start="CIELab", end="sRGB1")
    rgb = np.clip(rgb, 0, 1)  # Ensure values are within valid RGB range
    return rgb

def visualize_cielab_colors(ax, cielab_colors, informativeness, word):
    l_values, a_values, b_values = zip(*cielab_colors)
    cielab_colors_array = np.array(cielab_colors)

    rgb_colors = cielab_to_rgb([cielab_colors_array])[0]  # Assumes D65 illuminant and 2° observer by default
    # print(rgb_colors)

    # Plot directly on the provided axes
    ax.scatter(a_values, b_values, l_values, facecolors=rgb_colors, s=50, marker='o')
    ax.set_xlim((-128, 127))
    ax.set_ylim((-128, 127))
    ax.set_zlim((0, 100))

    ax.set_xlabel('a', fontsize=26, fontweight='bold', labelpad=10)
    ax.set_ylabel('b', fontsize=26, fontweight='bold', labelpad=10) 
    ax.set_zlabel('L', fontsize=26, fontweight='bold', labelpad=10)
    ax.set_title(word + ', I= ' + str(round(informativeness[word], 2)), fontsize=26, fontweight='bold') 
    #ax.set_title(word + ' ' + r'$I_w$, = ' + str(round(informativeness[word], 2)), fontsize=30, fontweight='bold') 

def scatter_plot(ax, word):
    cielabs = word_CIELABs[word]  # Use your dictionary of CIELAB values
    visualize_cielab_colors(ax, cielabs, informativeness, word)

data = pd.read_csv('../condition3_generated/experiment2/condition_a/dump_context/msg_rf_seed111/epoch30_model_data.csv')
word_CIELABs = {}
informativeness = {}
for index, row in data.iterrows():
    if row['name'] not in word_CIELABs:
        word_CIELABs[row['name']] = [ast.literal_eval(row['tar_cielab'])]
        informativeness[row['name']] = row['informativeness']
    else:
        word_CIELABs[row['name']].append(ast.literal_eval(row['tar_cielab']))

print(word_CIELABs)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter_plot(ax, 'blue')  # Replace 'sky' with any word you’re analyzing
plt.savefig("blue.png")
plt.show()