import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os


n=500

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    return mpl.colors.to_hex(((1-mix)*np.array(c1) + mix*np.array(c2)) / 255.001)


def get_spectrum_plot(colors, colors_name):
    
    bin_length = np.ceil((n / (len(colors)-1)))
    fig, ax = plt.subplots(figsize=(8, 1))

    for x in range(n):
        if x % bin_length == 0:
            c1 = colors[int(x / bin_length)]
            c2 = colors[int(x / bin_length)+1]
        
        ax.axvline(x, color=colorFader(c1,c2,(x-np.floor(x/bin_length)*bin_length)/bin_length), linewidth=4) 

    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the x and y axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    print("DONE")

    os.makedirs(os.path.join(os.getcwd(), "paper_colors"), exist_ok=True)
    fig.savefig(os.path.join(os.getcwd(), "paper_colors", colors_name), bbox_inches='tight', pad_inches=0)
    plt.show()


# intended traversable
colors_intended = [[0, 255, 0], [0, 204, 0], [0, 153, 0], [0, 102, 0]]
colors_name = "spectrum_traversable_intended.pdf"
get_spectrum_plot(colors_intended, colors_name)

# unintended traversable
colors_unintended = [[204, 255, 0], [153, 204, 0], [204, 102, 0]]
colors_name = "spectrum_traversable_unintended.pdf"
get_spectrum_plot(colors_unintended, colors_name)

# terrain
colors_terrain = [[255, 255, 0], [255, 255, 0]]
colors_name = "spectrum_terrain.pdf"
get_spectrum_plot(colors_terrain, colors_name)

# road
colors_road = [[255, 128, 0], [255, 128, 0]]
colors_name = "spectrum_road.pdf"
get_spectrum_plot(colors_road, colors_name)

# human and vehicle
colors_human_vehicle = [[255, 0, 0], [204, 0, 0], [153, 0, 0], [102, 0, 0], [51, 0, 0]]
colors_name = "spectrum_human_vehicles.pdf"
get_spectrum_plot(colors_human_vehicle, colors_name)

# construction and nature
colors_construction_nature = [[127, 0, 255], [102, 0, 204], [76, 0, 153], [51, 0, 102], [153, 0, 153], [204, 0, 204]]
colors_name = "spectrum_construction_nature.pdf"
get_spectrum_plot(colors_construction_nature, colors_name)

# objects
colors_objects = [[0, 0, 255], [0, 0, 204], [0, 0, 153], [0, 0, 102], [0, 0, 51]]
colors_name = "spectrum_objects.pdf"
get_spectrum_plot(colors_objects, colors_name)

# sky and unknown
colors_sky_unknown = [[102, 0, 51], [0, 0, 0]]
colors_name = "spectrum_sky_unknown.pdf"
# get_spectrum_plot(colors_sky_unknown, colors_name)
