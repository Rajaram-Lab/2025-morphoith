# %%

import seaborn as sns
import matplotlib.pyplot as plt

def rgb2hex(rgbList):
    r, g, b = [min(255, max(0, int(float(val)))) for val in rgbList]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def apply_plot_settings():
    plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
    sns.set(font_scale=1.5)
    sns.set_style("white")

def add_significance(ax, x1, x2, y, p_value, text_offset=0.05, line_height=0.025, tail_length=.8):
    ax.plot([x1, x1, x2, x2], [y- tail_length, y+line_height, y+line_height, y - tail_length], lw=1.8, c='black')

    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'ns' 
    ax.text((x1 + x2) * 0.5, y + text_offset, text, ha='center', va='bottom', color='black')
