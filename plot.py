import matplotlib.pyplot as plt

def plot_stairsteps(x_core, y_core, c_core, label, line_width = 1):
    plt.plot([x_core[0], x_core[0] + (x_core[1]-x_core[0])/2], [y_core[0], y_core[0]], color=c_core, label=label)
    plt.plot([x_core[0] + (x_core[1]-x_core[0])/2, x_core[0] + (x_core[1]-x_core[0])/2], [y_core[0], y_core[1]], color=c_core)
    for i in range(1, (len(x_core) - 1)):
        left_x = x_core[i] - (x_core[i]-x_core[i-1])/2
        right_x = x_core[i] + (x_core[i+1]-x_core[i])/2
        plt.plot([left_x, right_x], [y_core[i], y_core[i]], color=c_core, linewidth = line_width)
        plt.plot([right_x, right_x], [y_core[i], y_core[i+1]], color=c_core, linewidth = line_width)

def pretty_plot(axe, legend=False, top=False, bottom = False, right=False, last=False):
    if legend==True:
        plt.legend(fontsize=15)
    if top==False:
        axe.spines['top'].set_visible(False)
    if bottom==False:
        axe.spines['bottom'].set_visible(False)
    if right==False:
        axe.spines['right'].set_visible(False)
    axe.tick_params('y', labelsize=15)
    if last==False:
        axe.tick_params('x', bottom=False, labelbottom=False)
    else:
        axe.tick_params('x', labelsize=15)
        
def pretty_plot_vertical(axe, title, y_min, y_max, x_label, y_label):
    plt.title(title, fontsize=20)
    plt.ylim(y_min, y_max)
    plt.xlabel(x_label, fontsize=20)
    axe.xaxis.tick_top()
    axe.xaxis.set_label_position('top') 
    plt.ylabel(y_label, fontsize=20)
    pretty_plot(axe, legend=False, top = True, last = True)
