import matplotlib.pyplot as plt


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
