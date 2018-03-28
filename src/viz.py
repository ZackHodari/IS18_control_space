import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helper import load_from_file
import numpy as np
import os


def plot_learning_curves(experiment_labels, experiment_stats,
                         title, file_name,
                         window=(None,), save=False):
    """
    Takes list of experiment labels and results, plots learning curves for each metric

    :param experiment_labels: list of names to label each experiments' results using
    :param experiment_stats: list of results dictionaries, containing epochs/stats lists for each metric
    :param title: heading at the top of the figure
    :param file_name: prefix used when saving barchart figure
    :param window: tuple that represents a valid slice, allowing for selective plotting along x-axis
    :param save: boolean (default: False)
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    cmap = plt.get_cmap('Set1')

    # For each model, plot the change in the validation and training set error and accuracy over training
    for i, (label, log_data) in enumerate(zip(experiment_labels, experiment_stats)):
        colour = cmap(float(i) / len(experiment_stats))

        axs[0].plot(log_data['error_train']['epochs'][slice(*window)],
                    log_data['error_train']['stats'][slice(*window)], label=label, c=colour)
        axs[0].plot(log_data['error_valid']['epochs'][slice(*window)],
                    log_data['error_valid']['stats'][slice(*window)], '--', c=colour)
            
        axs[1].plot(log_data['accuracy_train']['epochs'][slice(*window)],
                    log_data['accuracy_train']['stats'][slice(*window)], c=colour)
        axs[1].plot(log_data['accuracy_valid']['epochs'][slice(*window)],
                    log_data['accuracy_valid']['stats'][slice(*window)], '--', c=colour)
    
    handles, labels = axs[0].get_legend_handles_labels()
    extra = Rectangle((0, 0), 1, 1, fc='w', fill=False, edgecolor='none', linewidth=0)
    lgd_pos = (0.6 * 2 - 0.1, -0.1)  # 0.5 per subplot, 0.1 per space between subplot (i.e. 0.6 minus one space)
    lgd = axs[0].legend([extra]+handles, ['solid = train, dotted = valid']+labels, 
                        loc='upper center', bbox_to_anchor=lgd_pos, fancybox=True, shadow=True, ncol=2)
    
    plt.setp(lgd.get_lines(), linewidth=4.)
    
    tit = fig.suptitle('Error & accuracy learning curves - {}'.format(title), fontsize=20)
    axs[0].set_xlabel('Epoch number')
    axs[0].set_ylabel('log Error')
    axs[0].set_title('Error')
        
    axs[1].set_xlabel('Epoch number')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')

    if save:
        print('saving to results/graphs/{}-epochgraph.pdf'.format(file_name))
        plt.savefig(
            os.path.join(os.environ['project_DIR'], 'results', 'graphs', '{}-epochgraph.pdf'.format(file_name)),
            bbox_extra_artists=(lgd, tit,), 
            bbox_inches='tight'
        )

    return fig, axs


def plot_performance(experiment_labels, experiment_stats,
                     title, file_name,
                     loc='upper left', save=False):
    """
    Takes list of experiment labels and results, plots the minimum/maximum values for each metric on a bar chart

    :param experiment_labels: list of names to label each experiments' results using
    :param experiment_stats: list of results dictionaries, containing epochs/stats lists for each metric
    :param title: heading at the top of the figure
    :param file_name: prefix used when saving barchart figure
    :param loc: position of the legend in the figure, allows repositioning to avoid occluding bars
    :param save: boolean (default: False)
    """

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    indices = np.arange(len(experiment_stats))

    heights_error_train, heights_error_valid = [], []
    heights_acc_train, heights_acc_valid = [], []

    # For each model, plot the optimal values of validation and training set error and accuracy
    for i, (label, log_data) in enumerate(experiment_stats[::-1]):
        heights_error_train.append(np.min(log_data['error_train']['stats']))
        heights_error_valid.append(np.min(log_data['error_valid']['stats']))

        heights_acc_train.append(np.max(log_data['accuracy_train']['stats']))
        heights_acc_valid.append(np.max(log_data['accuracy_valid']['stats']))

    # draw horizontal bars
    width = 0.35
    rects11 = axs[0].barh(indices + width, heights_error_train, width, color='c', label='training')
    rects12 = axs[0].barh(indices, heights_error_valid, width, color='DarkOrange', label='validation')

    rects21 = axs[1].barh(indices + width, heights_acc_train, width, color='c')
    rects22 = axs[1].barh(indices, heights_acc_valid, width, color='DarkOrange')

    # add labels to both axes
    axs[0].set_ylabel('Experiment names')

    axs[0].set_xlabel('Error')
    axs[1].set_xlabel('Accuracy')

    axs[0].set_yticks(indices + width)
    axs[1].set_yticks(indices + width)

    axs[0].set_yticklabels(experiment_labels)
    axs[1].set_yticklabels([])

    # annotate exact numbers on each bar
    def autolabel(rects, ax):
        # get y-axis height so we can calculate label position
        (x_min, x_max) = ax.get_xlim()
        x_width = x_max - x_min

        # for all bars in the horizontal bar chart
        for rect in rects:
            width = rect.get_width()

            # fraction of axis height taken up by this rectangle
            p_width = (width / x_width)

            # check if we can fit the label outside the bar, otherwise put label inside the bar
            if p_width < 0.18:
                label_position = width + (x_width * 0.38 / min(6, len(experiment_labels)))
            else:
                label_position = width - (x_width * 0.38 / min(6, len(experiment_labels)))

            ax.text(label_position, rect.get_y() + rect.get_height()/7.,
                    '{0:.4f}'.format(width), fontsize=max(16, 100/len(experiment_labels)),
                    ha='center', va='bottom')

    # add labels to training/validation bars for both axes
    autolabel(rects11, axs[0])
    autolabel(rects12, axs[0])
    autolabel(rects21, axs[1])
    autolabel(rects22, axs[1])

    tit = fig.suptitle('Optimal values of error & accuracy - {}'.format(title), fontsize=20)
    axs[0].legend(loc=loc)
    plt.tight_layout()
    plt.subplots_adjust(top=0.925)

    if save:
        print('saving to results/graphs/{}-barchart.pdf'.format(file_name))
        plt.savefig(
            os.path.join(os.environ['project_DIR'], 'results', 'graphs', '{}-barchart.pdf'.format(file_name)),
            bbox_extra_artists=(tit,),
            bbox_inches='tight'
        )

    return fig, axs


def viz_experiments(experiments_info, title, file_name,
                    bar=True, curve=True,
                    window=(None,), loc='upper left', save=False):
    """
    Takes list of experiment tuples, plots curves/bar charts for each output type (in case multiple were used)

    :param experiments_info: list of label-file_name tuples
    :param title: heading at the top of the figure
    :param file_name: prefix used when saving barchart figure
    :param bar: boolean (default: True) plot the bar charts
    :param curve: boolean (default: True) plot the learning curves
    :param window: tuple that represents a valid slice, allowing for selective plotting along x-axis for learning curves
    :param loc: position of the legend in the figure, allows repositioning to avoid occlusion in bar charts
    :param save: boolean (default: False)
    """

    experiment_labels = map(lambda (label, name): label, experiments_info)
    experiment_stats = map(lambda (label, name): load_from_file(
        os.path.join(os.environ['project_DIR'], 'results', name, 'results.log')), experiments_info)

    # collect all output types used by these experiments
    output_names = set()
    for results in experiment_stats:
        output_names = output_names.union(results.keys())

    # for all output types create different plots
    for output_name in list(output_names):
        output_type_stats = [results[output_name] for results in experiment_stats if output_name in results]

        output_type_title = '{} - {}'.format(title, output_name)
        output_type_file_name = '{}-{}'.format(file_name, output_name)

        if curve:
            plot_learning_curves(experiment_labels, output_type_stats, output_type_title, output_type_file_name, window, save)

        if bar:
            plot_performance(experiment_labels, output_type_stats, output_type_title, output_type_file_name, loc, save)



