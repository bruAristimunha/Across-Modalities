import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from matplotlib.gridspec import GridSpec
from numpy import percentile, median
from seaborn import violinplot, swarmplot

def print_classification_by_mod(merge_data):

    exposicao1 = merge_data[merge_data['Exposures']=='E1']

    exposicao1_aud = exposicao1[exposicao1['Modality']=='Auditory']
    exposicao1_vis = exposicao1[exposicao1['Modality']=='Visual']

    exposicao2 = merge_data[merge_data['Exposures']=='E2']

    exposicao2_aud = exposicao2[exposicao2['Modality']=='Auditory']
    exposicao2_vis = exposicao2[exposicao2['Modality']=='Visual']

    print("##################################################")
    print("Classificação nos Dados Visual, exposição 1")

    y_test = exposicao1_vis['Predicted Bin'].values
    y_pred = exposicao1_vis['Real Bin'].values


    print(classification_report(y_test, y_pred))
    
    print("##################################################")
    print("Classificação nos Dados Visual, exposição 2")

    y_test = exposicao2_vis['Predicted Bin'].values
    y_pred = exposicao2_vis['Real Bin'].values


    print(classification_report(y_test, y_pred))

    print("##################################################")
    print("Classificação nos Dados Auditivo, exposição 1")
    y_test = exposicao1_aud['Predicted Bin'].values
    y_pred = exposicao1_aud['Real Bin'].values

    print(classification_report(y_test, y_pred))

    print("##################################################")
    print("Classificação nos Dados Auditivo, exposição 2")
    y_test = exposicao2_aud['Predicted Bin'].values
    y_pred = exposicao2_aud['Real Bin'].values
    print(classification_report(y_test, y_pred))
    


def myLineError(ax, x, y, asymmetric_error, color_option):
    ax.errorbar(x, y, yerr=asymmetric_error, fmt='.',
                ecolor=color_option, color=color_option)
    ax.plot(x, y, color=color_option)
    ax.set(ylim=(0, 6.5))


def get_error_plot(df):
    x = df['Real Bin']
    y = df['median']
    quartil_25 = df['25th']
    quartil_75 = df['75th']
    asymmetric_error = [y-quartil_25, quartil_75-y]

    return x, y, asymmetric_error

def make_figure(merge_across_mode, merge_within_mode, ck_across, ck_within):

    plot_across = merge_across_mode.groupby(['Modality',
                                             'Exposures',
                                             'Real Bin'])['Predicted Bin'].agg({'median': median,
                                                                                '25th': lambda a: percentile(a, q=25),
                                                                                '75th': lambda a: percentile(a, q=75)})
    plot_across = plot_across.reset_index()

    plot_within = merge_within_mode.groupby(['Modality',
                                             'Exposures',
                                             'Real Bin'])['Predicted Bin'].agg({'median': median,
                                                                                '25th': lambda a: percentile(a, q=25),
                                                                                '75th': lambda a: percentile(a, q=75)})

    plot_within = plot_within.reset_index()

    fig = plt.figure(constrained_layout=False, figsize=(15, 15))

    gs = GridSpec(4, 4, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    exposi_aud_1, exposi_aud_2, exposi_vis_1, exposi_vis_2 = [
        x for _, x in plot_within.groupby(['Modality', 'Exposures'])]

    #import pdb; pdb.set_trace()
    
    x, y, asymmetric_error = get_error_plot(exposi_aud_1)
    myLineError(ax0, x, y, asymmetric_error, 'red')
    
    #########################################################
    x, y, asymmetric_error = get_error_plot(exposi_vis_1)
    myLineError(ax1, x, y, asymmetric_error, 'blue')

    ##########################################################
    x, y, asymmetric_error = get_error_plot(exposi_aud_2)
    myLineError(ax2, x, y, asymmetric_error, 'red')

    ##########################################################
    x, y, asymmetric_error = get_error_plot(exposi_vis_2)
    myLineError(ax3, x, y, asymmetric_error, 'blue')
    ##########################################################

    
    
    

    
    
    
    
    ax4 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[0, 3])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

    exposi_aud_1, exposi_aud_2, exposi_vis_1, exposi_vis_2 = [
        x for _, x in plot_across.groupby(['Modality', 'Exposures'])]

    x, y, asymmetric_error = get_error_plot(exposi_aud_1)
    myLineError(ax4, x, y, asymmetric_error, 'red')
    
    
    ##########################################################
    x, y, asymmetric_error = get_error_plot(exposi_vis_1)
    myLineError(ax5, x, y, asymmetric_error, 'blue')

    ##########################################################
    x, y, asymmetric_error = get_error_plot(exposi_aud_2)
    myLineError(ax6, x, y, asymmetric_error, 'red')

    ##########################################################
    x, y, asymmetric_error = get_error_plot(exposi_vis_2)
    myLineError(ax7, x, y, asymmetric_error, 'blue')
    ##########################################################

    ax_ck_with = fig.add_subplot(gs[3, 0:2])
    ax_ck_acro = fig.add_subplot(gs[3, 2:4])

    ax_ck_acro = violinplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                            data=ck_across, palette=['#de0000', '#001a8d'], ax=ax_ck_acro)
    ax_ck_acro = swarmplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                           data=ck_across, size=8, ax=ax_ck_acro,
                           palette=['#ff9293', '#aaaaff'],
                           dodge=True,
                           alpha=.2)

    ax_ck_acro.set_title('Across modalities')
    ax_ck_acro.get_legend().remove()

    ax_ck_with = violinplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                            data=ck_within, palette=['#de0000', '#001a8d'], ax=ax_ck_with)
    ax_ck_with = swarmplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                           data=ck_within, size=8, ax=ax_ck_with,
                           palette=['#ff9293', '#aaaaff'],
                           dodge=True,
                           alpha=.2)

    ax_ck_with.set_title('Within modalities')
    ax_ck_with.get_legend().remove()
    ax_ck_with = ax_ck_with.set(ylim=(-1, 1))

    ax0.tick_params(labelbottom=False)
    ax0.set_ylabel('E1')
    
    
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax2.set_ylabel('E2')
    ax3.tick_params(labelleft=False)

    ax4.tick_params(labelbottom=False)
    ax5.tick_params(labelbottom=False, labelleft=False)
    ax7.tick_params(labelleft=False)

    plt.tight_layout()
    plt.text(-2.0, 4.70, 'Within modalities', fontsize=16)

    plt.text(0.25, 4.70, 'Across modalities', fontsize=16)
    # format_axes(fig)
