from pandas import DataFrame
from seaborn import violinplot, swarmplot
from sklearn.metrics import cohen_kappa_score
from matplotlib.pylab import figure


def cohen_kappa(merge_df, plot_option, title):
    '''




    '''
    agg_Cohen = []

    for ind, person in merge_df.groupby(['Id_people', 'Exposures', 'Modality']):

        id_pessoa = person['Id_people'].head(1).values.item()
        exposures = person['Exposures'].head(1).values.item()
        modality = person['Modality'].head(1).values.item()

        c_k_s = cohen_kappa_score(
            y1=person['Real Bin'], y2=person['Predicted Bin'], weights='quadratic')

        agg_Cohen.append([id_pessoa, exposures, modality, c_k_s])

    agg_Cohen = DataFrame(agg_Cohen, columns=[
        'id_person', 'Exposures', 'Modality', 'Cohen_Kappa'])

    if (plot_option):
        fig = figure()
        ax = violinplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                        data=agg_Cohen, palette=['#de0000', '#001a8d'])

        ax = swarmplot(x="Exposures", y="Cohen_Kappa", hue="Modality",
                       data=agg_Cohen, size=8, ax=ax,
                       palette=['#ff9293', '#aaaaff'],
                       dodge=True,
                       alpha=.2)

        ax = ax.set(ylim=(-1, 1))

        fig.suptitle(title)

        return agg_Cohen, fig

    else:

        return agg_Cohen