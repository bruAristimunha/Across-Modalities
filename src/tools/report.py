from sklearn.metrics import classification_report

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