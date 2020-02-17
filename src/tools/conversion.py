
from xarray import DataArray
from pandas import DataFrame, merge
from numpy import concatenate
from scipy.stats import mode

def to_DataFrame(data, classe, CHANNEL_NAMES):
    '''
    TO-DO
    
    '''
    x_array = DataArray(data)
    x_array = x_array.rename({'dim_0': 'people','dim_1': 'channel','dim_2':'trial'})
    x_array = x_array.transpose('people', 'trial', 'channel')

    df = x_array.to_dataframe('channel').unstack()

    df_classe =  DataFrame(classe.stack()).reset_index()
    df_classe.columns = ['people','trial','group']
    df = df_classe.merge(df,on=['people','trial'])
    
    df.columns = ['people',  'trial', 'group']+CHANNEL_NAMES
    df = df.drop(['HEOG','VEOG'],1)
    
    return df

def split_exposure(merge_data):
    
    exposicao1 = merge_data[merge_data['Exposures']=='E1']

    exposicao1_aud = exposicao1[exposicao1['Modality']=='Auditory']
    exposicao1_vis = exposicao1[exposicao1['Modality']=='Visual']

    exposicao2 = merge_data[merge_data['Exposures']=='E2']

    exposicao2_aud = exposicao2[exposicao2['Modality']=='Auditory']
    exposicao2_vis = exposicao2[exposicao2['Modality']=='Visual']
    
    return exposicao1_aud, exposicao2_aud, exposicao1_vis, exposicao2_vis 


def merge_and_clean(df_aver, clean):
    
    clean =  clean.drop_duplicates()
    df_clean = merge(right=df_aver,left=clean, on=['people','trial'], how='outer', validate='one_to_many')
    df_clean = df_clean.fillna(False)
    df_clean = df_clean[~df_clean['bad_flag']]
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean.drop('bad_flag',1)



def merge_export(r_aud_1, r_aud_2, r_vis_1, r_vis_2, 
                 export_name="../data/processed/resu_fig5.csv",
                 export_mode=True):

    merge = DataFrame(concatenate([r_aud_1, r_aud_2, r_vis_1, r_vis_2]))

    #import pdb; pdb.set_trace()
    merge[0] = merge[0].astype(int)
    merge[1] = merge[1].astype(int)
    merge[2] = merge[2].astype(int)

    merge.columns = ['Id_people','Predicted Bin','Real Bin','Modality','Exposures']

    merge.to_csv(export_name,index=None)
    
    if (export_mode):
        
        merge_mode = merge.groupby(['Id_people', 
                                    'Modality', 
                                    'Exposures', 
                                    'Real Bin'])['Predicted Bin'].apply(lambda x: mode(x)[0]).reset_index()

        merge_mode['Predicted Bin'] = merge_mode['Predicted Bin'].astype(int)

        merge_mode.to_csv(export_name+'mode',index=None)
        
        return merge, merge_mode
    
    else:
        return merge

def to_DataFrame_autoenconder(data, classe, CHANNEL_NAMES):
    '''
    TO-DO
    
    '''
    x_array = DataArray(data)
    x_array = x_array.rename({'dim_0': 'people','dim_1': 'channel','dim_2':'time','dim_3':'trial'})
    x_array = x_array.transpose('people', 'trial', 'channel','time')
    x_array.to_dataframe('channel').unstack()

    df = x_array.to_dataframe('time').unstack()

    df_classe =  DataFrame(classe.stack()).reset_index()
    df_classe.columns = ['people','trial','group']
    df_v = df_classe.merge(df,on=['people','trial'], validate='one_to_many',how='outer')
    df_v['channel'] = df.reset_index()['channel']

    df_v['channel'].replace(dict(zip(list(range(64)),CHANNEL_NAMES)), inplace=True)

    time_legend = ['time '+str(i) for i in range(32)]

    df_v.columns = ['people',  'trial', 'group']+time_legend+['channel']

    df_v = df_v[['people',  'trial', 'group','channel']+time_legend]

    df_v = df_v[~((df_v['channel'] =='HEOG') | (df_v['channel'] =='VEOG'))]


    return df_v
    
