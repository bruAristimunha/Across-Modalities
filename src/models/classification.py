from sklearn.model_selection import LeaveOneOut
from tqdm.autonotebook import tqdm
from numpy import average, subtract
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from pandas import DataFrame, concat

def classification_within_modality(dataFrame, categoria, exposure):
    '''
    
    
    
    
    
    
    '''
    dataFrame_result = []
    loo = LeaveOneOut()
    
    pbar = tqdm( total=loo.get_n_splits(dataFrame))
    
    for ind, pearson in dataFrame.groupby('people'):
    
        X = pearson.drop(['trial', 'group','people'],1)
        y = pearson['group']

        loo = LeaveOneOut()
        

        for train_index, test_index in loo.split(X):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #Normalize
            train_mean = average(X_train,axis=0)
            
            X_train_without_mean = subtract(X_train,train_mean)           
            X_test_without_mean = subtract(X_test,train_mean)           

            
            clf = GaussianNB()
            
            clf.class_prior_ = [(1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]
            
            pca_ = PCA(random_state=42, svd_solver='full', n_components=0.99)
    
            pca = pca_.fit(X_train_without_mean)

            X_train_pca = pca.transform(X_train_without_mean)
            
            X_test_pca = pca.transform(X_test_without_mean)
            
            
            clf = clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)

            dataFrame_result.append([ind,y_pred,y_test.values,categoria,exposure])
            
            pbar.update(1)
            
    return dataFrame_result


def classification_across_modality(dataFrame1, dataFrame2, inp, exp):





    dataFrame_result = []

    for (ind_1, pearson_1), (ind_2, pearson_2)  in list(zip(dataFrame1.groupby('people'),dataFrame2.groupby('people'))):

        X_train = pearson_1.drop(['trial', 'group','people'],1)
        y_train = pearson_1['group']

        X_test = pearson_2.drop(['trial', 'group','people'],1)
        y_test = pearson_2['group']

        #Normalize
        train_mean = average(X_train,axis=0)

        X_train_without_mean = subtract(X_train,train_mean)           
        X_test_without_mean = subtract(X_test,train_mean)           


        clf = GaussianNB()

        clf.class_prior_ = [(1/6),(1/6),(1/6),(1/6),(1/6),(1/6)]

        pca_ = PCA(random_state=42, svd_solver='full', n_components=0.99)

        pca = pca_.fit(X_train_without_mean)

        X_train_pca = pca.transform(X_train_without_mean)

        X_test_pca = pca.transform(X_test_without_mean)


        clf = clf.fit(X_train_pca, y_train)

        y_pred = clf.predict(X_test_pca)
        for y_i_pred, y_i_test in list(zip(y_pred,y_test.values)):
            dataFrame_result.append([ind_1, y_i_pred,y_i_test, inp, exp])

    return dataFrame_result
