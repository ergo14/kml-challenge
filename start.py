import numpy as np
import pandas as pd

from utils import prepare_numeric_data
from spectrum_kernel import prepare_kspectrum_data
from gaussian_svm import GaussianSVM



def dataset_prediction_0():
    X_train, y_train, X_val, y_val, X_te = prepare_numeric_data(
        dataset=0,
        path='data',
        scale_method='minmax',
        with_intercept=False,
        num_train=1500,
    )
    
    model = GaussianSVM(C=1, l=0.55).fit(X_train, y_train)
    return model.predict_class(X_te).reshape(-1)


def dataset_prediction_1():
    X_train, y_train, X_val, y_val, X_te = prepare_kspectrum_data(
        dataset=1,
        path='data',
        k=4,
        scale_method='minmax',
        with_intercept=False,
        num_train=1500,
    )

    model = GaussianSVM(C=5, l=1.61).fit(X_train, y_train)
    return model.predict_class(X_te).reshape(-1)


def dataset_prediction_2():
    X_train, y_train, X_val, y_val, X_te = prepare_kspectrum_data(
        dataset=2,
        path='data',
        k=5,
        scale_method='minmax',
        with_intercept=False,
        num_train=1500,
    )

    model = GaussianSVM(C=3, l='auto').fit(X_train, y_train)
    return model.predict_class(X_te).reshape(-1)


def predict_for_dataset(dataset, model):
    X_train, y_train, X_val, y_val, X_te = prepare_numeric_data(
        dataset=dataset,
        path='data',
        scale_method='minmax',
        with_intercept=False,
        num_train=1500,
    )
    
    model = model.fit(X_train, y_train)
    return model.predict_class(X_te)
    

prediction_arr = np.concatenate([dataset_prediction_0(), dataset_prediction_1(), dataset_prediction_2()])
prediction_df = pd.DataFrame(prediction_arr).rename(columns={0: 'Bound'})
prediction_df.to_csv('Yte.csv', index=True, index_label='Id')




