import numpy as np
import pandas as pd

from utils import prepare_numeric_data
from gaussian_svm import GaussianSVM



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
    

prediction_arr = np.concatenate(list(map(lambda x: x.reshape(-1), [
    predict_for_dataset(0,  GaussianSVM(C=1, l=0.55)),
    predict_for_dataset(1, GaussianSVM(C=1, l=1.15)),
    predict_for_dataset(2, GaussianSVM(C=1.1, l=1.17)),
])))

prediction_df = pd.DataFrame(prediction_arr).rename(columns={0: 'Bound'})

prediction_df.to_csv('Yte.csv', index=True, index_label='Id')




