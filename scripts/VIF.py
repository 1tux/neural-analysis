import numpy as np
import pygam
from sklearn.preprocessing import StandardScaler
import pandas as pd
'''
    VIF: https://en.wikipedia.org/wiki/Variance_inflation_factor
    Generalization (1) to GAM: by using GAM (without a link function / constraints) and trying to predict each of the features using the others.
    The ration betwen the original variance of the left-out feature divided by the residuals (errors) variance of it after the prediction reflects the gVIF.
    Generalization (2) in order to support two-dimensional features such as position, gVIF was extended even more. The determinent of the covariance matrix of both X,Y positional features reflect the variance of the 2d positional feature.
    Also, the natural mannger of scaling is taking the k^th root of that determinent where k is the number of dimensions. for each is simply the square-root as POSITION is two-dimensioal.
'''

def VIF_inline(X, y, factor_by=1):
    model = pygam.GAM()
    model.fit(X, y)
    y_hat = model.predict(X)
    return (y_hat - y) * factor_by

def calc_gam_general_VIF(dataset, feature_name):
    scaler = StandardScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)#.head(50000)
    if feature_name.endswith("_POS"):
        c1 = feature_name.replace("_POS","_X")
        c2 = feature_name.replace("_POS","_Y")
        y1 = dataset[c1]
        y2 = dataset[c2]
        
        var1 = np.cov([y1, y2])
        factor = 1 / np.sqrt(np.linalg.det(var1))
        dataset_ = dataset[dataset.columns.difference([c1, c2])]
        print(dataset_.columns, c1, c2)
        res1 = VIF_inline(dataset_, dataset[c1], factor)
        res2 = VIF_inline(dataset_, dataset[c2], factor) 
        var2 = np.cov([res1, res2])
        # print(np.linalg.det(var1 * factor))
        print("two-d variances", var1, var2)
        return np.sqrt(np.linalg.det(var1) / np.linalg.det(var2))
    else: 
        var = np.var(dataset[feature_name])
        dataset_ = dataset[dataset.columns.difference([feature_name])]
        res = VIF_inline(dataset_, dataset[feature_name])
        print(var, np.var(res))
    return var / np.var(res)


if __name__ == "__main__":
    behavioral_path, neuron_path = main.get_paths(72)
    df = pd.read_csv(behavioral_path).drop(columns=["Unnamed: 0"])
    df2 = df.dropna()
    scaler = StandardScaler()
    df3 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns, index=df2.index)#.head(50000)

    # """
    df3 = df3[['BAT_0_F_X', 'BAT_0_F_Y',\
               'BAT_1_F_X', 'BAT_1_F_Y',\
               'BAT_2_F_X', 'BAT_2_F_Y',\
               'BAT_3_F_X', 'BAT_3_F_Y',\
               'BAT_4_F_X', 'BAT_4_F_Y',\
              ]]
    # """
    features = [x for x in df3.columns if not x.endswith("Y")]
    # features = [x.replace("_X", "_POS") for x in features]
    print(features)

    for f in df3.columns:
        # print(f, calc_gam_VIF(df3, f))
        print(f, calc_gam_general_VIF(df3, f))
        if f.endswith("_X"):
            f_ = f.replace("_X", "_POS")
            print(f_, calc_gam_general_VIF(df3, f_))
    