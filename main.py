import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('wdbc.data')
df.drop('ID', axis=1, inplace=True)

df.head()

le = LabelEncoder()
le.fit(df['Diagonsis'])
df['Diagonsis'] = le.transform(df['Diagonsis'])
df.head()

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

l = [
    "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Area_se",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", 'Worst_Area'
]

mms = MinMaxScaler()
mms.fit(X[l])
X[l] = mms.transform(X[l])

X.head()

l = ["Malignant", "Benign"]
m = y[y == 1].count()
b = y[y == 0].count()

plt.figure()
plt.bar(l[0], m)
plt.bar(l[1], b)
plt.xlabel("Category of Cancer")
plt.ylabel("Number of Cases")
plt.title("Count of Malignant and Benign Cancers")
plt.show()

corr = X.corr()
plt.figure(figsize=(20, 20))
ax = sns.heatmap(corr, annot=True)
plt.show()

columns = np.full((corr.shape[0], ), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = X.columns[columns]

X = X[selected_columns]
selected_columns = list(selected_columns.values)
X.head()


def p_threshold(X, y, sl, columns):
    numOfVars = len(columns)
    for i in range(0, numOfVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numOfVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    t = columns[j]
                    X.drop(t, axis=1, inplace=True)
                    columns.pop(j)
    print(regressor_OLS.summary())
    return X, columns


SL = 0.05
X, selected_columns = p_threshold(X, y, SL, selected_columns)

plt.figure(figsize=(20, 20))
j = 0
for i in X.columns:
    plt.subplot(6, 3, j + 1)
    j += 1
    sns.distplot(X[i][y == 1], label='Malignant')
    sns.distplot(X[i][y == 0], label='Benign')
    plt.legend(loc='best')
plt.suptitle('Breast Cancer Analysis')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
