

```python
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
```


```python
train_data=pd.read_csv("data/train.csv").as_matrix()
test_data=pd.read_csv("data/test.csv").as_matrix()
clf=ExtraTreesClassifier()
total=train_data.shape[0]
total_test=test_data.shape[0]
```

    /home/oem/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    /home/oem/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      



```python
#training data
xtrain=train_data[0:total, 1:]
train_label=train_data[0:total, 0]
```


```python
clf.fit(xtrain, train_label)
```




    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)




```python
xtest=test_data[0:total_test, 0:]
```


```python
p=clf.predict(xtest)
output=[]
```


```python
for i in range(0, total_test):
    curr_list=[]
    curr_list.append(i + 1)
    curr_list.append(p[i])
    output.append(curr_list)
```


```python
df=pd.DataFrame(output, columns=['ImageId', 'Label'])
```


```python
df.to_csv("output.csv", index=False)
```
