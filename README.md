<a name="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents (click me)</summary>
  <ol>
    <li>
      <a href="#About-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#Data-Analysis">Data Analysis</a>
      <ul>
        <li><a href="#Dataset-Description">Dataset Description</a></li>
        <li><a href="#Example-plots-in-EDA">Example plots in EDA</a></li>
        <li><a href="#Data-Splitting">Data Splitting</a></li>
        <li><a href="#Examples-of-model-fitting-with-all-variables">Examples of model fitting with all variables</a></li>
        <li><a href="#Accuracy-of-Models">Accuracy of Models</a></li>
        <li><a href="#Feature-Selection">Feature Selection</a></li>
        <li><a href="#Results-after-variable-selection">Results after variable selection</a></li>
      </ul>
    </li>   
    <li><a href="#Imbalanced-Classification">Imbalanced Classification</a></li>    
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Marketing selling campaigns is an important strategy to enhance business. There are different ways that can help to determine whether a marketing campaign will be successful or not. 
A term deposit is a major source of income for a bank. 


The goal of this project is to find the best machine learning model to predict if a customer will subscribe (yes/no) to a term deposit based on the existing record from a Portuguese banking institution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [pandas][pandas-url]
* [numpy][numpy-url]
* [sklearn][sklearn-url]
* [matplotlib][matplotlib-url]
* [seaborn][seaborn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Data Analysis

### Dataset Description

4521 observations and 17 variables in the dataset, including Personal Client Information, Bank Client Information, Last Contact of the current campaign, Outcome: Has the client subscribed to a term deposit?

### Example plots in EDA

* Correlation heatmap
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/cor1016.png" width="400" height="300">
</div>

* Boxplot of clients’ age by jobs who subscribed to the term deposit
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/box1016.png" width="700" height="300">
</div>

* Scatter plot and curves
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/scatter1016.png" width="500" height="400">
</div>

### Data splitting

* Split Data
  ```sh
  X, y = df1.iloc[:, 0:16], df1.iloc[:, 16]
  X_nontest, X_test, y_nontest, y_test  = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2022)
  X_train, X_val, y_train, y_val  = train_test_split(X_nontest, y_nontest, test_size=0.2, 
                                                   shuffle=True, random_state=2022)
  ```

### Examples of Model fitting with all variables

* KNN
```sh
model_KNN = KNeighborsClassifier(n_neighbors=3)
model_KNN.fit(X_train, y_train)
```

* Example of Tune the model
```sh
from sklearn.model_selection import GridSearchCV
params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator = model_KNN, param_grid = params, cv = 10, verbose=2, n_jobs = 4)
grid.fit(X_val, y_val)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
```

* Support Vector Machine
```sh
model_svm_rbf = SVC(kernel='rbf', random_state=2022) # radial basis function
model_svm_rbf.fit(X_train, y_train)
```

* Random Forest
```sh
model_rf = RandomForestClassifier(max_depth=10, random_state=2022) # The maximum depth of the tree = 10.
model_rf.fit(X_train, y_train)
```

* Gradient Boosting
```sh
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=2022)
model_gb.fit(X_train, y_train)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Accuracy of models
* Training: KNN is the best model, Perceptron is the worst.
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/train1016.png" width="400" height="350">
</div>

* Testing: Gradient Boosting is the best model, Perceptron is again the worst.
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/test1016.png" width="400" height="350">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Feature Selection
* Mutual Information Gain
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/rf1016.png" width="500" height="500">
</div>


* Features Importance with Random Forest
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/drf1016.png" width="500" height="500">
</div>

Both algorithms suggest to us that the “duration” feature is very important for modeling a customer’s decision of subscribing to a term account. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Results after variable selection
* Confusion Matrix

Logistic regression with random forest feature selection has the best prediction.
<div align="center">
<img src="https://github.com/JILI1028/bank-marketing/blob/main/images/matirx1016.png" width="800" height="500">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Imbalanced classification
For the whole dataset, there were 521 successes and 4000 failures, about 11.52% success rate.

Is there a big difference for the best model if we consider it as an imbalanced classification problem?

Answer: Applied weighted regression estimator for logistic regression and the overall performances are similar.


<!-- CONTACT -->
## Contact

Ji Li - jil1@umbc.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[pandas.js]: https://pandas.pydata.org/static/img/pandas_white.svg
[pandas-url]: https://pandas.pydata.org/
[sklearn.js]: https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png
[sklearn-url]: https://scikit-learn.org/stable/
[matplotlib.js]: https://matplotlib.org/_static/logo_light.svg
[matplotlib-url]: https://pandas.pydata.org/](https://matplotlib.org/
[seaborn-url]: https://seaborn.pydata.org/
[seaborn.js]: https://seaborn.pydata.org/_static/logo-wide-lightbg.svg
[numpy-url]: https://numpy.org/
[heatmap.js]: https://github.com/JILI1028/bank-marketing/blob/main/images/cor1016.png
