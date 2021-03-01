# REAL-X


This code implements REAL-X, a method for explaining predictions. REAL-X learns a selector model that explains any instance of data with a single forwards pass. We equipped our implementation of REAL-X with EVAL-X, a method for quantitively evaluating explainations.

---

## Why Explain with REAL-X: 

To learn about how REAL-X works, check out [our paper](https://arxiv.org). Below we summarize the benefits of REAL-X:

1. Efficently explains any instance of data with a single forward pass.
2. Produce high fidelity/accuracy explainations that can verified with EVAL-X
3. Avoid encoding predictions within explainations and out-of-distribution issues (see [our paper](https://arxiv.org))

---

## Running REAL-X

Explaining with REAL-X invloves three steps: 

1. Initialize the selector model and predictor model. (Any model archetecture can be specified)
2. Choose the REAL-X hyperparameter (lambda) and any other training hyperparameters (i.e. batch_size)
3. Train the predictor and the selector model.

Once REAL-X can been trained, its selector model can be used directly to generate explainations. REAL-X explainations can also be validated with EVAL-X (built-in). 

Please, check out our [example](example.ipynb) to see how we apply REAL-X to explain MNIST classifications.

### Training REAL-X

This implementation of REAL-X is designed to work with the Keras API.

 ```python
 # initialize REALX w/ the selector model, predictor_model, and REAL-X hyperparameter (lambda)
 realx = REALX(selector_model, predictor_model, lambda)
 ```
 
  ```python
 # train the predictor and selector model
realx.predictor.compile(loss=...,
                        optimizer=...,
                        metrics=...)
realx.predictor.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
realx.build_selector()
realx.selector.compile(loss=None,
                       optimizer=...,
                       metrics=...)
realx.selector.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
 ```
 
 ### Generating Explainations with REAL-X
 ```python
 # generate explainations
 explainations = realx.select(x_test, batch_size, discrete=True)
 ```
 
 ### Evaluating Explainations with EVAL-X
 ```python
 # evaluate explainations with EVAL-X
realx.evalx.compile(loss=...,
                        optimizer=...,
                        metrics=...)
realx.evalx.fit(x_train,
                    y_train,
                    epochs=...,
                    batch_size=...)
y_eval = realx.evaluate(x_test, batch_size)

eAUROC = roc_auc_score(y_test, y_eval, 'micro')
eACC = accuracy_score(y_test.argmax(1), y_eval.argmax(1))
 ```
 
 ---
 

# Citing this code
If you use this code, please cite the following paper ([available here](https://arxiv.org/)):
```
Jethani,  Neil, Sudarshan, Mukund, Aphinyanaphongs, Yin, and Rajesh Ranganath. "Have We Learned to Explain?: How Interpretability Methods Can Learn to Encode Predictions in their Interpretations." Proceedings of the Twenty Fourth International Conference on Artificial Intelligence and Statistics (2021).
```

---

## Dependencies
The code for REAL-X runs with Python and Tensorflow version 2.2.0. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `scikit-learn`
- `scipy`
- `matplotlib`

```
sudo pip install -r requirements.txt
```