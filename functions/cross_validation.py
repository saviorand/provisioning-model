from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from abess import LinearRegression
import numpy as np


def panel_cross_validation(models, model_names, x, y):
    """
    Conducts cross-validation using different models and returns the best model along with details on evaluated models.

    Parameters:
    - models: list of machine learning models to evaluate
    - model_names: list of names for the models
    - x: feature matrix
    - y: target vector

    Returns:
    - best_model: the best model
    - results: a dictionary containing the results of all evaluated models
    """
    candidate_splits = [5]
    # do this initially [2, 3, 5, 7, 10, 15]
    results = {}

    best_model = None
    best_score = float('inf')
    best_model_name = ''
    best_n_splits = 0
    best_lasso_param = None

    for model, model_name in zip(models, model_names):
        for n in candidate_splits:
            tscv = TimeSeriesSplit(n_splits=n)
            lasso_param = None
            coefs = None
            intercept = None

            if model_name == 'Lasso':
                parameters = {'alpha': [0.01, 0.1, 0.5, 0.59, 0.6, 0.7, 1, 10, 100]}
                grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=tscv)
                pipeline = Pipeline(
                    [('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler()), ('model', grid)])
                pipeline.fit(x, y)
                lasso_param = grid.best_params_
                best_lasso_model = grid.best_estimator_
                coefs = best_lasso_model.coef_
                intercept = best_lasso_model.intercept_

            elif model_name == 'PCA':
                pipeline = Pipeline(
                    [('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler()), ('pca', model),
                     ('regression', LinearRegression())])
            else:
                pipeline = Pipeline(
                    [('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler()), ('model', model)])

            pipeline.fit(x, y)

            if model_name != 'Lasso' and model_name != 'PCA':
                fitted_model = pipeline.named_steps['model']
                coefs = fitted_model.coef_
                intercept = fitted_model.intercept_

            scores = cross_val_score(pipeline, x, y, cv=tscv, scoring='neg_mean_squared_error')
            mean_score = -np.mean(scores)
            std_score = np.std(scores)

            if mean_score < best_score:
                best_score = mean_score
                best_model = pipeline
                best_model_name = model_name
                best_n_splits = n
                best_lasso_param = lasso_param

            print(
                f"Model: {model_name}, Mean MSE: {mean_score:.4f}, Std MSE: {std_score:.4f}, n_splits: {n}, Lasso alpha: {lasso_param}")
            results[model_name + '_' + str(n)] = {'Mean_MSE': mean_score, 'Std_MSE': std_score,
                                                  'Lasso_alpha': lasso_param, 'Coefficients': coefs,
                                                  'Intercept': intercept}

    print(
        f"Best Model: {best_model_name}, Mean MSE: {best_score}, n_splits: {best_n_splits}, Lasso alpha: {best_lasso_param}")
    return best_model, results


def filter_and_find_best_model(results):
    best_model = None
    best_score = float('inf')

    for model_key, model_info in results.items():
        model_name = model_key
        model_coef = model_info.get('Coefficients', None)
        mean_score = model_info.get('Mean_MSE', float('inf'))

        # print(f"Debug: model_coef = {model_coef}")

        if model_coef is not None:
            if not all(coef == 0 for coef in model_coef):
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model_name  # Changed this to model_name, as we are not storing the actual model object in this example

    return best_model


def plot_cross_validation_results(results):
    model_names = []
    mean_mses = []
    std_mses = []
    coefficients = []

    for model_name, model_result in results.items():
        model_names.append(model_name)
        mean_mses.append(model_result['Mean_MSE'])
        std_mses.append(model_result['Std_MSE'])
        coefficients.append(model_result['Coefficients'])

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Plotting Mean MSE with error bars for Std MSE
    axs[0].barh(model_names, mean_mses, xerr=std_mses, align='center', alpha=0.7, color='b', ecolor='black', capsize=10)
    axs[0].set_xlabel('Mean MSE')
    axs[0].set_title('Model Performance')

    # Plotting coefficients for the models
    for idx, coef in enumerate(coefficients):
        axs[1].plot(coef, label=f'{model_names[idx]} coefficients')

    axs[1].set_xlabel('Feature index')
    axs[1].set_ylabel('Coefficient value')
    axs[1].set_title('Model Coefficients')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
