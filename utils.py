import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from IPython.display import display, Markdown

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from yellowbrick.regressor import ResidualsPlot


def display_markdown(*args, **kwargs):
    return display(Markdown(*args, **kwargs))


def make_evaluation_errors_table(y_train, y_train_pred, y_valid, y_valid_pred):
    """
    Crear dataframe con valores de error de una evaluación.
    """
    metric_functions = [
        ('R squared', r2_score),
        ('Mean absolute error', mean_absolute_error),
        ('Mean squared error', mean_squared_error)
    ]

    data = {
        metric_function_name: [
            metric_function(y_train, y_train_pred),
            metric_function(y_valid, y_valid_pred)
        ]
        for metric_function_name, metric_function in metric_functions
    }
    return pd.DataFrame(data, index=['Training', 'Validation'])


def display_evaluation_errors(*args, **kargs):
    """
    Mostrar valores de error de una evaluación.
    """
    display(make_evaluation_errors_table(*args, **kargs))


def display_evaluation_table(y_real, y_pred, n_values=10, with_errors=True):
    """
    Mostrar tabla con valores de una evaluación, comparando valores predichos
    con valores esperados.
    """
    data = {'Prediction': y_pred, 'Real': y_real}
    if with_errors:
        data['Error'] = (y_pred - y_real).abs()
    display(pd.DataFrame(data).sample(n_values))


def show_residuals_plot(model, X_train, y_train, X_valid, y_valid):
    residuals_plot = ResidualsPlot(model)
    residuals_plot.fit(X_train, y_train)
    residuals_plot.score(X_valid, y_valid)
    residuals_plot.show()


def show_qq_plot(y_valid, y_valid_pred):
    qq_plot = sm.qqplot(y_valid - y_valid_pred, stats.t, fit=True, line='45')
    qq_plot.show()


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100,
                       fill='█', print_end = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
