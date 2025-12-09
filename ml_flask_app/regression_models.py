# regression_models.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_features(df, target):
    # choose reasonable features (avoid target leak)
    # we'll use numeric columns excluding target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    # also remove product_id if numeric
    if 'product_id' in numeric_cols:
        numeric_cols.remove('product_id')
    X = df[numeric_cols].fillna(0).values
    return X, df[target].values, numeric_cols

def run_regression_models(df, target='profit'):
    X, y, feature_names = prepare_features(df, target)

    # split (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = {}

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr_train = lr.predict(X_train)
    pred_lr = lr.predict(X_test)
    mse_lr_train = mean_squared_error(y_train, pred_lr_train)
    mse_lr = mean_squared_error(y_test, pred_lr)
    mae_lr = mean_absolute_error(y_test, pred_lr)

    # Model 2: Polynomial Regression (degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    pred_poly_train = lr_poly.predict(X_train_poly)
    pred_poly = lr_poly.predict(X_test_poly)
    mse_poly_train = mean_squared_error(y_train, pred_poly_train)
    mse_poly = mean_squared_error(y_test, pred_poly)
    mae_poly = mean_absolute_error(y_test, pred_poly)

    metrics = {
        'LinearRegression': {
            'MSE': float(mse_lr), 
            'MAE': float(mae_lr),
            'MSE_train': float(mse_lr_train)
        },
        'PolynomialDegree2': {
            'MSE': float(mse_poly), 
            'MAE': float(mae_poly),
            'MSE_train': float(mse_poly_train)
        }
    }

    # pick best by MSE (lower better)
    best_name = min(metrics.keys(), key=lambda k: metrics[k]['MSE'])

    # Model comparison analysis
    comparison_analysis = {}
    comparison_analysis['best_model_key'] = best_name  # Store the key for template matching
    
    # Determine which model performs better and why
    if best_name == 'LinearRegression':
        mse_diff = mse_poly - mse_lr
        mae_diff = mae_poly - mae_lr
        comparison_analysis['best_model'] = 'Linear Regression'
        comparison_analysis['why_better'] = (
            f"Linear Regression performs better with MSE = {mse_lr:.4f} vs Polynomial's {mse_poly:.4f}. "
            f"This suggests the relationship between features and {target} is approximately linear. "
            f"Polynomial features may be adding unnecessary complexity without capturing meaningful non-linear patterns."
        )
    else:
        mse_diff = mse_lr - mse_poly
        mae_diff = mae_lr - mae_poly
        comparison_analysis['best_model'] = 'Polynomial Regression (degree=2)'
        comparison_analysis['why_better'] = (
            f"Polynomial Regression performs better with MSE = {mse_poly:.4f} vs Linear's {mse_lr:.4f}. "
            f"This indicates non-linear relationships exist between features and {target}. "
            f"The polynomial features capture interactions and squared terms that improve prediction accuracy."
        )
    
    # Check for overfitting
    lr_overfit_ratio = mse_lr / mse_lr_train if mse_lr_train > 0 else 1.0
    poly_overfit_ratio = mse_poly / mse_poly_train if mse_poly_train > 0 else 1.0
    
    comparison_analysis['overfitting_analysis'] = {
        'LinearRegression': {
            'train_mse': float(mse_lr_train),
            'test_mse': float(mse_lr),
            'ratio': float(lr_overfit_ratio),
            'assessment': 'No significant overfitting' if lr_overfit_ratio < 1.5 else 'Possible overfitting detected'
        },
        'PolynomialDegree2': {
            'train_mse': float(mse_poly_train),
            'test_mse': float(mse_poly),
            'ratio': float(poly_overfit_ratio),
            'assessment': 'No significant overfitting' if poly_overfit_ratio < 1.5 else 'Possible overfitting detected'
        }
    }
    
    # Tradeoffs
    comparison_analysis['tradeoffs'] = {
        'LinearRegression': {
            'pros': [
                'Simple and interpretable',
                'Fast training and prediction',
                'Less prone to overfitting',
                'Lower computational cost'
            ],
            'cons': [
                'Assumes linear relationships',
                'May miss non-linear patterns',
                'Limited flexibility'
            ]
        },
        'PolynomialDegree2': {
            'pros': [
                'Captures non-linear relationships',
                'Can model feature interactions',
                'More flexible than linear model'
            ],
            'cons': [
                'More complex and harder to interpret',
                'Higher computational cost',
                'More prone to overfitting',
                'Requires more data'
            ]
        }
    }

    # create actual vs predicted plot for best model
    if best_name == 'LinearRegression':
        preds = pred_lr
        model_obj = lr
    else:
        preds = pred_poly
        model_obj = lr_poly

    # Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.7, s=50)
    # diagonal line showing perfect prediction
    minv = min(y_test.min(), preds.min())
    maxv = max(y_test.max(), preds.max())
    ax.plot([minv, maxv], [minv, maxv], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"Actual vs Predicted ({best_name})", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - preds
    fig_residual, ax_residual = plt.subplots(figsize=(6, 5))
    ax_residual.scatter(preds, residuals, alpha=0.7, s=50)
    ax_residual.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    ax_residual.set_xlabel("Predicted Values", fontsize=12)
    ax_residual.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    ax_residual.set_title(f"Residual Plot ({best_name})", fontsize=14, fontweight='bold')
    ax_residual.legend()
    ax_residual.grid(True, alpha=0.3)

    results['models'] = {
        'LinearRegression': lr,
        'PolynomialDegree2': lr_poly
    }
    results['metrics'] = metrics
    results['best_model'] = best_name
    results['plot_fig'] = fig
    results['residual_plot_fig'] = fig_residual
    results['comparison_analysis'] = comparison_analysis
    return results


def evaluate_and_plot_regression(model, X_test, y_test):
    # helper if you want more granular evaluation (not used directly by app)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return {'mse': mse, 'mae': mae, 'fig': fig}
