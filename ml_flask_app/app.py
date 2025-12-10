# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np

from preprocessing import preprocess_dataframe
from kmeans_custom import KMeansCustom, compute_wcss_for_ks, choose_k_elbow
from regression_models import run_regression_models, evaluate_and_plot_regression

import matplotlib
matplotlib.use('Agg')  # for headless servers
import matplotlib.pyplot as plt

# config
UPLOAD_FOLDER = "data"
RESULTS_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


def save_plot(fig, name_prefix):
    filename = f"{name_prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return filename


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # handle file upload
    file = request.files.get("csvfile")
    if file and file.filename.endswith(".csv"):
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(saved_path)
    else:
        # fallback: try to use default dataset in /data/product_sales.csv
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], "product_sales.csv")
        if not os.path.exists(saved_path):
            return "No CSV uploaded and default dataset not found in data/product_sales.csv", 400

    # user parameters (target and optionally chosen k)
    target = request.form.get("target", "profit")  # profit or units_sold
    chosen_k = request.form.get("k_choice", "").strip()
    chosen_k = int(chosen_k) if chosen_k.isdigit() else None

    # read
    df = pd.read_csv(saved_path)

    # preprocessing
    df_clean, preprocess_summary = preprocess_dataframe(df.copy())

    # K-means WCSS for k=2..8
    features_for_kmeans = ["price", "cost", "units_sold", "promotion_frequency", "shelf_level"]
    X_k = df_clean[features_for_kmeans].values

    wcss = compute_wcss_for_ks(X_k, ks=range(2, 9))
    # elbow detection
    auto_k = choose_k_elbow(wcss)
    k_to_use = chosen_k or auto_k

    # save elbow plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ks = list(range(2, 9))
    ax1.plot(ks, wcss, '-o')
    ax1.set_xlabel("k")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow curve (k vs WCSS)")
    elbow_fname = save_plot(fig1, "elbow")

    # run kmeans with k_to_use
    kmeans = KMeansCustom(n_clusters=k_to_use, max_iter=200, tol=1e-4, random_state=42)
    cluster_labels = kmeans.fit_predict(X_k)
    df_clean["cluster"] = cluster_labels

    # Create dataframe with original (non-standardized) values for display statistics
    # df_clean has standardized values (z-scores) which can be negative
    # We need original values for meaningful statistics display
    df_for_stats = df.copy()
    
    # Apply same preprocessing steps EXCEPT standardization (to match df_clean row order)
    # Step 1: Drop rows with too many missing values
    df_for_stats['n_missing_row'] = df_for_stats.isna().sum(axis=1)
    df_for_stats = df_for_stats[df_for_stats['n_missing_row'] < 3].drop(columns=['n_missing_row'])
    
    # Step 2: Impute missing values
    NUMERIC_FEATURES = ["price", "cost", "units_sold", "promotion_frequency", "shelf_level", "profit"]
    for col in NUMERIC_FEATURES:
        if col in df_for_stats.columns:
            if df_for_stats[col].isna().sum() > 0:
                median = df_for_stats[col].median()
                df_for_stats[col] = df_for_stats[col].fillna(median)
    
    for col in df_for_stats.select_dtypes(include=['object', 'category']).columns:
        if df_for_stats[col].isna().sum() > 0:
            mode_val = df_for_stats[col].mode()
            if len(mode_val) > 0:
                df_for_stats[col] = df_for_stats[col].fillna(mode_val.iloc[0])
    
    # Step 3: Apply outlier capping
    for col in NUMERIC_FEATURES:
        if col not in df_for_stats.columns:
            continue
        q1 = df_for_stats[col].quantile(0.25)
        q3 = df_for_stats[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_for_stats[col] = df_for_stats[col].clip(lower=lower, upper=upper)
    
    # Step 4: Assign clusters (same order as df_clean)
    df_for_stats["cluster"] = cluster_labels
    
    # compute cluster statistics using original (non-standardized) values
    cluster_stats_df = df_for_stats.groupby("cluster").agg({
        "product_id": "count",
        "price": "mean",
        "units_sold": "mean",
        "profit": "mean",
        "promotion_frequency": "mean"
    }).rename(columns={"product_id": "n_products"}).reset_index()
    
    # Calculate overall averages for comparison (using original values)
    overall_avg_price = df_for_stats["price"].mean()
    overall_avg_units = df_for_stats["units_sold"].mean()
    overall_avg_profit = df_for_stats["profit"].mean()
    
    # Name clusters and generate business insights
    def name_cluster_and_insights(row):
        """Generate cluster name and business insights based on characteristics"""
        price = row["price"]
        units = row["units_sold"]
        profit = row["profit"]
        n_products = row["n_products"]
        promo_freq = row["promotion_frequency"]
        
        # Determine characteristics
        is_low_price = price < overall_avg_price
        is_high_price = price > overall_avg_price
        is_high_volume = units > overall_avg_units
        is_low_volume = units < overall_avg_units
        is_high_profit = profit > overall_avg_profit
        is_low_profit = profit < overall_avg_profit
        
        # Generate cluster name
        if is_low_price and is_high_volume:
            name = "Budget Best-Sellers"
            characteristics = "Low-price, high-volume products"
            insight = "Focus on maintaining stock levels and supply chain efficiency. These are volume-driven products."
        elif is_high_price and is_low_volume:
            name = "Premium Low-Volume"
            characteristics = "High-price, specialty items"
            insight = "Consider targeted promotions and premium marketing. These are niche, high-margin products."
        elif is_low_price and is_low_volume:
            name = "Budget Underperformers"
            characteristics = "Low-price, low-volume products"
            insight = "Review product positioning and marketing strategy. May need promotion or discontinuation."
        elif is_high_price and is_high_volume:
            name = "Premium Best-Sellers"
            characteristics = "High-price, high-volume products"
            insight = "These are your star products. Maintain quality and consider expanding product line."
        elif is_high_profit and is_high_volume:
            name = "High-Profit Volume Leaders"
            characteristics = "High-profit, high-volume products"
            insight = "Maximize inventory and marketing investment. These drive significant revenue."
        elif is_high_profit and is_low_volume:
            name = "High-Margin Specialties"
            characteristics = "High-profit, low-volume products"
            insight = "Premium positioning works. Consider expanding to similar market segments."
        elif is_low_profit and is_high_volume:
            name = "Volume-Driven Low-Margin"
            characteristics = "Low-profit, high-volume products"
            insight = "Optimize costs and pricing strategy. Volume compensates but margins need improvement."
        else:
            name = "Mid-Range Steady"
            characteristics = "Average price, volume, and profit"
            insight = "Stable performers. Monitor trends and consider incremental improvements."
        
        return {
            "cluster": int(row["cluster"]),
            "n_products": int(n_products),
            "price": round(price, 3),
            "units_sold": round(units, 3),
            "profit": round(profit, 3),
            "promotion_frequency": round(promo_freq, 3),
            "cluster_name": name,
            "characteristics": characteristics,
            "business_insight": insight
        }
    
    # Apply naming and insights to each cluster
    cluster_stats = [name_cluster_and_insights(row) for _, row in cluster_stats_df.iterrows()]

    # cluster scatter plot (price vs units_sold) - use original values for visualization
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    scatter = ax2.scatter(df_for_stats["price"], df_for_stats["units_sold"], c=df_for_stats["cluster"], cmap="tab10", s=35, alpha=0.8)
    # Note: Centroids are in standardized space, so we skip showing them on original scale
    # (showing them would require inverse transformation which is complex with multiple features)
    ax2.set_xlabel("price")
    ax2.set_ylabel("units_sold")
    ax2.set_title(f"Clusters (k={k_to_use}) — price vs units_sold")
    ax2.legend()
    cluster_scatter_fname = save_plot(fig2, "clusters")

    # Regression
    # run two models and get evaluation and a plot for chosen target
    # Use original (non-standardized) data for regression - models will handle scaling internally if needed
    regression_results = run_regression_models(df_for_stats.copy(), target=target)
    # regression_results includes predictions and metrics and a matplotlib figure
    reg_plot_fig = regression_results["plot_fig"]
    reg_plot_fname = save_plot(reg_plot_fig, "regression_actual_vs_predicted")
    residual_plot_fig = regression_results["residual_plot_fig"]
    residual_plot_fname = save_plot(residual_plot_fig, "regression_residuals")
    metrics_table = regression_results["metrics"]
    comparison_analysis = regression_results["comparison_analysis"]

    # prepare small dataset preview (use original values, not standardized)
    preview_html = df_for_stats.head(20).to_html(classes="table table-sm table-striped", index=False)

    # build results object for template
    results = {
        "preprocess_summary": preprocess_summary,
        "elbow_plot": elbow_fname,
        "auto_k": int(auto_k),
        "used_k": int(k_to_use),
        "cluster_stats": cluster_stats,
        "cluster_scatter_plot": cluster_scatter_fname,
        "regression_plot": reg_plot_fname,
        "residual_plot": residual_plot_fname,
        "regression_metrics": metrics_table,
        "comparison_analysis": comparison_analysis,
        "target": target,
        "preview_html": preview_html,
    }

    return render_template("results.html", results=results)


@app.route('/static/results/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
