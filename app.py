"""
Interactive Linear Regression Teaching App (Streamlit single-file version).

Usage:
    1. Make sure the required libraries are installed: streamlit, numpy, pandas, plotly, scikit-learn.
    2. Run locally with: streamlit run app.py
    3. Adjust controls in the sidebar to explore how slope, intercept, noise, and sample size affect
       linear regression estimates and evaluation metrics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_synthetic_data(
    slope: float,
    intercept: float,
    noise_std: float,
    n_points: int,
    include_outliers: bool,
    outlier_fraction: float,
) -> pd.DataFrame:
    """Sample synthetic data for y = slope * x + intercept with optional outliers."""
    rng = np.random.default_rng()
    x_vals = rng.uniform(-10, 10, size=n_points)
    noise = rng.normal(0.0, noise_std, size=n_points)
    y_vals = slope * x_vals + intercept + noise

    if include_outliers and outlier_fraction > 0.0:
        n_outliers = int(np.floor(outlier_fraction * n_points))
        if n_outliers > 0:
            idx = rng.choice(n_points, size=n_outliers, replace=False)
            offsets = rng.choice([-1.0, 1.0], size=n_outliers) * 5.0 * noise_std
            y_vals[idx] += offsets

    return pd.DataFrame({"x": x_vals, "y": y_vals})


def fit_normal_equation(df: pd.DataFrame) -> tuple[float, float]:
    """Closed-form solution using pseudo-inverse to guard against singular matrices."""
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    design_matrix = np.column_stack([x, np.ones_like(x)])
    theta = np.linalg.pinv(design_matrix) @ y
    return float(theta[0]), float(theta[1])


def fit_gradient_descent(
    df: pd.DataFrame,
    learning_rate: float,
    n_iters: int,
) -> tuple[float, float, list[float]]:
    """Estimate slope and intercept via manual gradient descent on MSE."""
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    n = len(x)
    w, b = 0.0, 0.0
    loss_history: list[float] = []

    for _ in range(n_iters):
        y_pred = w * x + b
        error = y_pred - y
        dw = (2.0 / n) * np.dot(error, x)
        db = (2.0 / n) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss_history.append(float(np.mean((y - y_pred) ** 2)))

    return w, b, loss_history


def fit_sklearn(df: pd.DataFrame) -> tuple[float, float]:
    """Use scikit-learn's LinearRegression for reference."""
    model = LinearRegression()
    x = df["x"].to_numpy().reshape(-1, 1)
    y = df["y"].to_numpy()
    model.fit(x, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return slope, intercept


def evaluate_model(df: pd.DataFrame, slope: float, intercept: float) -> dict[str, float]:
    """Compute regression diagnostics."""
    y_true = df["y"].to_numpy()
    y_pred = slope * df["x"].to_numpy() + intercept
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def plot_predictions(
    df: pd.DataFrame,
    true_slope: float,
    true_intercept: float,
    est_slope: float,
    est_intercept: float,
) -> go.Figure:
    """Create scatter plot with true and predicted regression lines."""
    x_grid = np.linspace(df["x"].min() - 1, df["x"].max() + 1, 200)
    true_line = true_slope * x_grid + true_intercept
    pred_line = est_slope * x_grid + est_intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            name="Observed data",
            marker=dict(size=8, color="#1f77b4", opacity=0.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=true_line,
            mode="lines",
            name="True line",
            line=dict(color="#2ca02c", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pred_line,
            mode="lines",
            name="Predicted line",
            line=dict(color="#d62728"),
        )
    )
    fig.update_layout(
        title="Scatter Plot with True & Predicted Lines",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_residuals(df: pd.DataFrame, slope: float, intercept: float) -> go.Figure:
    """Residual plot to highlight systematic deviations."""
    y_true = df["y"].to_numpy()
    y_pred = slope * df["x"].to_numpy() + intercept
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(df)),
            y=residuals,
            mode="markers+lines",
            name="Residuals",
            marker=dict(color="#9467bd", size=7),
        )
    )
    fig.add_hline(y=0.0, line=dict(color="black", dash="dot"))
    fig.update_layout(
        title="Residual Plot (y_true - y_pred)",
        xaxis_title="Sample index",
        yaxis_title="Residual",
        template="plotly_white",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Linear Regression Playground", layout="wide")
    st.title("線性回歸互動教學：y = a x + b")

    st.sidebar.header("資料與模型參數設定")

    slope = st.sidebar.slider("斜率 a", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    intercept = st.sidebar.slider(
        "截距 b", min_value=-10.0, max_value=10.0, value=0.0, step=0.5
    )
    noise_std = st.sidebar.slider(
        "雜訊標準差", min_value=0.0, max_value=5.0, value=1.0, step=0.1
    )
    n_points = st.sidebar.slider(
        "資料點數", min_value=10, max_value=2000, value=100, step=10
    )

    allow_outliers = st.sidebar.checkbox("加入離群點 (outliers)", value=False)
    outlier_fraction = 0.0
    if allow_outliers:
        outlier_fraction = st.sidebar.slider(
            "離群點比例",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
        )

    model_choice = st.sidebar.radio(
        "模型選擇",
        options=["Handmade (Normal Eq.)", "Handmade (GD)", "Sklearn"],
        index=0,
    )

    gd_params = {}
    if model_choice == "Handmade (GD)":
        st.sidebar.markdown(
            "更新公式：\n"
            "`w <- w - lr * (2/n * Σ((w*x_i + b - y_i) * x_i))`\n"
            "`b <- b - lr * (2/n * Σ(w*x_i + b - y_i))`"
        )
        gd_params["lr"] = st.sidebar.number_input(
            "學習率 lr", min_value=1e-5, max_value=1.0, value=0.01, step=0.01, format="%.4f"
        )
        gd_params["n_iters"] = st.sidebar.slider(
            "迭代次數 n_iters", min_value=100, max_value=5000, value=1000, step=100
        )

    if n_points < 2:
        st.error("需要至少 2 筆資料點才能估計線性模型。請調整資料點數。")
        st.stop()
    if noise_std < 0:
        st.error("雜訊標準差必須為非負數。請重新設定。")
        st.stop()

    data = generate_synthetic_data(
        slope,
        intercept,
        noise_std,
        n_points,
        include_outliers=allow_outliers,
        outlier_fraction=outlier_fraction,
    )

    st.subheader("資料預覽")
    st.write(data.head())

    if model_choice == "Handmade (Normal Eq.)":
        est_slope, est_intercept = fit_normal_equation(data)
        loss_history = None
    elif model_choice == "Handmade (GD)":
        est_slope, est_intercept, loss_history = fit_gradient_descent(
            data, gd_params["lr"], gd_params["n_iters"]
        )
    else:
        est_slope, est_intercept = fit_sklearn(data)
        loss_history = None

    metrics = evaluate_model(data, est_slope, est_intercept)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("真實斜率 a", f"{slope:.3f}")
        st.metric("估計斜率 â", f"{est_slope:.3f}")
    with col2:
        st.metric("真實截距 b", f"{intercept:.3f}")
        st.metric("估計截距 b̂", f"{est_intercept:.3f}")

    metric_df = pd.DataFrame(
        {
            "指標": ["MSE", "MAE", "R²"],
            "數值": [metrics["MSE"], metrics["MAE"], metrics["R²"]],
        }
    )
    st.subheader("模型評估指標")
    st.dataframe(metric_df.style.format({"數值": "{:.4f}"}), hide_index=True)

    st.subheader("資料與模型視覺化")
    st.plotly_chart(
        plot_predictions(data, slope, intercept, est_slope, est_intercept),
        use_container_width=True,
    )

    st.subheader("殘差圖 (y_true - y_pred)")
    st.plotly_chart(plot_residuals(data, est_slope, est_intercept), use_container_width=True)

    if loss_history:
        st.subheader("梯度下降 Loss 走勢")
        st.line_chart(loss_history)

    st.subheader("教學重點")
    st.markdown(
        "- 雜訊越大 (noise_std ↑) 時，資料點在真實線周圍的散佈越分散，估計的â與b̂更容易偏離真值，"
        "也會導致 MSE/MAE 上升、R² 下降。\n"
        "- 增加資料點數 (n_points ↑) 可降低估計的不確定性：雖然單點仍受雜訊影響，"
        "但平均後的參數估計會更穩定，誤差指標也更可靠。\n"
        "- 留意離群點：少量偏移即可大幅改變估計結果，建議透過殘差圖檢查異常樣本。"
    )

    st.caption("調整側欄參數觀察，並比較不同訓練方法 (手刻閉式解、GD、Sklearn) 的行為差異。")


if __name__ == "__main__":
    main()
