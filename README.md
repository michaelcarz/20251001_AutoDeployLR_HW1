# Interactive Linear Regression Teaching App

## Project Overview
This project delivers an interactive Streamlit application (`app.py`) to help learners grasp linear regression concepts. By tweaking slope, intercept, noise, and sample size parameters—and switching between different training methods with built-in visualizations—you can observe how model estimates and evaluation metrics respond in real time.

## Quick Start
1. Create and activate a virtual environment (e.g., `python -m venv venv` or Conda).
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the app: `streamlit run app.py`.

## Interface and Features
- **Parameter Controls**: Adjust slope `a`, intercept `b`, noise standard deviation `noise_std`, number of samples `n_points`, and optionally add outliers via the sidebar.
- **Model Options**: Compare three training approaches—handmade closed-form solution, handmade gradient descent, and scikit-learn—to see how their estimates differ.
- **Visualizations**: The main chart shows scatter data overlaid with the true line and predicted line; the residual chart plots `y_true - y_pred` to reveal systematic patterns.
- **Evaluation Metrics**: Instantly compute and display MSE, MAE, and R² to understand model quality and stability.

## CRISP-DM Summary
- **Business Understanding**: Clarify the teaching goal—equip learners with intuition for linear regression parameter estimation and evaluation. Ideal for classroom demos or self-study, leveraging interactivity to reinforce key ideas.
- **Data Understanding**: Data is synthetically generated with uniformly sampled `x` and Gaussian noise added to `y`; optional outliers simulate real-world irregularities. Parameter sliders let users explore distribution shifts and noise effects.
- **Data Preparation**: The app trains directly on the generated data without train/test splits or feature scaling. The generation process is fully controlled, so extra prep steps are unnecessary and would distract from the instructional focus.
- **Modeling**: Offers closed-form, gradient descent, and scikit-learn models side by side, highlighting their respective strengths, limitations, and convergence behavior, including live loss tracking for gradient descent.
- **Evaluation**: Uses MSE, MAE, and R² to assess predictive performance, with the residual plot assisting in diagnosing systematic errors or outlier influence. Users can rerun experiments to strengthen metric interpretation.
- **Deployment**: Deployed as a single-file Streamlit app for easy sharing and presentation; can be extended with additional features, models, or custom data ingestion to grow into a richer teaching platform.

## FAQ
- **Import Errors**: Ensure the virtual environment is active and packages from `requirements.txt` are installed.
- **Version Conflicts**: If dependency issues arise, adjust versions in `requirements.txt` or update/downgrade the conflicting libraries as suggested by error messages.
- **Outlier Toggle**: To disable outliers, simply uncheck the "Add outliers" option in the sidebar.

## License and Instruction Notice
This project is intended solely for coursework and educational demonstrations. Do not use it for commercial purposes or redistribute without permission.
