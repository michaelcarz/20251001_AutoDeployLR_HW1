# dev_log

- Generated `app.py` after the prompt: "Write a Streamlit single-file interactive linear regression teaching app with data generation, normal equation, gradient descent, sklearn models, metrics, plots, comments, validation." 
- Created `requirements.txt` after the prompt: "List stable versions of streamlit, numpy, pandas, scikit-learn, plotly." 
- Authored `README.md` directly in English covering project overview, quick start, interface details, CRISP-DM walkthrough, FAQ, and license notes.
- Provided a troubleshooting guide for launching the app: create/activate a virtual environment, run `pip install -r requirements.txt`, and start with `streamlit run app.py` or `python -m streamlit run app.py` when the CLI command is missing.
- Helped resolve the Windows "streamlit not recognized" error by suggesting activation of the venv and using the module invocation as a fallback.
- Guided the installation of Git on Windows (installer vs `winget`), verified the PATH, and corrected the mistyped `git --version.` command.
- Outlined the full Git workflow: `git init`, `git branch -M main`, staged project files, committed, configured the remote, and pushed to GitHub.
- Diagnosed and fixed the malformed remote URL plus missing branch by removing the bad remote, re-adding `origin`, creating `main`, and retrying `git push -u origin main`.
- Explained Streamlit Cloud deployment: select the GitHub repo, set `app.py` as the main file, point dependencies to `requirements.txt`, and redeploy/clear cache when dependencies change.
- Troubleshot the cloud `ModuleNotFoundError` for Plotly by verifying the requirements file path and ensuring Streamlit Cloud rebuilds with the dependency list.
- Rewrote all UI copy in `app.py` to match the reference English app at `https://aiotda.streamlit.app/`, including sidebar controls, metrics labels, and teaching notes.
- Logged instructions on how to update Streamlit Cloud after code changes (commit, push, trigger auto-redeploy or manually clear cache).
- Expanded this `dev_log.md` to capture the support interactions and configuration steps taken throughout the setup.
