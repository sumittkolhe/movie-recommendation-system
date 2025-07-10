@echo off
echo ===== Fixing ipywidgets Installation =====
echo.

echo Step 1: Installing/upgrading packages...
pip install --user --upgrade notebook ipywidgets jupyterlab widgetsnbextension
echo.

echo Step 2: Enabling notebook extension...
pip install --user jupyter
echo.

echo Step 3: Trying to enable the extension...
python -m jupyter nbextension enable --py widgetsnbextension
echo.

echo Step 4: Checking ipywidgets version...
python -c "import ipywidgets; print('ipywidgets version:', ipywidgets.__version__)"
echo.

echo ===== INSTRUCTIONS =====
echo 1. Close and reopen Jupyter Notebook/Lab
echo 2. Try running your notebook again
echo 3. If still not working, try running: python -m jupyter notebook
echo.
echo If issues persist, use the standalone movie_recommendation.py script instead.

pause 