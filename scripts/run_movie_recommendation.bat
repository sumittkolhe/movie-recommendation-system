@echo off
echo ===== Movie Recommendation System Launcher =====
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.6 or later.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn Pillow requests

echo.
echo Starting Movie Recommendation System GUI...
echo (This might take a moment as it loads and processes data)
echo.
python movie_recommendation.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo ==============================================
    echo   An error occurred running the application
    echo ==============================================
    echo.
    echo The application encountered an error and couldn't run properly.
    echo.
    echo Troubleshooting steps:
    echo 1. Make sure dataset files are in the correct location:
    echo    - tmdb_5000_movies.csv
    echo    - tmdb_5000_credits.csv
    echo.
    echo 2. Or make sure these pickled files exist:
    echo    - movies_data.pkl
    echo    - similarity_matrix.pkl
    echo.
    echo 3. Check that you have enough disk space and memory.
    echo.
    echo 4. Try running the command manually:
    echo    python movie_recommendation.py
    echo.
    echo 5. If the problem persists, try using the notebook version:
    echo    jupyter notebook movie_recommendation_system.ipynb
    echo.
    pause
) else (
    echo Application closed successfully.
) 