# Movie Recommendation System

This repository contains a movie recommendation system built using content-based filtering techniques.

## Project Structure

- `data/`: Contains the raw datasets
  - `tmdb_5000_credits.csv`: Movie credits data
  - `tmdb_5000_movies.csv`: Movie details data

- `models/`: Contains the trained models and processed data
  - `similarity_matrix.pkl`: Pre-computed similarity matrix
  - `movies_data.pkl`: Processed movies data

- `scripts/`: Contains the source code
  - `movie_recommendation.py`: Main application script
  - `movie_recommendation_system.ipynb`: Jupyter notebook with the development process
  - `fix_widgets.py`: Helper script to fix widget issues
  - `fix_ipywidgets.bat`: Batch script to fix ipywidgets installation
  - `run_movie_recommendation.bat`: Batch script to run the application

- `docs/`: Contains documentation
  - `README.md`: Detailed documentation
  - `images/`: Directory for documentation images

- `movie_posters/`: Directory for cached movie poster images

- `.venv/`: Python virtual environment (automatically created when running the application)

## Quick Start

1. Navigate to the `scripts` directory
2. Run `run_movie_recommendation.bat`

For more detailed instructions, please refer to the [full documentation](docs/README.md). 