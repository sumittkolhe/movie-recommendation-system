# Movie Posters Directory

This directory is used to store cached movie poster images downloaded from TMDB API.

When the application runs and retrieves movie poster images, they will be saved here to:
1. Reduce API calls on subsequent runs
2. Allow the application to work offline after initial use
3. Improve application performance

Images are saved with filenames matching their TMDB movie IDs. 