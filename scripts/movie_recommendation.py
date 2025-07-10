import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from PIL import Image, ImageTk
import io
import requests
import os
from urllib.parse import urlparse
from datetime import datetime
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.movies = None
        self.movies_final = None
        self.similarity = None
        self.cv = None
        self.top_movies = []
        # Get current directory for file paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.movie_posters_path = os.path.join(self.base_dir, "movie_posters")
        
        # Create posters directory if it doesn't exist
        if not os.path.exists(self.movie_posters_path):
            os.makedirs(self.movie_posters_path)
            
        print("Initializing Movie Recommendation System...")
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            print(f"Looking for dataset files in: {self.base_dir}")
            
            # Load the datasets with absolute paths
            movies_path = os.path.join(self.base_dir, 'tmdb_5000_movies.csv')
            credits_path = os.path.join(self.base_dir, 'tmdb_5000_credits.csv')
            
            print(f"Trying to load movies from: {movies_path}")
            print(f"Trying to load credits from: {credits_path}")
            
            # Check if files exist
            if not os.path.exists(movies_path):
                print(f"Error: Movies file not found at {movies_path}")
                return self.try_load_pickled_data()
                
            if not os.path.exists(credits_path):
                print(f"Error: Credits file not found at {credits_path}")
                return self.try_load_pickled_data()
            
            # Load the datasets
            try:
                self.movies = pd.read_csv(movies_path)
                credits = pd.read_csv(credits_path)
            except Exception as e:
                print(f"Error reading CSV files: {e}")
                return self.try_load_pickled_data()
            
            print(f"Successfully loaded movies dataset with {len(self.movies)} entries")
            print(f"Successfully loaded credits dataset with {len(credits)} entries")
            
            # Merge datasets on 'title'
            self.movies = self.movies.merge(credits, on='title')
            
            # Select relevant columns
            try:
                self.movies = self.movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'poster_path']]
            except KeyError as e:
                print(f"Error: Missing required column in dataset: {e}")
                print("Columns available:", self.movies.columns.tolist())
                return self.try_load_pickled_data()
            
            # Drop rows with missing values
            self.movies = self.movies.dropna()
            
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return self.try_load_pickled_data()
            
    def try_load_pickled_data(self):
        """Try to load previously processed data from pickle files"""
        try:
            print("Attempting to load pickled data as fallback...")
            movies_data_path = os.path.join(self.base_dir, 'movies_data.pkl')
            similarity_matrix_path = os.path.join(self.base_dir, 'similarity_matrix.pkl')
            
            if os.path.exists(movies_data_path) and os.path.exists(similarity_matrix_path):
                print(f"Found pickled data files, loading them...")
                import pickle
                
                with open(movies_data_path, 'rb') as f:
                    self.movies_final = pickle.load(f)
                    
                with open(similarity_matrix_path, 'rb') as f:
                    self.similarity = pickle.load(f)
                    
                print(f"Successfully loaded pickled data with {len(self.movies_final)} movies")
                self.top_movies = sorted(self.movies_final['title'].tolist())[:100]
                return True
            else:
                print("No pickled data files found.")
                return False
        except Exception as e:
            print(f"Error loading pickled data: {e}")
            return False
    
    def preprocess_data(self):
        """Process the data and create the model"""
        if self.movies is None:
            return False
            
        # Helper function to convert string representation of list to Python list
        def convert(text):
            try:
                return ast.literal_eval(text)
            except:
                return []  # Return empty list if parsing fails
        
        # Parse stringified lists/dictionaries
        self.movies['genres'] = self.movies['genres'].apply(convert)
        self.movies['keywords'] = self.movies['keywords'].apply(convert)
        self.movies['cast'] = self.movies['cast'].apply(convert)
        self.movies['crew'] = self.movies['crew'].apply(convert)
        
        # Extract genre names
        def extract_genres(genres_list):
            return [genre['name'].lower().replace(' ', '') for genre in genres_list]
        
        # Extract keywords
        def extract_keywords(keywords_list):
            return [keyword['name'].lower().replace(' ', '') for keyword in keywords_list]
        
        # Extract top 3 cast members
        def extract_cast(cast_list):
            top_cast = []
            for i, cast in enumerate(cast_list):
                if i < 3:  # Only include top 3 cast members
                    top_cast.append(cast['name'].lower().replace(' ', ''))
                else:
                    break
            return top_cast
        
        # Extract director
        def extract_director(crew_list):
            directors = []
            for crew_member in crew_list:
                if crew_member['job'] == 'Director':
                    directors.append(crew_member['name'].lower().replace(' ', ''))
            return directors
        
        # Apply extraction functions
        self.movies['genres'] = self.movies['genres'].apply(extract_genres)
        self.movies['keywords'] = self.movies['keywords'].apply(extract_keywords)
        self.movies['cast'] = self.movies['cast'].apply(extract_cast)
        self.movies['crew'] = self.movies['crew'].apply(extract_director)
        
        # Create tags by combining all features
        def create_tags(row):
            # Process overview text
            overview = row['overview'].lower() if isinstance(row['overview'], str) else ''
            overview_words = overview.split()
            # Keep only longer words that might be more meaningful
            overview_words = [word for word in overview_words if len(word) > 4]
            overview = ' '.join(overview_words)
            
            # Combine all features with appropriate weights (repeating important features)
            return ' '.join(row['genres']) + ' ' + \
                   ' '.join(row['keywords']) + ' ' + \
                   ' '.join(row['cast']) + ' ' + \
                   ' '.join(row['cast']) + ' ' + \
                   ' '.join(row['crew']) + ' ' + \
                   ' '.join(row['crew']) + ' ' + \
                   overview
        
        self.movies['tags'] = self.movies.apply(create_tags, axis=1)
        
        # Check if poster_path exists, if not, add an empty column
        if 'poster_path' not in self.movies.columns:
            print("'poster_path' column not found, adding empty column")
            self.movies['poster_path'] = None
        
        # Keep only required columns for recommendation
        try:
            self.movies_final = self.movies[['movie_id', 'title', 'tags', 'poster_path']]
        except KeyError as e:
            print(f"Warning: Column issue: {e}. Using only available columns.")
            # Fallback to minimum required columns
            required_cols = ['title', 'tags']
            if 'movie_id' in self.movies.columns:
                required_cols.insert(0, 'movie_id')
            else:
                # Create a dummy movie_id column if it doesn't exist
                self.movies['movie_id'] = range(len(self.movies))
                required_cols.insert(0, 'movie_id')
                
            self.movies_final = self.movies[required_cols]
            # Add empty poster_path column if missing
            if 'poster_path' not in self.movies_final.columns:
                self.movies_final['poster_path'] = None
        
        print("Creating movie vectors...")
        # Vectorize the tags
        self.cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = self.cv.fit_transform(self.movies_final['tags']).toarray()
        
        print("Calculating similarity matrix...")
        # Calculate cosine similarity matrix
        self.similarity = cosine_similarity(vectors)
        
        print(f"Model ready! Dataset contains {len(self.movies_final)} movies.")
        
        # Get top popular movies for initial display
        self.top_movies = sorted(self.movies_final['title'].tolist())[:100]
        return True
        
    def get_recommendations(self, movie_title, num_recommendations=5):
        """Get movie recommendations based on a movie title"""
        if self.movies_final is None or self.similarity is None:
            print("Error: Data not loaded or processed")
            return []
            
        try:
            # Find the movie index
            idx = self.get_movie_index(movie_title)
            if idx == -1:
                print(f"Movie '{movie_title}' not found in dataset")
                return []
                
            # Get similarity scores and sort
            sim_scores = list(enumerate(self.similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]  # Skip the movie itself
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Get recommended movies with all details
            recommendations = []
            for idx in movie_indices:
                try:
                    movie_data = self.movies_final.iloc[idx].to_dict()
                    
                    # Handle missing poster_path
                    if 'poster_path' not in movie_data or pd.isna(movie_data['poster_path']):
                        movie_data['poster_path'] = None
                        
                    recommendations.append(movie_data)
                except Exception as e:
                    print(f"Error getting movie data at index {idx}: {e}")
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
            
    def get_movie_details(self, title):
        """Get details for a specific movie"""
        if self.movies_final is None:
            return None
            
        try:
            movie_row = self.movies_final[self.movies_final['title'] == title]
            if len(movie_row) == 0:
                return None
                
            movie_data = movie_row.iloc[0].to_dict()
            
            # Handle missing poster_path
            if 'poster_path' not in movie_data or pd.isna(movie_data['poster_path']):
                movie_data['poster_path'] = None
                
            return movie_data
        except Exception as e:
            print(f"Error getting movie details: {e}")
            return None
    
    def search_movies(self, query, limit=10):
        """Search for movies containing the query string"""
        if not query:
            return self.top_movies[:limit]
            
        if self.movies_final is None:
            print("Error: No movie data available for search")
            return self.top_movies[:limit]
            
        try:
            matching_movies = []
            query = query.lower().strip()
            
            # Use pandas to find matching titles
            matching_titles = []
            for _, row in self.movies_final.iterrows():
                title = row['title']
                if isinstance(title, str) and query in title.lower():
                    matching_titles.append(title)
                    if len(matching_titles) >= limit:
                        break
            
            if matching_titles:
                print(f"Found {len(matching_titles)} movies matching '{query}'")
                return matching_titles
            else:
                print(f"No movies found matching '{query}'")
                return []
        except Exception as e:
            print(f"Error in search: {e}")
            return self.top_movies[:limit]
        
    def get_poster_url(self, poster_path):
        """Convert poster path to full URL"""
        if poster_path and not pd.isna(poster_path) and poster_path != '':
            # TMDB API poster URL format
            if poster_path.startswith('/'):
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                return poster_path  # If it's already a complete URL
        return None
        
    def download_poster(self, poster_url, movie_title):
        """Download movie poster and save locally"""
        if not poster_url:
            return None
            
        # Clean filename
        clean_title = "".join([c if c.isalnum() else "_" for c in movie_title])
        filename = os.path.join(self.movie_posters_path, f"{clean_title}.jpg")
        
        # If already downloaded, return the path
        if os.path.exists(filename):
            return filename
            
        # Download the image
        try:
            response = requests.get(poster_url, timeout=5)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
        except Exception as e:
            print(f"Error downloading poster for '{movie_title}': {e}")
        
        return None

    def get_movie_index(self, title):
        """Get the index of a movie by title"""
        try:
            indices = self.movies_final[self.movies_final['title'] == title].index
            if len(indices) > 0:
                return indices[0]
            return -1
        except Exception as e:
            print(f"Error finding movie index: {e}")
            return -1
            

class MovieRecommendationApp:
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Initialize the recommendation system
        self.rec_system = MovieRecommendationSystem()
        
        # Create UI elements
        self.setup_ui()
        
        # Start loading data in background
        self.loading_thread = threading.Thread(target=self.load_data)
        self.loading_thread.daemon = True
        self.loading_thread.start()

    def setup_ui(self):
        """Setup the UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        
        # Header
        header_frame = ttk.Frame(main_frame, padding="10")
        header_frame.pack(fill=tk.X)
        
        header_label = ttk.Label(header_frame, text="Movie Recommendation System", style="Header.TLabel")
        header_label.pack(anchor="center")
        
        # Status bar
        self.status_var = tk.StringVar(value="Loading movie data...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Search and movie selection frame
        selection_frame = ttk.Frame(main_frame, padding="10")
        selection_frame.pack(fill=tk.X, pady=5)
        
        search_frame = ttk.Frame(selection_frame)
        search_frame.pack(fill=tk.X)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<KeyRelease>", self.on_search_keyrelease)
        self.search_entry.bind("<Return>", lambda event: self.on_search())
        
        self.search_button = ttk.Button(search_frame, text="Search", command=self.on_search, state=tk.DISABLED)
        self.search_button.pack(side=tk.LEFT, padx=5)
        
        # Movie selection frame
        selection_frame = ttk.Frame(main_frame, padding="10")
        selection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(selection_frame, text="Select Movie:").pack(side=tk.LEFT, padx=5)
        
        self.movie_var = tk.StringVar()
        self.movie_combo = ttk.Combobox(selection_frame, textvariable=self.movie_var, width=50, state="readonly")
        self.movie_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Label(selection_frame, text="Number of recommendations:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.num_var = tk.IntVar(value=5)
        num_values = list(range(1, 11))
        self.num_combo = ttk.Combobox(selection_frame, textvariable=self.num_var, values=num_values, width=5, state="readonly")
        self.num_combo.pack(side=tk.LEFT, padx=5)
        
        self.recommend_button = ttk.Button(selection_frame, text="Get Recommendations", command=self.on_recommend, state=tk.DISABLED)
        self.recommend_button.pack(side=tk.LEFT, padx=20)
        
        # Results frame
        self.results_frame = ttk.Frame(main_frame, padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create recommendation frames (initially hidden)
        self.recommendation_frames = []
        for i in range(10):  # Maximum 10 recommendations
            frame = ttk.Frame(self.results_frame, padding="5", relief=tk.GROOVE, borderwidth=1)
            # Will be packed when showing recommendations
            
            # Left side for poster
            poster_frame = ttk.Frame(frame, width=150, height=225)
            poster_frame.pack(side=tk.LEFT, padx=5)
            poster_frame.pack_propagate(False)
            
            self.poster_label = ttk.Label(poster_frame)
            self.poster_label.pack(fill=tk.BOTH, expand=True)
            
            # Right side for movie details
            details_frame = ttk.Frame(frame)
            details_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            title_label = ttk.Label(details_frame, text="", style="Title.TLabel")
            title_label.pack(anchor=tk.W, pady=5)
            
            similarity_label = ttk.Label(details_frame, text="")
            similarity_label.pack(anchor=tk.W)
            
            # Store references
            frame_data = {
                'frame': frame,
                'poster_label': self.poster_label,
                'title_label': title_label,
                'similarity_label': similarity_label,
                'poster_image': None  # Will store the PhotoImage object
            }
            self.recommendation_frames.append(frame_data)
            
        # Initial loading message
        self.loading_label = ttk.Label(self.results_frame, text="Loading data and building model...\nPlease wait.", font=("Arial", 12))
        self.loading_label.pack(expand=True)

    def load_data(self):
        """Load data in a background thread"""
        try:
            self.rec_system.load_data()
            # Schedule UI update on main thread
            self.root.after(0, self.data_loaded)
        except Exception as e:
            error_msg = f"Error loading movie data: {str(e)}"
            print(error_msg)
            # Schedule error display on main thread
            self.root.after(0, lambda: self.status_var.set(error_msg))
    
    def data_loaded(self):
        """Called when data loading is complete"""
        self.status_var.set("Model ready. Select a movie to get recommendations.")
        
        # Remove loading label
        self.loading_label.pack_forget()
        
        # Enable UI elements
        self.search_button.config(state=tk.NORMAL)
        self.search_entry.config(state=tk.NORMAL)
        self.recommend_button.config(state=tk.NORMAL)
        self.movie_combo['state'] = 'readonly'
        
        # Populate dropdown with top movies
        self.movie_combo['values'] = self.rec_system.top_movies
        
        # Display welcome message
        welcome_frame = ttk.Frame(self.results_frame, padding="20")
        welcome_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(welcome_frame, text="Welcome to the Movie Recommendation System", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(welcome_frame, text="1. Search for a movie or select one from the dropdown list", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        ttk.Label(welcome_frame, text="2. Choose the number of recommendations you want", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        ttk.Label(welcome_frame, text="3. Click 'Get Recommendations' to see similar movies", 
                 font=("Arial", 12)).pack(anchor=tk.W, pady=5)
        
        # Example movies
        ttk.Label(welcome_frame, text="Popular Movies:", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        for i, movie in enumerate(self.rec_system.top_movies[:5]):
            ttk.Label(welcome_frame, text=f"• {movie}", 
                     font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    
    def on_search_keyrelease(self, event):
        """Handle search field key release"""
        if len(self.search_var.get()) >= 2:
            self.on_search()
    
    def on_search(self):
        """Search for movies based on input"""
        search_term = self.search_var.get().strip()
        if not search_term:
            return
            
        try:
            # Search for matching movies
            results = self.rec_system.search_movies(search_term, limit=20)
            
            if results:
                # Update dropdown with results
                self.movie_combo['values'] = results
                # Select first result
                self.movie_var.set(results[0])
                self.status_var.set(f"Found {len(results)} matches for '{search_term}'")
                # Show dropdown
                self.movie_combo.focus()
            else:
                self.status_var.set(f"No movies found matching '{search_term}'")
        except Exception as e:
            self.status_var.set(f"Search error: {str(e)}")
            print(f"Error in search: {e}")
            
    def on_recommend(self):
        """Get and display movie recommendations"""
        movie_title = self.movie_var.get()
        if not movie_title:
            messagebox.showinfo("Selection Required", "Please select a movie first.")
            return
            
        num_recommendations = self.num_var.get()
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Show loading message
        loading_label = ttk.Label(self.results_frame, text="Loading recommendations...", 
                                 font=("Arial", 14))
        loading_label.pack(pady=50)
        
        # Show loading status
        self.status_var.set(f"Finding movies similar to '{movie_title}'...")
        self.root.update()
        
        # Get recommendations in a background thread
        def get_recommendations():
            try:
                recommendations = self.rec_system.get_recommendations(movie_title, num_recommendations)
                self.root.after(0, lambda: self.display_recommendations(recommendations, movie_title))
            except Exception as e:
                error_msg = f"Error getting recommendations: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: self.show_error_message(error_msg))
                
        threading.Thread(target=get_recommendations, daemon=True).start()
        
    def show_error_message(self, error_msg):
        """Display error message in the results frame"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Show error message
        error_frame = ttk.Frame(self.results_frame, padding="20")
        error_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(error_frame, text="Error Getting Recommendations", 
                 font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(error_frame, text=f"Sorry, we encountered an error: {error_msg}", 
                 font=("Arial", 12)).pack(pady=5)
        ttk.Label(error_frame, text="Please try another movie or restart the application.", 
                 font=("Arial", 12)).pack(pady=5)
        
        self.status_var.set("Error getting recommendations")
        
    def display_recommendations(self, recommendations, source_movie):
        """Display movie recommendations in the UI"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        if not recommendations:
            self.show_error_message("No recommendations found")
            return
            
        # Create scrollable frame for results
        canvas = tk.Canvas(self.results_frame)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add header
        header = ttk.Label(scrollable_frame, text=f"Movies Similar to '{source_movie}'", 
                          font=("Arial", 16, "bold"))
        header.pack(pady=10, anchor=tk.CENTER)
        
        # Display each recommendation
        for i, movie in enumerate(recommendations):
            movie_frame = ttk.Frame(scrollable_frame, padding=10)
            movie_frame.pack(fill=tk.X, pady=5)
            movie_frame.configure(relief="solid", borderwidth=1)
            
            # Left side - poster
            poster_frame = ttk.Frame(movie_frame, width=100)
            poster_frame.pack(side=tk.LEFT, padx=10)
            
            # Create poster label - use tk.Label instead of ttk.Label to support height/width options
            poster_label = tk.Label(poster_frame, text="Loading...", 
                                   width=10, height=8, relief=tk.GROOVE)
            poster_label.pack()
            
            # Try to load poster image in background
            try:
                poster_path = movie.get('poster_path')
                if poster_path:
                    poster_url = self.rec_system.get_poster_url(poster_path)
                    threading.Thread(
                        target=self.load_poster_image,
                        args=(poster_url, movie.get('title', 'Unknown'), poster_label, i)
                    ).start()
                else:
                    poster_label.config(text="No Image")
            except Exception as e:
                print(f"Error setting up poster: {e}")
                poster_label.config(text="No Image")
            
            # Right side - movie info
            info_frame = ttk.Frame(movie_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            
            # Title
            title_label = ttk.Label(info_frame, text=f"{i+1}. {movie.get('title', 'Unknown Title')}", 
                                   font=("Arial", 12, "bold"))
            title_label.pack(anchor=tk.W, pady=(0, 5))
            
            # Overview (truncated)
            overview = movie.get('overview', '')
            if overview:
                if len(overview) > 200:
                    overview = overview[:200] + "..."
                overview_label = ttk.Label(info_frame, text=overview, wraplength=500)
                overview_label.pack(anchor=tk.W, pady=5)
                
            # Additional details
            details_frame = ttk.Frame(info_frame)
            details_frame.pack(fill=tk.X, pady=5)
            
            # Release date
            release_date = movie.get('release_date', '')
            if release_date:
                try:
                    release_date = release_date.split('-')[0]  # Just get the year
                except:
                    pass
                ttk.Label(details_frame, text=f"Year: {release_date}").pack(side=tk.LEFT, padx=(0, 15))
                
            # Rating
            vote_avg = movie.get('vote_average', '')
            if vote_avg:
                ttk.Label(details_frame, text=f"Rating: {vote_avg}/10").pack(side=tk.LEFT, padx=(0, 15))
                
        # Update status
        self.status_var.set(f"Found {len(recommendations)} movies similar to '{source_movie}'")
        
    def load_poster_image(self, poster_url, movie_title, poster_label, index):
        """Load poster image in background thread"""
        try:
            # Download the poster
            poster_path = self.rec_system.download_poster(poster_url, movie_title)
            
            if poster_path and os.path.exists(poster_path):
                # Open and resize image
                img = Image.open(poster_path)
                img = img.resize((150, 225), Image.LANCZOS)
                
                # Convert to PhotoImage and update label (must be done on main thread)
                photo_img = ImageTk.PhotoImage(img)
                
                # Store reference and update UI on main thread
                self.root.after(0, lambda: self.update_poster(photo_img, poster_label, index))
        except Exception as e:
            print(f"Error loading image: {e}")
            self.root.after(0, lambda: poster_label.config(text="No Image"))

    def update_poster(self, photo_img, poster_label, index):
        """Update poster image (called on main thread)"""
        # Keep a reference to prevent garbage collection
        self.recommendation_frames[index]['poster_image'] = photo_img
        poster_label.config(image=photo_img)


def main():
    try:
        # Check for PIL/Pillow
        import PIL
    except ImportError:
        print("PIL/Pillow library not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "Pillow"])
        print("Pillow installed. Restarting...")
        import PIL
    
    try:
        # Check for requests
        import requests
    except ImportError:
        print("Requests library not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests"])
        print("Requests installed. Restarting...")
        import requests
    
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 