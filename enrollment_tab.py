import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.mixture import GaussianMixture
import pickle
import threading
from queue import Queue
import time


class EnrollmentTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.training_path = tk.StringVar()
        self.destination_path = tk.StringVar()
        self.num_utterances = tk.StringVar()
        self.mfcc_features = tk.StringVar(value="22")  
        self.use_dmfcc = tk.BooleanVar()
        self.use_ddmfcc = tk.BooleanVar()
        self.n_components = tk.IntVar(value=5)
        self.processing = False
        self.queue = Queue()
        
        self.setup_ui()
        
        self.pack(expand=True, fill='both', padx=10, pady=10)
        
    def setup_ui(self):
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        
        title_label = ttk.Label(main_frame, 
                                text="Enroll Speakers From a Training Dataset", 
                                font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        
        train_frame = ttk.LabelFrame(main_frame, text="Training Data", padding="5 5 5 5")
        train_frame.pack(fill='x', padx=5, pady=5)
        
        train_entry = ttk.Entry(train_frame, textvariable=self.training_path, width=50)
        train_entry.pack(side='left', padx=5)
        ttk.Button(train_frame, text="Browse", command=self.browse_training).pack(side='left', padx=5)
        
        
        dest_frame = ttk.LabelFrame(main_frame, text="Destination", padding="5 5 5 5")
        dest_frame.pack(fill='x', padx=5, pady=5)
        
        dest_entry = ttk.Entry(dest_frame, textvariable=self.destination_path, width=50)
        dest_entry.pack(side='left', padx=5)
        ttk.Button(dest_frame, text="Browse", command=self.browse_destination).pack(side='left', padx=5)
        
        
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5 5 5 5")
        params_frame.pack(fill='x', padx=5, pady=5)
        
        
        utterances_frame = ttk.Frame(params_frame)
        utterances_frame.pack(fill='x', pady=2)
        ttk.Label(utterances_frame, text="Number of Utterances:").pack(side='left', padx=5)
        ttk.Entry(utterances_frame, textvariable=self.num_utterances, width=10).pack(side='left', padx=5)
        
        
        mfcc_frame = ttk.Frame(params_frame)
        mfcc_frame.pack(fill='x', pady=2)
        ttk.Label(mfcc_frame, text="MFCC Components:").pack(side='left', padx=5)
        ttk.Entry(mfcc_frame, textvariable=self.n_components, width=10).pack(side='left', padx=5)

        
        mfcc_frame = ttk.Frame(params_frame)
        mfcc_frame.pack(fill='x', pady=2)
        ttk.Label(mfcc_frame, text="MFCC Features:").pack(side='left', padx=5)
        ttk.Entry(mfcc_frame, textvariable=self.mfcc_features, width=10).pack(side='left', padx=5)
        
        
        ttk.Checkbutton(params_frame, text="Use DMFCC", variable=self.use_dmfcc).pack(fill='x', pady=2)
        
        
        ttk.Checkbutton(params_frame, text="Use DDMFCC", variable=self.use_ddmfcc).pack(fill='x', pady=2)
        
        
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5 5 5 5")
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(padx=5)
        
        
        ttk.Button(main_frame, text="Enroll Speakers", command=self.start_enrollment).pack(pady=10)

    def extract_features(self, audio_path, n_mfcc=22, use_dmfcc=False, use_ddmfcc=False):
        """Extract MFCC features from an audio file."""
        try:
            
            y, sr = librosa.load(audio_path, sr=None)
            
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            features = [mfccs]
            
            if use_dmfcc:
                
                dmfcc = librosa.feature.delta(mfccs)
                features.append(dmfcc)
                
            if use_ddmfcc:
                
                ddmfcc = librosa.feature.delta(mfccs, order=2)
                features.append(ddmfcc)
            
            
            combined_features = np.vstack(features)
            
            
            return combined_features.T
            
        except Exception as e:
            raise Exception(f"Error processing {audio_path}: {str(e)}")
    
    def process_speaker(self, speaker_folder, num_utterances, n_mfcc, use_dmfcc, use_ddmfcc):
        """Process a single speaker's recordings."""
        all_features = []
        
        
        audio_files = [f for f in os.listdir(speaker_folder) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        
        
        if num_utterances and num_utterances < len(audio_files):
            audio_files = np.random.choice(audio_files, num_utterances, replace=False)
        
        
        for audio_file in audio_files:
            file_path = os.path.join(speaker_folder, audio_file)
            features = self.extract_features(file_path, n_mfcc, use_dmfcc, use_ddmfcc)
            all_features.append(features)
        
        
        if all_features:
            combined_features = np.vstack(all_features)
            return combined_features
        else:
            raise Exception("No valid audio files found")
    
    def train_gmm(self, features, covariance_type='diag'):
        # To revisit later for making covariance_type a hyperparameter in the GUI
        """Train a GMM model on the extracted features."""
        gmm = GaussianMixture(n_components=self.n_components.get(), covariance_type=covariance_type, random_state=42)
        gmm.fit(features)
        return gmm
    
    def enrollment_thread(self):
        """Background thread for processing enrollment."""
        try:
            training_path = self.training_path.get()
            dest_path = self.destination_path.get()
            n_mfcc = int(self.mfcc_features.get())
            use_dmfcc = self.use_dmfcc.get()
            use_ddmfcc = self.use_ddmfcc.get()
            num_utterances = int(self.num_utterances.get()) if self.num_utterances.get() else None
            
            
            speaker_folders = [d for d in os.listdir(training_path) 
                             if os.path.isdir(os.path.join(training_path, d))]
            
            total_speakers = len(speaker_folders)
            
            for i, speaker in enumerate(speaker_folders, 1):
                speaker_path = os.path.join(training_path, speaker)
                
                
                self.queue.put(('progress', i, total_speakers))
                
                
                features = self.process_speaker(
                    speaker_path, 
                    num_utterances,
                    n_mfcc,
                    use_dmfcc,
                    use_ddmfcc
                )
                
                
                gmm = self.train_gmm(features)
                
                
                model_path = os.path.join(dest_path, f"{speaker}.gmm")
                with open(model_path, 'wb') as f:
                    pickle.dump(gmm, f)
            
            self.queue.put(('complete', None, None))
            
        except Exception as e:
            self.queue.put(('error', str(e), None))
    
    def update_progress(self):
        """Update the GUI with progress from the processing thread."""
        try:
            while True:
                msg_type, value, total = self.queue.get_nowait()
                
                if msg_type == 'progress':
                    progress = (value / total) * 100
                    self.progress_bar['value'] = progress
                    self.progress_label['text'] = f"Processing speaker {value}/{total}"
                elif msg_type == 'complete':
                    self.progress_bar['value'] = 100
                    self.progress_label['text'] = "Enrollment complete!"
                    messagebox.showinfo("Success", "Speaker enrollment completed successfully!")
                    self.processing = False
                    return
                elif msg_type == 'error':
                    self.progress_label['text'] = "Error occurred"
                    messagebox.showerror("Error", f"An error occurred: {value}")
                    self.processing = False
                    return
                
        except:
            pass
        
        if self.processing:
            self.after(100, self.update_progress)
    
    def start_enrollment(self):
        """Start the enrollment process."""
        
        if not self.training_path.get() or not self.destination_path.get():
            messagebox.showerror("Error", "Please select both training and destination folders")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
        
        try:
            int(self.mfcc_features.get())
        except ValueError:
            messagebox.showerror("Error", "Number of MFCC features must be a valid integer")
            return
            
        
        self.processing = True
        self.progress_bar['value'] = 0
        self.progress_label['text'] = "Starting enrollment..."
        
        
        thread = threading.Thread(target=self.enrollment_thread)
        thread.daemon = True
        thread.start()
        
        
        self.update_progress()
    
    def browse_training(self):
        folder = filedialog.askdirectory(title="Select Training Data Folder")
        if folder:
            self.training_path.set(folder)
            
            min_recordings = self.get_min_recordings(folder)
            self.num_utterances.set(str(min_recordings))
    
    def browse_destination(self):
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.destination_path.set(folder)
    
    def get_min_recordings(self, folder):
        min_count = float('inf')
        for speaker_folder in os.listdir(folder):
            full_path = os.path.join(folder, speaker_folder)
            if os.path.isdir(full_path):
                recordings = len([f for f in os.listdir(full_path) 
                                if f.lower().endswith(('.wav', '.mp3', '.flac'))])
                min_count = min(min_count, recordings)
        return min_count if min_count != float('inf') else 0
