import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import librosa
import noisereduce as nr
from queue import Queue
import threading
import time
from sklearn.mixture import GaussianMixture
import pickle


class TrainOnFlyTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.recording_duration = tk.StringVar(value="15")
        self.segment_duration = tk.StringVar(value="3")
        self.speaker_name = tk.StringVar()
        self.sampling_rate = tk.StringVar(value="16000")
        self.remove_silence = tk.BooleanVar()
        self.reduce_noise = tk.BooleanVar(value=False)
        self.normalize_audio = tk.BooleanVar(value=True)
        self.enroll_speaker = tk.BooleanVar()
        self.models_path = tk.StringVar(value="Speaker Models")
        
        self.is_recording = False
        self.processing = False
        self.queue = Queue()
        
        
        self.mfcc_features = tk.StringVar(value="22")
        self.n_components = tk.IntVar(value=5)
        self.use_dmfcc = tk.BooleanVar()
        self.use_ddmfcc = tk.BooleanVar()
        
        self.setup_ui()
        self.pack(expand=True, fill='both', padx=10, pady=10)
    
    def setup_ui(self):
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        
        title_label = ttk.Label(main_frame, 
                                text="Train Speaker Model On The Fly", 
                                font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        
        recording_frame = ttk.LabelFrame(main_frame, text="Recording", padding="10")
        recording_frame.pack(fill='x', padx=5, pady=5)
        
        
        name_frame = ttk.Frame(recording_frame)
        name_frame.pack(fill='x', pady=2)
        ttk.Label(name_frame, text="Speaker Name:").pack(side='left', padx=5)
        ttk.Entry(name_frame, textvariable=self.speaker_name).pack(side='left', padx=5)
        
        
        duration_frame = ttk.Frame(recording_frame)
        duration_frame.pack(fill='x', pady=2)
        ttk.Label(duration_frame, text="Recording Duration (seconds):").pack(side='left', padx=5)
        ttk.Entry(duration_frame, textvariable=self.recording_duration, width=10).pack(side='left', padx=5)
        
        
        sampling_frame = ttk.Frame(recording_frame)
        sampling_frame.pack(fill='x', pady=2)
        ttk.Label(sampling_frame, text="Sampling Rate (Hz):").pack(side='left', padx=5)
        ttk.Entry(sampling_frame, textvariable=self.sampling_rate, width=10).pack(side='left', padx=5)
        
        
        segment_frame = ttk.Frame(recording_frame)
        segment_frame.pack(fill='x', pady=2)
        ttk.Label(segment_frame, text="Segment Duration (seconds):").pack(side='left', padx=5)
        ttk.Entry(segment_frame, textvariable=self.segment_duration, width=10).pack(side='left', padx=5)
        
        
        preprocess_frame = ttk.LabelFrame(main_frame, text="Preprocessing Options", padding="5 5 5 5")
        preprocess_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(preprocess_frame, text="Remove Silence", variable=self.remove_silence).pack(fill='x', pady=2)
        ttk.Checkbutton(preprocess_frame, text="Reduce Noise", variable=self.reduce_noise).pack(fill='x', pady=2)
        ttk.Checkbutton(preprocess_frame, text="Normalize Audio", variable=self.normalize_audio).pack(fill='x', pady=2)
        
        
        enroll_frame = ttk.LabelFrame(main_frame, text="Enrollment Options", padding="5 5 5 5")
        enroll_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(enroll_frame, text="Enroll Speaker After Recording", 
                       variable=self.enroll_speaker).pack(fill='x', pady=2)
        
        
        self.models_dir_frame = ttk.Frame(enroll_frame)
        ttk.Label(self.models_dir_frame, text="Models Directory:").pack(side='left', padx=5)
        ttk.Entry(self.models_dir_frame, textvariable=self.models_path).pack(side='left', padx=5, expand=True, fill='x')
        ttk.Button(self.models_dir_frame, text="Browse", command=self.browse_models_dir).pack(side='left', padx=5)
        
        
        self.mfcc_frame = ttk.Frame(enroll_frame)

        
        ttk.Label(self.mfcc_frame, text="MFCC Components:").pack(side='left', padx=5)
        ttk.Entry(self.mfcc_frame, textvariable=self.n_components, width=10).pack(side='left', padx=5)
        
        ttk.Label(self.mfcc_frame, text="MFCC Features:").pack(side='left', padx=5)
        ttk.Entry(self.mfcc_frame, textvariable=self.mfcc_features, width=10).pack(side='left', padx=5)
        
        ttk.Checkbutton(self.mfcc_frame, text="Use DMFCC", variable=self.use_dmfcc).pack(side='left', padx=5)
        ttk.Checkbutton(self.mfcc_frame, text="Use DDMFCC", variable=self.use_ddmfcc).pack(side='left', padx=5)
        
        self.enroll_speaker.trace('w', self.toggle_enrollment_options)
        
        
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5 5 5 5")
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(padx=5)
        
        
        self.record_button = ttk.Button(main_frame, text="Start Recording", command=self.start_recording)
        self.record_button.pack(pady=10)
    
    def toggle_mfcc_options(self, *args):
        if self.enroll_speaker.get():
            self.mfcc_frame.pack(fill='x', pady=2)
        else:
            self.mfcc_frame.pack_forget()
    
    def start_recording(self):
        if not self.speaker_name.get():
            messagebox.showerror("Error", "Please enter a speaker name")
            return
            
        try:
            duration = float(self.recording_duration.get())
            segment_dur = float(self.segment_duration.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid durations")
            return
        
        if self.is_recording:
            messagebox.showwarning("Warning", "Recording is already in progress")
            return
        
        self.is_recording = True
        self.record_button.configure(state='disabled')
        self.progress_bar['value'] = 0
        self.progress_label['text'] = "Recording..."
        
        
        thread = threading.Thread(target=self.record_audio)
        thread.daemon = True
        thread.start()
        
        
        self.update_progress()
    
    def toggle_enrollment_options(self, *args):
        """Show/hide enrollment-related options based on checkbox state"""
        if self.enroll_speaker.get():
            self.models_dir_frame.pack(fill='x', pady=2)
            self.mfcc_frame.pack(fill='x', pady=2)
        else:
            self.models_dir_frame.pack_forget()
            self.mfcc_frame.pack_forget()

    def record_audio(self):
        try:
            duration = float(self.recording_duration.get())
            fs = int(self.sampling_rate.get())  
            
            
            self.queue.put(('status', "Recording...", None))
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            
            
            speaker_dir = os.path.join("recordings", self.speaker_name.get())
            os.makedirs(speaker_dir, exist_ok=True)
            
            
            self.queue.put(('status', "Processing audio...", None))
            processed_audio = self.process_audio(recording.flatten(), fs)
            
            
            segments = self.segment_audio(processed_audio, fs)
            
            
            self.queue.put(('status', "Saving segments...", None))
            for i, segment in enumerate(segments):
                filename = os.path.join(speaker_dir, f"segment_{i+1}.wav")
                sf.write(filename, segment, fs)
            
            
            if self.enroll_speaker.get():
                self.queue.put(('status', "Enrolling speaker...", None))
                self.enroll_speaker_model(speaker_dir)
            
            self.queue.put(('complete', None, None))
            
        except Exception as e:
            self.queue.put(('error', str(e), None))
    
    def process_audio(self, audio, fs):
        """Apply selected preprocessing steps to audio."""
        if self.remove_silence.get():
            
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        if self.reduce_noise.get():
            
            audio = nr.reduce_noise(y=audio, sr=fs)
        
        if self.normalize_audio.get():
            
            audio = librosa.util.normalize(audio)
        
        return audio
    
    def segment_audio(self, audio, fs):
        """Segment audio into fixed-length segments."""
        segment_length = int(float(self.segment_duration.get()) * fs)
        segments = []
        
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i + segment_length]
            if len(segment) == segment_length:  
                segments.append(segment)
        
        return segments
    
    def browse_models_dir(self):
        directory = filedialog.askdirectory(title="Select Models Directory")
        if directory:
            self.models_path.set(directory)
    
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
    
    def train_gmm(self, features, covariance_type='diag'):
        """Train a GMM model on the extracted features."""
        gmm = GaussianMixture(n_components=self.n_components.get(), covariance_type=covariance_type, random_state=42)
        gmm.fit(features)
        return gmm

    def enroll_speaker_model(self, speaker_dir):
        """Enroll speaker using the recorded segments."""
        try:
            n_mfcc = int(self.mfcc_features.get())
            use_dmfcc = self.use_dmfcc.get()
            use_ddmfcc = self.use_ddmfcc.get()
            
            
            audio_files = [f for f in os.listdir(speaker_dir) 
                         if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            all_features = []
            for audio_file in audio_files:
                file_path = os.path.join(speaker_dir, audio_file)
                features = self.extract_features(file_path, n_mfcc, use_dmfcc, use_ddmfcc)
                all_features.append(features)
            
            if all_features:
                
                combined_features = np.vstack(all_features)
                
                
                gmm = self.train_gmm(combined_features)
                
                
                models_dir = self.models_path.get()
                os.makedirs(models_dir, exist_ok=True)
                
                
                model_path = os.path.join(models_dir, f"{self.speaker_name.get()}.gmm")
                with open(model_path, 'wb') as f:
                    pickle.dump(gmm, f)
            else:
                raise Exception("No valid audio segments found")
            
        except Exception as e:
            raise Exception(f"Error during enrollment: {str(e)}")
    
    def update_progress(self):
        """Update the GUI with progress from the processing thread."""
        progress_step = 10*int(self.recording_duration.get())
        try:
            while True:
                msg_type, value, total = self.queue.get_nowait()
                
                if msg_type == 'status':
                    self.progress_label['text'] = value
                elif msg_type == 'complete':
                    self.progress_bar['value'] = 100
                    self.progress_label['text'] = "Processing complete!"
                    self.is_recording = False
                    self.record_button.configure(state='normal')
                    messagebox.showinfo("Success", "Recording and processing completed successfully!")
                    return
                elif msg_type == 'error':
                    self.progress_label['text'] = "Error occurred"
                    self.is_recording = False
                    self.record_button.configure(state='normal')
                    messagebox.showerror("Error", f"An error occurred: {value}")
                    return
                
        except:
            pass
        
        if self.is_recording:
            if self.progress_bar['value'] < 100:
                self.progress_bar['value'] += 1
            self.after(progress_step, self.update_progress)
