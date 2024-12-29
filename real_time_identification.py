import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import librosa
import pickle
import os
from pathlib import Path
from scipy.signal import butter, filtfilt
import noisereduce as nr


class RealTimeIdentificationTab(ttk.Frame):
    def __init__(self, notebook):
        super().__init__(notebook)
        self.queue = queue.Queue()
        self.is_recording = False
        self.speaker_models = {}
        self.models_dir = None
        
        
        self.sample_rate = 16000
        self.frame_length = int(0.025 * self.sample_rate)  
        self.hop_length = int(0.010 * self.sample_rate)    
        self.buffer_duration = 3  
        self.step_size = 0.1  
        self.audio_buffer = np.zeros(self.sample_rate * self.buffer_duration)
        
        
        self.mfcc_features = tk.StringVar(value="22")
        self.use_dmfcc = tk.BooleanVar(value=False)
        self.use_ddmfcc = tk.BooleanVar(value=False)
        
        
        self.reduce_noise = tk.BooleanVar(value=True)
        self.normalize_audio = tk.BooleanVar(value=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        
        title_label = ttk.Label(main_frame, text="Real-Time Speaker Identification", 
                              font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_path_label = ttk.Label(model_frame, 
                                        text="Models Directory: Not selected")
        self.model_path_label.pack(side=tk.LEFT, padx=5)
        
        self.select_dir_button = ttk.Button(model_frame, 
                                          text="Select Models Directory",
                                          command=self.select_models_directory)
        self.select_dir_button.pack(side=tk.RIGHT, padx=5)
        
        
        settings_frame = ttk.LabelFrame(main_frame, text="Processing Settings", 
                                      padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        
        feature_frame = ttk.Frame(settings_frame)
        feature_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(feature_frame, text="Feature Extraction:").pack(side=tk.LEFT, 
                                                                padx=5)
        ttk.Entry(feature_frame, textvariable=self.mfcc_features, width=10).pack(side='left', padx=5)
        ttk.Checkbutton(feature_frame, text="Use DMFCC", 
                       variable=self.use_dmfcc).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(feature_frame, text="Use DDMFCC", 
                       variable=self.use_ddmfcc).pack(side=tk.LEFT, padx=10)
        
        
        preprocess_frame = ttk.Frame(settings_frame)
        preprocess_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(preprocess_frame, text="Audio Preprocessing:").pack(side=tk.LEFT, 
                                                                    padx=5)
        ttk.Checkbutton(preprocess_frame, text="Reduce Noise", 
                       variable=self.reduce_noise).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(preprocess_frame, text="Normalize Audio", 
                       variable=self.normalize_audio).pack(side=tk.LEFT, padx=10)
        
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        
        self.toggle_button = ttk.Button(control_frame, text="Start Recording", 
                                      command=self.toggle_recording,
                                      state=tk.DISABLED)
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        
        
        self.status_label = ttk.Label(control_frame, 
                                    text="Status: Please select models directory")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        
        results_frame = ttk.LabelFrame(main_frame, text="Identification Results", 
                                     padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        
        self.prediction_label = ttk.Label(results_frame, 
                                        text="Detected Speaker: None",
                                        font=('Helvetica', 12))
        self.prediction_label.pack(pady=10)
        
        
        self.confidence_text = tk.Text(results_frame, height=10, width=50)
        self.confidence_text.pack(pady=10)
    
    def select_models_directory(self):
        directory = filedialog.askdirectory(title="Select Speaker Models Directory")
        if directory:
            self.models_dir = Path(directory)
            self.model_path_label.config(
                text=f"Models Directory: {self.models_dir}")
            self.load_models()
    
    def load_models(self):
        try:
            self.speaker_models.clear()
            model_files = list(self.models_dir.glob("*.gmm"))
            
            if not model_files:
                self.status_label.config(
                    text="Status: No .gmm files found in selected directory!")
                self.toggle_button.config(state=tk.DISABLED)
                return
            
            for model_path in model_files:
                speaker_name = model_path.stem
                with open(model_path, 'rb') as f:
                    self.speaker_models[speaker_name] = pickle.load(f)
            
            self.status_label.config(
                text=f"Status: Loaded {len(self.speaker_models)} speaker models")
            self.toggle_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_label.config(text=f"Status: Error loading models - {str(e)}")
            self.toggle_button.config(state=tk.DISABLED)
    
    def preprocess_audio(self, audio):
        if self.reduce_noise.get():
            
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate)
        
        if self.normalize_audio.get():
            
            audio = librosa.util.normalize(audio)
        
        return audio
    
    def extract_features(self, audio):
        
        n_mfcc = int(self.mfcc_features.get())
        mfccs = librosa.feature.mfcc(y=audio, 
                                   sr=self.sample_rate,
                                   n_mfcc=n_mfcc,
                                   hop_length=self.hop_length,
                                   n_fft=self.frame_length)
        
        features = [mfccs]
        
        if self.use_dmfcc.get():
            
            delta_mfccs = librosa.feature.delta(mfccs)
            features.append(delta_mfccs)
        
        if self.use_ddmfcc.get():
            
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features.append(delta2_mfccs)
        
        
        combined_features = np.concatenate(features, axis=0)
        
        
        feature_vector = np.mean(combined_features, axis=1).reshape(1, -1)
        
        return feature_vector
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if not self.speaker_models:
            self.status_label.config(text="Status: No models loaded!")
            return
            
        self.is_recording = True
        self.toggle_button.config(text="Stop Recording")
        self.status_label.config(text="Status: Recording...")
        self.select_dir_button.config(state=tk.DISABLED)
        
        
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
        
        
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_recording(self):
        self.is_recording = False
        self.toggle_button.config(text="Start Recording")
        self.status_label.config(text="Status: Stopped")
        self.select_dir_button.config(state=tk.NORMAL)
    
    def record_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            self.queue.put(indata.copy())
        
        try:
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=int(self.sample_rate * self.step_size)):
                while self.is_recording:
                    time.sleep(self.step_size)
        except Exception as e:
            print(f"Error in audio recording: {str(e)}")
            self.stop_recording()
    
    def process_audio(self):
        while self.is_recording:
            try:
                
                audio_chunk = self.queue.get(timeout=1)
                
                
                self.audio_buffer = np.roll(self.audio_buffer, 
                                          -len(audio_chunk.flatten()))
                self.audio_buffer[-len(audio_chunk.flatten()):] = \
                    audio_chunk.flatten()
                
                
                processed_audio = self.preprocess_audio(self.audio_buffer)
                
                
                feature_vector = self.extract_features(processed_audio)
                
                
                predictions = {}
                for speaker, model in self.speaker_models.items():
                    try:
                        score = model.score(feature_vector)
                        predictions[speaker] = score
                    except Exception as e:
                        print(f"Error predicting for {speaker}: {str(e)}")
                
                
                self.update_results(predictions)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {str(e)}")
    
    def update_results(self, predictions):
        if not predictions:
            return
            
        
        sorted_speakers = sorted(predictions.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        
        most_likely_speaker = sorted_speakers[0][0]
        self.prediction_label.config(
            text=f"Detected Speaker: {most_likely_speaker}")
        
        
        self.confidence_text.delete(1.0, tk.END)
        self.confidence_text.insert(tk.END, "Confidence Scores:\n\n")
        for speaker, score in sorted_speakers:
            self.confidence_text.insert(
                tk.END, f"{speaker}: {score:.2f}\n")
