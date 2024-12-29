import os
import tkinter as tk
from tkinter import font as tkfont
from tkinter import Tk, Label, Button, filedialog, messagebox, ttk, StringVar, Entry
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr
import scipy.io.wavfile as wav
import numpy as np


def select_directory():
    global source_path
    source_path = filedialog.askdirectory(title="Select Source Folder")
    if source_path:
        source_label.config(text=f"Source Folder: .../{source_path.split('/')[-1]}")
    else:
        source_label.config(text="Source Folder: Not selected")

def select_destination():
    global destination_path
    destination_path = filedialog.askdirectory(title="Select Destination Folder")
    if destination_path:
        destination_label.config(text=f"Destination: .../{destination_path.split('/')[-1]}")
    else:
        destination_label.config(text="Destination: Not selected", width=35)

def convert_to_format():
    """Convert audio files to the selected format."""
    if not source_path:
        messagebox.showerror("Error", "Please select a source folder.")
        return
    if not destination_path:
        messagebox.showerror("Error", "Please select a destination folder.")
        return
    
   
    output_format = output_format_var.get().strip()
    if not output_format:
        output_format = "wav" 

    try:
        progress_bar["value"] = 0 
        files = []
        for root_dir, _, filenames in os.walk(source_path): 
            files.extend(os.path.join(root_dir, f) for f in filenames if not f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')))
        total_files = len(files)

        if total_files == 0:
            messagebox.showerror("Error", "No audio files to convert.")
            return

       
        for i, file_path in enumerate(files):
            audio = AudioSegment.from_file(file_path)
            rel_path = os.path.relpath(file_path, source_path)
            output_path = os.path.join(destination_path, os.path.splitext(rel_path)[0] + "." + output_format)
            os.makedirs(os.path.dirname(output_path), exist_ok=True) 
            audio.export(output_path, format=output_format)

           
            progress = (i + 1) / total_files * 100
            progress_bar["value"] = progress
            root.update_idletasks() 

        messagebox.showinfo("Success", "Folder conversion completed!")
    except Exception as e:
        print("Error", f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")


def select_directory_silence():
    global source_path_silence
    source_path_silence = filedialog.askdirectory(title="Select Source Folder")
    if source_path_silence:
        source_label_silence.config(text=f"Source Folder: .../{source_path_silence.split('/')[-1]}")
    else:
        source_label_silence.config(text="Source Folder: Not selected")

def select_destination_silence():
    global destination_path_silence
    destination_path_silence = filedialog.askdirectory(title="Select Destination Folder")
    if destination_path_silence:
        destination_label_silence.config(text=f"Destination: .../{destination_path_silence.split('/')[-1]}")
    else:
        destination_label_silence.config(text="Destination: Not selected", width=35)

def remove_silence_from_files():
    if not source_path_silence:
        messagebox.showerror("Error", "Please select a source folder.")
        return
    if not destination_path_silence:
        messagebox.showerror("Error", "Please select a destination folder.")
        return

    try:
        silence_threshold = silence_thresh_var.get()
        if not silence_threshold.strip():
            messagebox.showerror("Error", "Please enter a valid silence threshold (e.g., -50).")
            return

        silence_threshold = int(silence_threshold)

        progress_bar_silence["value"] = 0 
        files = []
        for root_dir, _, filenames in os.walk(source_path_silence): 
            files.extend(os.path.join(root_dir, f) for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac")))
        total_files = len(files)

        if total_files == 0:
            messagebox.showerror("Error", "No audio files to process.")
            return

       
        for i, file_path in enumerate(files):
            audio = AudioSegment.from_file(file_path)

           
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=silence_threshold)
            if nonsilent_ranges:
               
                processed_audio = sum(audio[start:end] for start, end in nonsilent_ranges)

               
                rel_path = os.path.relpath(file_path, source_path_silence)
                output_path = os.path.join(destination_path_silence, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                processed_audio.export(output_path, format="wav")
            else:
                print(f"File {file_path} contains only silence and was skipped.")

           
            progress = (i + 1) / total_files * 100
            progress_bar_silence["value"] = progress
            root.update_idletasks() 

        messagebox.showinfo("Success", "Silence removal completed!")
    except Exception as e:
        print("Error", f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

def select_directory_segmentation():
    global source_path_segmentation
    source_path_segmentation = filedialog.askdirectory(title="Select Source Folder")
    if source_path_segmentation:
        source_label_segmentation.config(text=f"Source Folder: .../{source_path_segmentation.split('/')[-1]}")
    else:
        source_label_segmentation.config(text="Source Folder: Not selected")

def select_destination_segmentation():
    global destination_path_segmentation
    destination_path_segmentation = filedialog.askdirectory(title="Select Destination Folder")
    if destination_path_segmentation:
        destination_label_segmentation.config(text=f"Destination: .../{destination_path_segmentation.split('/')[-1]}")
    else:
        destination_label_segmentation.config(text="Destination: Not selected", width=35)

def segment_audio_files():
    if not source_path_segmentation:
        messagebox.showerror("Error", "Please select a source folder.")
        return
    if not destination_path_segmentation:
        messagebox.showerror("Error", "Please select a destination folder.")
        return

    try:
       
        utterance_length = int(utterance_length_var.get())
        if utterance_length <= 0:
            messagebox.showerror("Error", "Please enter a valid utterance length.")
            return

        progress_bar_segmentation["value"] = 0 
        files = []
        for root_dir, _, filenames in os.walk(source_path_segmentation): 
            files.extend(os.path.join(root_dir, f) for f in filenames if f.lower().endswith((".wav", ".mp3", ".flac")))
        total_files = len(files)

        if total_files == 0:
            messagebox.showerror("Error", "No audio files to segment.")
            return

       
        for i, file_path in enumerate(files):
            audio = AudioSegment.from_file(file_path)
            total_duration = len(audio) / 1000 

           
            rel_path = os.path.relpath(file_path, source_path_segmentation)
            file_name = os.path.splitext(rel_path)[0]
            output_folder = os.path.join(destination_path_segmentation, file_name)
            os.makedirs(output_folder, exist_ok=True)
           
            for j in range(0, int(total_duration), utterance_length):
                segment = audio[j * 1000: (j + utterance_length) * 1000] 
                segment.export(os.path.join(output_folder, f"{file_name}_{j // utterance_length + 1}.wav"), format="wav")

           
            progress = (i + 1) / total_files * 100
            progress_bar_segmentation["value"] = progress
            root.update_idletasks() 

        messagebox.showinfo("Success", "Segmentation completed!")
    except Exception as e:
        print("Error", f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")


def select_source_directory_trimmer():
    """Select source directory for Trimmer."""
    global source_folder_trimmer
    source_folder_trimmer = filedialog.askdirectory()
    source_label_trimmer.config(text=f"Source Folder: .../{source_folder_trimmer.split('/')[-1]}")


def select_destination_directory_trimmer():
    """Select destination directory for Trimmer."""
    global destination_folder_trimmer
    destination_folder_trimmer = filedialog.askdirectory()
    destination_label_trimmer.config(text=f"Destination: .../{destination_folder_trimmer.split('/')[-1]}")


def trim_utterances():
    """Trim audio files to the specified length."""
    if not source_folder_trimmer or not destination_folder_trimmer:
        messagebox.showwarning("Warning", "Please select both source and destination folders!")
        return

    try:
        utterance_length = float(utterance_length_var_trim.get())
        if utterance_length <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Invalid utterance length! Please enter a positive number.")
        return

   
    audio_files = []
    for root_dir, _, files in os.walk(source_folder_trimmer):
        audio_files.extend([os.path.join(root_dir, f) for f in files if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))])

    progress_bar_trimmer["maximum"] = len(audio_files)
    progress_bar_trimmer["value"] = 0

    for idx, file_path in enumerate(audio_files):
        try:
            audio = AudioSegment.from_file(file_path)
            trimmed_audio = audio[:utterance_length * 1000] 

           
            relative_path = os.path.relpath(file_path, source_folder_trimmer)
            dest_path = os.path.join(destination_folder_trimmer, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

           
            trimmed_audio.export(dest_path, format=os.path.splitext(dest_path)[1][1:])
            progress_bar_trimmer["value"] += 1
            root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to trim {file_path}: {e}")

    messagebox.showinfo("Success", "Trimming completed successfully!")


def select_source_directory_noise_reduction():
    """Select source directory for Noise Reduction."""
    global source_folder_noise_reduction
    source_folder_noise_reduction = filedialog.askdirectory()
    source_label_noise_reduction.config(text=f"Source Folder: .../{source_folder_noise_reduction.split('/')[-1]}")


def select_destination_directory_noise_reduction():
    """Select destination directory for Noise Reduction."""
    global destination_folder_noise_reduction
    destination_folder_noise_reduction = filedialog.askdirectory()
    destination_label_noise_reduction.config(text=f"Destination: .../{destination_folder_noise_reduction.split('/')[-1]}")


def reduce_noise():
    """Reduce noise in audio files."""
    if not source_folder_noise_reduction or not destination_folder_noise_reduction:
        messagebox.showwarning("Warning", "Please select both source and destination folders!")
        return

   
    audio_files = []
    for root_dir, _, files in os.walk(source_folder_noise_reduction):
        audio_files.extend([os.path.join(root_dir, f) for f in files if f.lower().endswith(('.wav'))])

    if not audio_files:
        messagebox.showwarning("Warning", "No WAV files found in the source folder!")
        return

    progress_bar_noise_reduction["maximum"] = len(audio_files)
    progress_bar_noise_reduction["value"] = 0

    for idx, file_path in enumerate(audio_files):
        try:
            rate, data = wav.read(file_path)

           
            if len(data.shape) > 1:
                data = data.mean(axis=1).astype(np.int16)

            reduced_noise = nr.reduce_noise(y=data.astype(float), sr=rate)

           
            relative_path = os.path.relpath(file_path, source_folder_noise_reduction)
            dest_path = os.path.join(destination_folder_noise_reduction, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            wav.write(dest_path, rate, reduced_noise.astype(np.int16))

           
            progress_bar_noise_reduction["value"] += 1
            root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process {file_path}: {e}")

    messagebox.showinfo("Success", "Noise reduction completed successfully!")

def select_source_directory_audio_normalization():
    """Select source directory for Audio Normalization."""
    global source_folder_audio_normalization
    source_folder_audio_normalization = filedialog.askdirectory()
    source_label_audio_normalization.config(text=f"Source Folder: .../{source_folder_audio_normalization.split('/')[-1]}")


def select_destination_directory_audio_normalization():
    """Select destination directory for Audio Normalization."""
    global destination_folder_audio_normalization
    destination_folder_audio_normalization = filedialog.askdirectory()
    destination_label_audio_normalization.config(text=f"Destination: .../{destination_folder_audio_normalization.split('/')[-1]}")


def normalize_audio_files():
    """Normalize volume of audio files."""
    if not source_folder_audio_normalization or not destination_folder_audio_normalization:
        messagebox.showwarning("Warning", "Please select both source and destination folders!")
        return

   
    audio_files = []
    for root_dir, _, files in os.walk(source_folder_audio_normalization):
        audio_files.extend([os.path.join(root_dir, f) for f in files if f.lower().endswith(('.wav'))])

    if not audio_files:
        messagebox.showwarning("Warning", "No WAV files found in the source folder!")
        return

    progress_bar_audio_normalization["maximum"] = len(audio_files)
    progress_bar_audio_normalization["value"] = 0

    for idx, file_path in enumerate(audio_files):
        try:
           
            audio = AudioSegment.from_file(file_path)

           
            normalized_audio = audio.normalize()

           
            relative_path = os.path.relpath(file_path, source_folder_audio_normalization)
            dest_path = os.path.join(destination_folder_audio_normalization, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

           
            normalized_audio.export(dest_path, format="wav")

           
            progress_bar_audio_normalization["value"] += 1
            root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process {file_path}: {e}")

    messagebox.showinfo("Success", "Audio normalization completed successfully!")

class ModernTheme:
   
    PRIMARY_COLOR = "#2196F3" 
    SECONDARY_COLOR = "#f5f5f5" 
    TEXT_COLOR = "#212121" 
    ACCENT_COLOR = "#1976D2" 
    SUCCESS_COLOR = "#4CAF50" 
    
   
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    BUTTON_WIDTH = 20
    BUTTON_HEIGHT = 2
    PADDING = 10

def setup_styles():
    style = ttk.Style()
    style.configure("TNotebook", background=ModernTheme.SECONDARY_COLOR)
    style.configure("TNotebook.Tab", padding=[10, 5], font=('Arial', 10))
    style.configure("TFrame", background=ModernTheme.SECONDARY_COLOR)
    style.configure("Action.TButton",
                   padding=5,
                   font=('Arial', 10, 'bold'),
                   background=ModernTheme.PRIMARY_COLOR)
    style.configure("TProgressbar",
                   thickness=20,
                   background=ModernTheme.PRIMARY_COLOR)

def create_modern_button(parent, text, command):
    button = tk.Button(parent,
                      text=text,
                      command=command,
                      bg=ModernTheme.PRIMARY_COLOR,
                      fg="white",
                      font=('Arial', 10, 'bold'),
                      relief=tk.FLAT,
                      width=ModernTheme.BUTTON_WIDTH,
                      height=ModernTheme.BUTTON_HEIGHT,
                      cursor="hand2")
    button.bind('<Enter>', lambda e: e.widget.configure(bg=ModernTheme.ACCENT_COLOR))
    button.bind('<Leave>', lambda e: e.widget.configure(bg=ModernTheme.PRIMARY_COLOR))
    return button

def create_modern_label(parent, text, is_header=False):
    if is_header:
        return tk.Label(parent,
                       text=text,
                       font=('Arial', 16, 'bold'),
                       fg=ModernTheme.TEXT_COLOR,
                       bg=ModernTheme.SECONDARY_COLOR,
                       pady=10)
    return tk.Label(parent,
                   text=text,
                   font=('Arial', 10),
                   fg=ModernTheme.TEXT_COLOR,
                   bg=ModernTheme.SECONDARY_COLOR)

root = tk.Tk()
root.title("Audio Processing Toolkit")
root.geometry(f"{ModernTheme.WINDOW_WIDTH}x{ModernTheme.WINDOW_HEIGHT}")
root.configure(bg=ModernTheme.SECONDARY_COLOR)

setup_styles()

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=ModernTheme.PADDING, pady=ModernTheme.PADDING)

class BaseFrame(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.configure(style="TFrame")
        self.columnconfigure(1, weight=1)
        
       
        header = create_modern_label(self, title, is_header=True)
        header.grid(row=0, column=0, columnspan=2, pady=(20, 30), padx=20)


conversion_frame = BaseFrame(notebook, "Format Conversion")
notebook.add(conversion_frame, text="Conversion")

create_modern_button(conversion_frame, "Select Source Folder", select_directory).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label = create_modern_label(conversion_frame, "Source Folder: Not selected")
source_label.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(conversion_frame, "Select Destination Folder", select_destination).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label = create_modern_label(conversion_frame, "Destination: Not selected")
destination_label.grid(row=2, column=1, pady=10, padx=20, sticky="w")

format_frame = ttk.Frame(conversion_frame, style="TFrame")
format_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
create_modern_label(format_frame, "Output Format:").pack(side=tk.LEFT, padx=(0, 10))
output_format_var = tk.StringVar(value="wav")
ttk.Entry(format_frame, textvariable=output_format_var, width=10).pack(side=tk.LEFT)

progress_bar = ttk.Progressbar(conversion_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(conversion_frame, "Convert", convert_to_format).grid(
    row=5, column=0, columnspan=2, pady=20)

silence_frame = BaseFrame(notebook, "Silence Removal")
notebook.add(silence_frame, text="Silence Removal")

create_modern_button(silence_frame, "Select Source Folder", select_directory_silence).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label_silence = create_modern_label(silence_frame, "Source Folder: Not selected")
source_label_silence.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(silence_frame, "Select Destination Folder", select_destination_silence).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label_silence = create_modern_label(silence_frame, "Destination: Not selected")
destination_label_silence.grid(row=2, column=1, pady=10, padx=20, sticky="w")

threshold_frame = ttk.Frame(silence_frame, style="TFrame")
threshold_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
create_modern_label(threshold_frame, "Silence Threshold:").pack(side=tk.LEFT, padx=(0, 10))
silence_thresh_var = tk.StringVar(value='-50')
ttk.Entry(threshold_frame, textvariable=silence_thresh_var, width=10).pack(side=tk.LEFT)

progress_bar_silence = ttk.Progressbar(silence_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar_silence.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(silence_frame, "Remove Silence", remove_silence_from_files).grid(
    row=5, column=0, columnspan=2, pady=20)

segmentation_frame = BaseFrame(notebook, "Audio Segmentation")
notebook.add(segmentation_frame, text="Segmentation")

create_modern_button(segmentation_frame, "Select Source Folder", select_directory_segmentation).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label_segmentation = create_modern_label(segmentation_frame, "Source Folder: Not selected")
source_label_segmentation.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(segmentation_frame, "Select Destination Folder", select_destination_segmentation).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label_segmentation = create_modern_label(segmentation_frame, "Destination: Not selected")
destination_label_segmentation.grid(row=2, column=1, pady=10, padx=20, sticky="w")

length_frame = ttk.Frame(segmentation_frame, style="TFrame")
length_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
create_modern_label(length_frame, "Utterance Length (s):").pack(side=tk.LEFT, padx=(0, 10))
utterance_length_var = tk.StringVar(value='3')
ttk.Entry(length_frame, textvariable=utterance_length_var, width=10).pack(side=tk.LEFT)

progress_bar_segmentation = ttk.Progressbar(segmentation_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar_segmentation.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(segmentation_frame, "Segment Audio", segment_audio_files).grid(
    row=5, column=0, columnspan=2, pady=20)

trimmer_frame = BaseFrame(notebook, "Utterance Trimmer")
notebook.add(trimmer_frame, text="Trimmer")

create_modern_button(trimmer_frame, "Select Source Folder", select_source_directory_trimmer).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label_trimmer = create_modern_label(trimmer_frame, "Source Folder: Not selected")
source_label_trimmer.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(trimmer_frame, "Select Destination Folder", select_destination_directory_trimmer).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label_trimmer = create_modern_label(trimmer_frame, "Destination: Not selected")
destination_label_trimmer.grid(row=2, column=1, pady=10, padx=20, sticky="w")

trim_length_frame = ttk.Frame(trimmer_frame, style="TFrame")
trim_length_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
create_modern_label(trim_length_frame, "Utterance Length (s):").pack(side=tk.LEFT, padx=(0, 10))
utterance_length_var_trim = tk.StringVar(value="3")
ttk.Entry(trim_length_frame, textvariable=utterance_length_var_trim, width=10).pack(side=tk.LEFT)

progress_bar_trimmer = ttk.Progressbar(trimmer_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar_trimmer.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(trimmer_frame, "Trim", trim_utterances).grid(
    row=5, column=0, columnspan=2, pady=20)

noise_reduction_frame = BaseFrame(notebook, "Noise Reduction")
notebook.add(noise_reduction_frame, text="Noise Reduction")

create_modern_button(noise_reduction_frame, "Select Source Folder", select_source_directory_noise_reduction).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label_noise_reduction = create_modern_label(noise_reduction_frame, "Source Folder: Not selected")
source_label_noise_reduction.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(noise_reduction_frame, "Select Destination Folder", select_destination_directory_noise_reduction).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label_noise_reduction = create_modern_label(noise_reduction_frame, "Destination: Not selected")
destination_label_noise_reduction.grid(row=2, column=1, pady=10, padx=20, sticky="w")

progress_bar_noise_reduction = ttk.Progressbar(noise_reduction_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar_noise_reduction.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(noise_reduction_frame, "Reduce Noise", reduce_noise).grid(
    row=5, column=0, columnspan=2, pady=20)

audio_normalization_frame = BaseFrame(notebook, "Audio Normalization")
notebook.add(audio_normalization_frame, text="Audio Normalization")

create_modern_button(audio_normalization_frame, "Select Source Folder", select_source_directory_audio_normalization).grid(
    row=1, column=0, pady=10, padx=20, sticky="w")
source_label_audio_normalization = create_modern_label(audio_normalization_frame, "Source Folder: Not selected")
source_label_audio_normalization.grid(row=1, column=1, pady=10, padx=20, sticky="w")

create_modern_button(audio_normalization_frame, "Select Destination Folder", select_destination_directory_audio_normalization).grid(
    row=2, column=0, pady=10, padx=20, sticky="w")
destination_label_audio_normalization = create_modern_label(audio_normalization_frame, "Destination: Not selected")
destination_label_audio_normalization.grid(row=2, column=1, pady=10, padx=20, sticky="w")

progress_bar_audio_normalization = ttk.Progressbar(audio_normalization_frame, length=600, mode="determinate", style="TProgressbar")
progress_bar_audio_normalization.grid(row=4, column=0, columnspan=2, pady=(20, 10), padx=20, sticky="ew")

create_modern_button(audio_normalization_frame, "Normalize Audio", normalize_audio_files).grid(
    row=5, column=0, columnspan=2, pady=20)

root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - ModernTheme.WINDOW_WIDTH) // 2
y = (screen_height - ModernTheme.WINDOW_HEIGHT) // 2
root.geometry(f"{ModernTheme.WINDOW_WIDTH}x{ModernTheme.WINDOW_HEIGHT}+{x}+{y}")

root.mainloop()
