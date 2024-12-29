import tkinter as tk
from tkinter import ttk
from enrollment_tab import EnrollmentTab
from train_on_fly_tab import TrainOnFlyTab
from real_time_identification import RealTimeIdentificationTab


class SpeakerIdentificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Identification")
        self.root.geometry("800x640")
        
        
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        
        self.enrollment_tab = EnrollmentTab(self.notebook)
        self.train_on_fly_tab = TrainOnFlyTab(self.notebook)
        self.real_time_tab = RealTimeIdentificationTab(self.notebook)
        
        self.notebook.add(self.enrollment_tab, text="Enrollment")
        self.notebook.add(self.train_on_fly_tab, text="Train On The Fly")
        self.notebook.add(self.real_time_tab, text="Real-Time Identification")

def main():
    root = tk.Tk()
    app = SpeakerIdentificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
