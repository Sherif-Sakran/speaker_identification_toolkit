# Robust and Real-Time Speaker Identification Tool
This tool was developed as part of the study titled "Developing a Robust and Real-Time Speaker Identification Tool: Comparative Analysis and Practical Implementation"
For users, you can download the tools from here https://gla-my.sharepoint.com/:f:/g/personal/3011251s_student_gla_ac_uk/EsjatuJDFOBEvqq595QicXgBjn5VTPg72_gJZwEmVkwN8A?e=ObSt5z
For researchers and developers, you can start with tool as follows:
1. Clone the repo
2. Create a virtual environment
3. Install the packages in requirements.txt
4. For the audio processing toolkit, run audio_processing.py
5. For the speaker identification tool, run speaker_identification.py

# 1. Audio Processing Toolkit
This toolkit has several functionalities that can be useful in audio processing applications. It has the following tabs:
1. Format conversion: convert audio recordings (mainly mp3 and ogg) in a directory (recursively) to the conversion format (set to wav by default).
2. Silence removal: remove silence from audio recordings in a directory (recursively).
3. Segmentation: it reads recordings from a directory (non-recursively), then it segments each recording to n-second utterances then add them to a folder in the destination directory.
4. Trimmer: it trims recordings in a directory (recursively) to shorter ones.
5. Noise Reduction: processes the background noise in recordings in a directory (recursively).
6. Audio Normalization: normalize the volume of recordings in a directory (recursively).

# 2. Speaker Identification
This tool provides thre major functionalities for the speaker identification task as follows:
1. Enrollment: it enrolls speakers from a training set directory. It balances the number of utterancds per speaker by using the number of utterances of the speaker with the minimum number of utterances. The hyperparameters values are set to default values that are found to be practical according to the experiments of the study.
2. Train on the fly: it enrolls a speaker in a very short amount of time (<= 1 sec) and is found to yield very high accuracy with only as short as 15 seconds. It helps to mitigate the envrionmental variability challenge.
3. Real-Time Identification: it continously identify the speaker by making a prediction every 100 ms by taking the last n seconds in the same way sliding window algorithms work.
