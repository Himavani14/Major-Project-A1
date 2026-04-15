from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import cv2

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import resample
from keras.preprocessing.image import img_to_array
from keras.models import Model
from sklearn.preprocessing import label_binarize

import logging
import tensorflow as tf
import soundfile as sf
import resampy

import tkinter as tk
from tkinter import Text, ttk
from PIL import Image, ImageTk

import sounddevice as sd
from tkinter import END

from sklearn.ensemble import VotingClassifier
from tinydb import TinyDB, Query
import hashlib
from tkinter import messagebox


accuracy = []
precision = []
recall = []
fscore = []

model_folder = "model"
os.makedirs(model_folder, exist_ok=True)


main = tkinter.Tk()
main.configure(bg='#f0f8ff') 
main.title("AUTOMATED ANIMAL SPECIES") 
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")


def Upload_Dataset():
    global file_name, categories, X, Y 
    text.delete('1.0', END)
    
    file_name = filedialog.askdirectory(initialdir=".")
    
    if not file_name:
        text.insert(END, "No directory selected.\n")
        return
    categories = [d for d in os.listdir(file_name) if os.path.isdir(os.path.join(file_name, d))]
    
    if not categories:
        text.insert(END, "No class folders found in the selected directory.\n")
        return

    text.insert(END, 'Dataset loaded successfully.\n')
    text.insert(END, "Classes found in dataset: " + str(categories) + "\n")
    


def audio_feature_pipeline():
    global categories, file_name, X, Y

    def load_and_preprocess_audio(file_path, target_sr=16000):
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        return audio.astype(np.float32), target_sr

    def extract_audio_features(audio, sr):
        import numpy as np
        import librosa

        # 1. MFCCs + Delta + Delta-Delta
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
   
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 3. Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_mean = np.mean(mel, axis=1)

        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # 5. Tonnetz
        y_harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        # 6. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr, axis=1)

        # 7. RMS Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms, axis=1)

        # 8. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = np.mean(centroid, axis=1)

        # 9. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidth_mean = np.mean(bandwidth, axis=1)

        # 10. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio)
        flatness_mean = np.mean(flatness, axis=1)

        # Combine all features
        feature_vector = np.hstack([
            mfccs,
            chroma_mean,
            mel_mean,
            contrast_mean,
            tonnetz_mean,
            zcr_mean,
            rms_mean,
            centroid_mean,
            bandwidth_mean,
            flatness_mean
        ])

        return feature_vector

    if os.path.exists("X.npy") and os.path.exists("Y.npy"):
        print("Loading existing features and labels...")
        X = np.load("X.npy")
        Y = np.load("Y.npy")
        text.insert(END, f"Feature extraction loaded\n")
    else:
        print("Extracting handcrafted audio features...")
        features = []
        labels = []
        class_to_idx = {name: idx for idx, name in enumerate(categories)}

        for class_name in categories:
            class_dir = os.path.join(file_name, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(class_dir, file)
                    print(file_path)
                    audio, sr = load_and_preprocess_audio(file_path)
                    if audio.size == 0:
                        continue
                    combined_features = extract_audio_features(audio, sr)
                    features.append(combined_features)
                    labels.append(class_to_idx[class_name])

        X = np.array(features)
        Y = np.array(labels)
        np.save("X.npy", X)
        np.save("Y.npy", Y)
        text.insert(END, f"Feature extraction completed and saved\n")

    return X, Y
    

def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, "Feature extraction completed and saved\n")

    

def Calculate_Metrics(algorithm, predict, y_test, predict_proba=None):
    global categories, accuracy, precision, recall, fscore, text

    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, algorithm + " Accuracy  :  " + str(a) + "\n")
    text.insert(END, algorithm + " Precision : " + str(p) + "\n")
    text.insert(END, algorithm + " Recall    : " + str(r) + "\n")
    text.insert(END, algorithm + " FScore    : " + str(f) + "\n")

    conf_matrix = confusion_matrix(y_test, predict)
    CR = classification_report(y_test, predict, target_names=categories)
    text.insert(END, algorithm + ' Classification Report \n')
    text.insert(END, algorithm + str(CR) + "\n\n")

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="magma", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(algorithm + " Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

    if predict_proba is not None:
        y_test_bin = label_binarize(y_test, classes=np.arange(len(categories)))
        if np.unique(y_test).shape[0] < 2:
            text.insert(END, algorithm + " ROC AUC Score: Not defined (only one class in y_test)\n")
            return

        try:
            roc_auc = roc_auc_score(y_test_bin, predict_proba, average='macro', multi_class='ovr') * 100
            text.insert(END, algorithm + " ROC AUC Score: " + str(roc_auc) + "\n")

            plt.figure(figsize=(8, 6))
            for i in range(len(categories)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], predict_proba[:, i])
                roc_auc_i = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{categories[i]} (AUC = {roc_auc_i:.2f})')

            plt.plot([0, 1], [0, 1], 'k--')  
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(algorithm + ' ROC Curves')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()

        except ValueError as e:
            text.insert(END, algorithm + f" ROC AUC")


def existing_classifier_SVC():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Existing_SVM_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading Existing SVM model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training SVM model...\n")
        mlmodel = SVC(kernel='sigmoid', probability=True, C=1e-6, gamma='auto')
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        text.insert(END, "Model saved at: " + model_filename + "\n")

    y_pred = mlmodel.predict(x_test)
    y_proba = mlmodel.predict_proba(x_test)

    Calculate_Metrics("Existing SVM", y_pred, y_test, y_proba)


def existing_classifier_DTC():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Existing_DTC_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading Existing Decision Tree model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training Decision Tree model...\n")
        mlmodel = DecisionTreeClassifier(
            criterion="entropy", 
            max_leaf_nodes=2, 
            max_features=None  
        )
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        text.insert(END, "Model saved at: " + model_filename + "\n")

    y_pred = mlmodel.predict(x_test)
    y_proba = mlmodel.predict_proba(x_test)

    Calculate_Metrics("Existing DTC", y_pred, y_test, y_proba)


def classifier_HybridModel():
    global x_train, y_train, x_test, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Hybrid_SVM_LGBM_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading Hybrid SVM + LGBM model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training Hybrid SVM + LGBM model...\n")

        svm_model = SVC(probability=True)
        lgbm_model = LGBMClassifier(
            objective='multiclass', 
            num_class=len(set(y_train)),
            boosting_type='gbdt',
            learning_rate=0.1,
            n_estimators=100
        )

        mlmodel = VotingClassifier(
            estimators=[('svm', svm_model), ('lgbm', lgbm_model)],
            voting='soft'
        )

        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        text.insert(END, "Hybrid model saved at: " + model_filename + "\n")

    y_pred = mlmodel.predict(x_test)
    y_proba = mlmodel.predict_proba(x_test)

    Calculate_Metrics("Hybrid SVM + LGBM", y_pred, y_test, y_proba)


def Prediction():
    global mlmodel, categories, model_folder,sf,np,sd
    filename = filedialog.askopenfilename(initialdir="Test Data", title="Select Audio File", filetypes=[("WAV files", "*.wav")])
    text.delete('1.0', END)

    if not filename or not filename.lower().endswith(".wav"):
        text.insert(END, "Invalid file format. Please select a .wav file.\n")
        return

    text.insert(END, f"{filename} Loaded\n")

    def load_and_preprocess_audio(file_path, target_sr=16000):
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) 
        if sr != target_sr:
            audio = resampy.resample(audio, sr, target_sr)
        return audio.astype(np.float32), target_sr

    def extract_audio_features(audio, sr):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rmse = librosa.feature.rms(y=audio)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        feature_vector = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1),
            np.mean(zcr),
            np.mean(rmse),
            np.mean(centroid),
            np.mean(bandwidth),
            np.mean(rolloff),
            tempo
        ])
        return feature_vector

    audio, sr = load_and_preprocess_audio(filename)
    if audio.size == 0:
        text.insert(END, "Audio loading failed.\n")
        return

    text.insert(END, "Playing audio...\n")
    sd.play(audio, sr)
    sd.wait()

    test_features = extract_audio_features(audio, sr).reshape(1, -1)

    model_path = os.path.join(model_folder, "Hybrid_SVM_LGBM_model.pkl")
    if not os.path.exists(model_path):
        text.insert(END, "Trained model not found.\n")
        return
    mlmodel = joblib.load(model_path)

    prediction = mlmodel.predict(test_features)[0]
    predicted_class = categories[prediction] if prediction < len(categories) else str(prediction)
    text.insert(END, f"Predicted Outcome From Test Audio: {predicted_class}\n")

    y, sr = librosa.load(filename, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    waveplot_path = os.path.join(model_folder, "waveplot.png")
    plt.savefig(waveplot_path)
    plt.close()

    waveplot_img = cv2.imread(waveplot_path)
    if waveplot_img is not None:
        cv2.putText(waveplot_img, f"Prediction: {predicted_class}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Waveplot with Prediction", waveplot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        text.insert(END, "Failed to load waveplot image.\n")


def setBackground():
    global bg_photo
    image_path = r"BG_image\background.jpg" 
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
    bg_label.lower()

db = TinyDB("users_db.json")
users_table = db.table("users")


def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            if users_table.search((User.username == username) & (User.role == role)):
                messagebox.showerror("Error", f"{role} with this username already exists!")
                return

            users_table.insert({
                "username": username,
                "password": hashed_password,
                "role": role
            })

            messagebox.showinfo("Success", f"{role} Signup Successful!")
            signup_window.destroy()
            show_login_screen()
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            User = Query()
            result = users_table.search(
                (User.username == username) &
                (User.password == hashed_password) &
                (User.role == role)
            )

            if result:
                messagebox.showinfo("Success", f"{role} Login Successful!")
                login_window.destroy()
                clear_buttons()
                if role == "Admin":
                    show_main_buttons()
                elif role == "User":
                    show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid Credentials!")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


def clear_buttons():
    for widget in main.winfo_children():
        if widget not in [title, text]:  
            widget.destroy()
        
    setBackground() 

    title.lift()
    text.lift()

import tkinter as tk

def show_main_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()  

    tk.Button(main, text="Upload Dataset",
              command=Upload_Dataset,
              font=font1).place(x=20, y=100)

    tk.Button(main, text="Audio Feature Extraction",
              command=audio_feature_pipeline,
              font=font1).place(x=20, y=150)

    tk.Button(main, text="Train Test Splitting",
              command=Train_Test_Splitting,
              font=font1).place(x=20, y=200)

    tk.Button(main, text="Support Vector Machine",
              command=existing_classifier_SVC,
              font=font1).place(x=20, y=250)

    tk.Button(main, text="Decision Tree Classifier",
              command=existing_classifier_DTC,
              font=font1).place(x=20, y=300)

    tk.Button(main, text="Proposed SVC and LGBM",
              command=classifier_HybridModel,
              font=font1).place(x=20, y=350)


    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=20, y=400)


def show_user_buttons():
    font1 = ('times', 13, 'bold')
    clear_buttons()
    tk.Button(main, text="Prediction",
              command=Prediction,
              font=font1).place(x=20, y=200)

    tk.Button(main, text="Exit", command=close, font=font1).place(x=20, y=250)

    tk.Button(main, text="Logout", command=show_login_screen, font=font1, bg="red").place(x=20, y=300)

def show_login_screen():
    clear_buttons()
    font1 = ('times', 14, 'bold')

    tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='red').place(x=100, y=100)
    tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='red').place(x=400, y=100)
    tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=700, y=100)
    tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=1000, y=100)

def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(
    main,
    text="AUTOMATED ANIMAL SPECIES IDENTIFICATION VIA DEEP LEARNING ON BIOACOUSTIC RECORDINGS",
    bg='#003366', 
    fg='white',
    font=font,
    height=3,
    width=120
)
title.pack(pady=10)

       
font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=220)
text.config(font=font1) 


font1 = ('times', 14, 'bold')


show_login_screen()
main.mainloop()