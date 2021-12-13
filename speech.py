#Import All Important Libraries
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Define the motions dictionary
emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}

#Emotions we want to observe
observed_emotions = ['calm', 'sad', 'angry', 'fearful']

#function for extracting mfcc, chroma, and mel features from sound file
def extract_feature(file_name, mfcc, chroma, mel):
  with soundfile.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate=sound_file.samplerate
    if chroma:
      stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
      mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result=np.hstack((result, mfccs))
    if chroma:
      chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      result=np.hstack((result, chroma))
    if mel:
      mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
      result=np.hstack((result, mel))
  return result


#Load the data and extract features for each sound file
def load_data(filepath='speech-emotion-recognition-ravdess-data\Actor_*', test_size = 0.2):
  x, y = [], []
  for folder in glob.glob(filepath):
    print(folder)
    for file in glob.glob(folder + '/*.wav'):
      file_name = os.path.basename(file)
      emotion = emotions[file_name.split('-')[2]]
      if emotion not in observed_emotions:
        continue
      feature = extract_feature(file, mfcc = True, chroma = True, mel = True)
      x.append(feature)
      y.append(emotion)
  return train_test_split(np.array(x), y, test_size = test_size, random_state = 9)

x_train,x_test,y_train,y_test=load_data(test_size=0.2)

#Shape of train and test set and Number of features extracted
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

# #Initialised Multi Layer Perceptron Classifier
# model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = 'adaptive', max_iter = 500)

# used gridsearch to find various accuracies
model = MLPClassifier(alpha=0.001,epsilon = 1e-08, hidden_layer_sizes=(250, 150), activation='logistic', batch_size=256, learning_rate='invscaling', max_iter=1000)

classifier = model.fit(x_train, y_train)

print(x_train)

#Predict for the test set
y_pred = model.predict(x_test)

print(y_pred)


#Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)

#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix

#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_test)

print(cm)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Printing the accuracy
print("Accuracy of MLPClassifier : ", accuracy(cm))

# t = extract_feature('C:\Savani\Third Year\SER-final\speech-emotion-recognition-ravdess-data\Actor_24\24.wav', mfcc = True, chroma = True, mel = True)

# t = extract_feature(os.path.join("speech-emotion-recognition-ravdess-data", "Actor_24", "03-01-06-02-02-02-24.wav"), mfcc=True, chroma=True, mel=True)


def recognize():
    path = 'form-input/'
    files = glob.glob(os.path.join(path, "*"))
    for file in files:
      t = extract_feature(file, mfcc=True, chroma=True, mel=True)
      ans=model.predict([t])
      print(ans)
      if ans == 'calm':
            ans = 'You are a CALM person 😌.You inspire me with your calmness!'
      elif ans == 'sad':
            ans = 'I see you are SAD 😢.Cheer up champ, the world is beutiful and I am always with you!'
      elif ans == 'angry':
            ans = 'Why are you ANGRY 😤? Count 1 to 10 and please give me a huge smile!'
      elif ans == 'fearful':
            ans = 'You are SCARED like a chicken 😱.Take a deep breath and imagine your happy place!'
      else:
            ans = 'You are not Calm or Angry or Sad or even Fearful. Hope you are HAPPY 😄!'
      return ans

# recognize()