RAVDESS_PATH = 'Ravdess_Dataset/Audio'
TESS_PATH = 'Tess_Dataset/Audio'
SAVEE_PATH = 'Savee_Dataset/AudioData'
X_TRAIN_DATA = 'X.joblib'
Y_TRAIN_DATA = 'Y.joblib'

X_TEST_DATA = 'X_test.joblib'
Y_TEST_DATA = 'Y_test.joblib'

SAMPLE_RATE = 48000


SUPPORTED_AUDIO_FILES = ['.wav', '.mp3', '.tmp']
SUPPORTED_VIDEO_FILES = ['.avi', '.mp4']

'''
Ravdess Emotions
01 = neutral, 02 = neutral, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

Proposed Emotions
0 - neutral, 1 - happy, 2 - sad, 3 - angry, 4 - fearful, 5 - disgust, 6 - surprised
'''


tess_to_emotions = {
    'angry': 3,
    'disgust': 5,
    'fear': 4,
    'happy': 1,
    'neutral': 0,
    'ps': 6,
    'sad': 2,
}

ravdess_to_emotions = {
    '01': 0,
    '02': 0,
    '03': 1,
    '04': 2,
    '05': 3,
    '06': 4,
    '07': 5,
    '08': 6
}

savee_to_emotions = {
    'a': 3,
    'd': 5,
    'f': 4,
    'h': 1,
    'n': 0,
    'sa': 2,
    'su': 6
}

emotions = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fearful',
    5: 'disgust',
    6: 'surprised'
}

