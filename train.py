import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA

try:
    from featureExtractor import*
except ImportError:
    print "Make sure FeatureGen.pyc file is in the current directory"
    exit()

emotions = ["anger", "contempt", "disgust", "fear", "happy","sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3,C=0.025 ,gamma=0.00000001)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("sorted_set/%s/*" %emotion)
    return files

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            print(item  )
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            img = clahe.apply(gray)
            # landmarks_vectorised = get_landmarks(clahe_image)
            dets=detector(img,1)

            if len(dets)==0:
                print "Unable to find any face."
                return

            for k,d in enumerate(dets):

                shape=predictor(img,d)
                landmarks=[]
                for i in range(68):
                    landmarks.append(shape.part(i).x)
                    landmarks.append(shape.part(i).y)


                landmarks=numpy.array(landmarks)

                print "Generating features......"
                features=generateFeatures(landmarks)
                features= numpy.asarray(features)

                training_data.append(features) #append image array to training data list
                training_labels.append(emotions.index(emotion))
    return training_data, training_labels

training_data, training_labels = make_sets()
print(training_labels)
npar_train = np.array(training_data)
npar_trainlabs = np.array(training_labels)
# pca=joblib.load("pcadata.pkl")

# pca.fit(npar_train)
# ash = pca.transform(npar_train)
# joblib.dump(pca,'pca.pkl')
clf.fit(npar_train, npar_trainlabs)
joblib.dump(clf, 'model.pkl')
