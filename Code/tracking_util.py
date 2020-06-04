import cv2
import numpy as np
import PySimpleGUI as sg
from sklearn.neighbors import KNeighborsClassifier
import operator



class DetectionAndGroupStore:
    def __init__(self, img):
        self.groups = []
        self.detections = []
        self.img_copy = img.copy()
        self.img = img
        self.drawing = False

    def draw_rectangle(self,event,x,y,flags,param):
        global ix,iy #,drawing,img,img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.img = self.img_copy.copy()
                cv2.rectangle(self.img,(ix,iy),(x,y),(0,255,0),1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.img,(ix,iy),(x,y),(0,255,0),1)
            self.img_copy = self.img
            group_number = enter_group_number()
            self.detections.append((ix,x,iy,y))
            self.groups.append(group_number)

def enter_group_number():
    layout = [
              [sg.Text('Please enter group number of player')],
              [sg.Text('Group number', size=(15, 1)), sg.InputText()],
              [sg.Submit()]
             ]
    window = sg.Window('Group number', layout)
    event, values = window.Read()
    window.Close()
    return values[0]

def train_kNN(hists, group_numbers):
    knn = KNeighborsClassifier(n_neighbors=3)
    hists_merged = []
    for hist in hists:
        hists_merged.append(np.hstack((hist[0], hist[1])))

    hists_merged = np.asarray(hists_merged)
    r, c, d = hists_merged.shape
    hists_merged = hists_merged.reshape(r, c*d)

    knn.fit(hists_merged, group_numbers)

    return knn

def calc_hists(initial_detections, img_hsv):
    hists = []
    for detection in initial_detections:
        hist = []
        x1,x2,y1,y2 = detection

        first_half = y1 + int(np.ceil((y2-y1)/2)) + 1
        hist_first_half_HS = cv2.calcHist([img_hsv[y1:first_half, x1:(x2+1)]], [0,1], None, [10,5], [0, 180, 0, 256])
        hist_first_half_V = cv2.calcHist([img_hsv[y1:first_half, x1:(x2+1)]], [2], None, [5], [0, 256]).reshape((1,5))
        hist_second_half_HS = cv2.calcHist([img_hsv[first_half:(y2+1), x1:(x2+1)]], [0,1], None, [10,5], [0, 180, 0, 256])
        hist_second_half_V = cv2.calcHist([img_hsv[first_half:(y2+1), x1:(x2+1)]], [2], None, [5], [0, 256]).reshape((1,5))
        prob_dist_div_first_half = 1/(np.sum(hist_first_half_HS)+np.sum(hist_first_half_V))
        if prob_dist_div_first_half > 10000:
            sys.exit(0)
        prob_dist_div_second_half = 1/(np.sum(hist_second_half_HS)+np.sum(hist_second_half_V))
        if prob_dist_div_second_half > 10000:
            sys.exit(0)
        hist_first_half = prob_dist_div_first_half*np.vstack((hist_first_half_HS, hist_first_half_V))
        hist_second_half = prob_dist_div_second_half*np.vstack((hist_second_half_HS, hist_second_half_V))
        hist.append(hist_first_half)
        hist.append(hist_second_half)
        hists.append(hist)

    return hists

def make_tracks(initial_detections, hists, player_groups):
    tracks = []
    for i in range(len(initial_detections)):
        x1,x2,y1,y2 = initial_detections[i]
        hist = hists[i]
        vector = np.array([[y1 + (y2-y1)/2], [x1 + (x2-x1)/2], [0], [0]])
        tracks.append((vector, hist))

    return tracks

def initiate_states(initial_detections):
    states = []

    for i in range(len(initial_detections)):
        x1,x2,y1,y2 = initial_detections[i]
        state = 1. * np.array([[y1 + (y2-y1)/2.], [x1 + (x2-x1)/2.], [0], [0]])
        states.append(state)

    return states

def calcApp(img, reference, detection_hist):
    J = 2
    lambda_coef = 20
    hist_ref_first, hist_ref_second = reference
    hist_first, hist_second = detection_hist
    dist_all = cv2.compareHist(hist_ref_first, hist_first, cv2.HISTCMP_BHATTACHARYYA)**2 + cv2.compareHist(hist_ref_second, hist_second, cv2.HISTCMP_BHATTACHARYYA)**2
    match_prob = np.exp(-lambda_coef*(1/J)*dist_all)


    return match_prob

def calcMotion(deviation, vector, prediction):
    deviation = np.sqrt(6)
    factor = (1/(deviation*np.sqrt(np.pi)))
    distance = np.sqrt(((vector[0] - prediction[0])**2 + (vector[1] - prediction[1])**2).astype(float))
    mot_prob = factor*np.exp(-(distance**2/deviation**2))

    return mot_prob

def calcAvgValues(detections):
    avg_h = 0
    avg_w = 0
    for detection in detections:
        x1,x2,y1,y2 = detection
        avg_w = avg_w + (x2-x1)
        avg_h = avg_h + (y2-y1)
    avg_w = int(np.ceil(avg_w)/len(detections))
    avg_h = int(np.ceil(avg_h)/len(detections))

    return avg_h, avg_w

def remove_duplicate_tracks(states, predicted_states, updated_tracks, ids, player_groups):
    tracks_to_clean = []
    ind_to_delete = []
    for i in range(len(updated_tracks)):
        vector_first, _ = updated_tracks[i]
        for j in range((i+1), len(updated_tracks)):
            vector_second, _ = updated_tracks[j]
            distance = np.sqrt((vector_first[0][0] - vector_second[0][0])**2 + (vector_first[1][0] - vector_second[1][0])**2)
            if distance == 0 and player_groups[i]==player_groups[j]:
                print("Usao: " + str(ids[i]) + " " + str(ids[j]))
                tracks_to_clean.append((i,j))

    for clean_tracks in tracks_to_clean:
        first, second = clean_tracks
        first_statePre = predicted_states[first]
        statePost = states[first]
        second_statePre = predicted_states[second]
        motion_first = np.sqrt((first_statePre[0][0]-statePost[0][0])**2 + (first_statePre[1][0] - statePost[1][0])**2)
        motion_second = np.sqrt((second_statePre[0][0]-statePost[0][0])**2 + (second_statePre[1][0] - statePost[1][0])**2)
        if motion_first < motion_second:
            ind_to_delete.append(second)
        elif motion_first > motion_second:
            ind_to_delete.append(first)

    for index in sorted(ind_to_delete, reverse = True):
        del states[index]
        del updated_tracks[index]
        del ids[index]
        del player_groups[index]


    return updated_tracks, ids, player_groups

def remove_occluded_over_threshold(occluded_thershold, occluded, states, updated_tracks, ids, player_groups):
    list_to_remove = [j for j in range(len(occluded)) if occluded[j] >= occluded_thershold]

    for index in sorted(list_to_remove, reverse = True):
        del occluded[index]
        del states[index]
        del updated_tracks[index]
        del ids[index]
        del player_groups[index]

    return updated_tracks, ids, player_groups, occluded
