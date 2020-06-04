import cv2
import numpy as np
import sys
import copy
import pickle
import time
import operator
import tracking_util

if __name__ == "__main__":
    r_max = 15
    max_distance = 25
    deviation = np.sqrt(6)
    factor = (1/(deviation*np.sqrt(np.pi)))
    dt = 1/25
    occluded_thershold = 100
    transitionMatrix = 1. * np.array([[1., 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    processNoiseCov = 0.25 * np.array([[dt**4/4, 0, dt**3/3, 0], [0, dt**4/4, 0, dt**3/3], [dt**3/3, dt**3/3, dt**2, 0], [dt**3/3, dt**3/3, 0, dt**2]])
    mean = np.array([0,0,0,0])

    ids = []
    last_detection_groups = []
    occluded = []
    time_start = time.time()
    for i in range(1000):
        img = cv2.imread("/home/ivan/linux mint/Pictures/frame" + str(i) + ".png")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if i==0:
        #    img_copy = img.copy()
        #    dtsgs = tracking_util.DetectionAndGroupStore(img)
        #    cv2.namedWindow('image')
        #    cv2.setMouseCallback('image',dtsgs.draw_rectangle)
            #drawing = False

        #    while(1):
        #        cv2.imshow('image',dtsgs.img)
        #        k = cv2.waitKey(1) & 0xFF
        #        if k == 27:
        #            break

        #    cv2.destroyAllWindows()
        #    initial_detections = dtsgs.detections
        #    player_groups = dtsgs.groups
        #    with open("/home/ivan/Desktop/trening.txt","w") as fil:
        #        for j in range(len(initial_detections)):
        #            x1,x2,y1,y2 = initial_detections[j]
        #            fil.write(str(player_groups[j]) + " " + str(y1) + " " + str(y2) + " " + str(x1) + " " + str(x2) + "\n")
        #    sys.exit(0)
            with open("/home/ivan/Desktop/trening.txt","r") as fil:
                train = fil.readlines()
            with open("/home/ivan/Desktop/prva_detekcija.txt","r") as fil:
                data = fil.readlines()

            train = [x.strip().split() for x in train]
            data = [x.strip().split() for x in data]

            player_groups_train = [int(x[0]) for x in train]
            player_groups = [int(x[0]) for x in data]

            initial_detections = [(int(x[3]), int(x[4]), int(x[1]), int(x[2])) for x in data]
            train_detections = [(int(x[3]), int(x[4]), int(x[1]), int(x[2])) for x in train]

            player_groups = [player_groups[i] for i in range(len(player_groups))
            if ((initial_detections[i][1]-initial_detections[i][0])*(initial_detections[i][3]-initial_detections[i][2])) >= 100]

            player_groups_train = [player_groups_train[i] for i in range(len(player_groups_train))
            if ((train_detections[i][1]-train_detections[i][0])*(train_detections[i][3]-train_detections[i][2])) >= 100]

            initial_detections = [detection for detection in initial_detections if ((detection[1]-detection[0])*(detection[3]-detection[2])) >= 100]
            train_detections = [detection for detection in train_detections if ((detection[1]-detection[0])*(detection[3]-detection[2])) >= 100]

            hists = tracking_util.calc_hists(initial_detections, img_hsv)
            hists_train = tracking_util.calc_hists(train_detections, img_hsv)

            kNN_classifier = tracking_util.train_kNN(hists_train, player_groups_train)

            hists = [hists[i] for i in range(len(hists)) if player_groups[i] < 5]
            initial_detections = [initial_detections[i] for i in range(len(initial_detections)) if player_groups[i] < 5]
            player_groups = [player_groups[i] for i in range(len(player_groups)) if player_groups[i] < 5]
            last_detection_groups = copy.deepcopy(player_groups)

            avg_h, avg_w = tracking_util.calcAvgValues(initial_detections)
            tracks = tracking_util.make_tracks(initial_detections, hists, player_groups)
            states = tracking_util.initiate_states(initial_detections)
            occluded = [0]*len(initial_detections)
            possible_occlusion = [0]*len(initial_detections)

            with open("/home/ivan/Desktop/trajektorije.txt","w") as file:
                for j in range(len(initial_detections)):
                    ids.append(j)
                    x1,x2,y1,y2 = initial_detections[j]
                    file.write(str(i) + " " + str(ids[j]) + " " + str(player_groups[j]) + " " + str(y1) + " " + str(y2) + " " + str(x1) + " " + str(x2) + "\n")
        else:
            if i%25==0:
                kNN_classifier = tracking_util.train_kNN(hists_train, player_groups_train)
            with open("/home/ivan/Desktop/detectionsyolo/frame" + str(i+1) + ".png.txt", "r") as f:
                detections_aux = f.readlines()
            detections_aux = [detection.strip().split() for detection in detections_aux]
            detections = [(int(detection[0]), int(detection[2]), int(detection[1]), int(detection[3])) for detection in detections_aux]
            detection_score = [float(detections_aux[j][5]) for j in range(len(detections)) if ((detections[j][2]-detections[j][0])*(detections[j][3]-detections[j][1])) >= 100]
            detections = [detection for detection in detections if ((detection[2]-detection[0])*(detection[3]-detection[1])) >= 100]

            avg_h, avg_w = tracking_util.calcAvgValues(detections)

            temps = []
            mot_app_joint_prob = []
            app_det_joint_prob = []
            motion_prob = []
            possible_tracks = []
            all_possible_positions = []
            tracks_distance = []
            pot_possible_tracks = []
            tracks_color_prob = []
            det_dis_tracks = []
            det_diff_dis_tracks = []
            different_team_assigned = []
            for j in range(len(detections)):
                det_dis_tracks.append(list())
                det_diff_dis_tracks.append(list())
                pot_possible_tracks.append(list())
                possible_tracks.append(list())

            for j in range(len(tracks)):
                all_possible_positions.append(list())
                different_team_assigned.append(0)

            hists = tracking_util.calc_hists(detections, img_hsv)
            predicted_states = []
            detection_groups = [kNN_classifier.predict(np.hstack((hist[0], hist[1])).reshape(1,-1)) for hist in hists]
            for j in range(len(tracks)):
                vector, reference = tracks[j]
                predict_state = np.dot(transitionMatrix, states[j]) + np.random.multivariate_normal(mean, processNoiseCov, check_valid='ignore')
                predicted_states.append(predict_state)
                predicted = predict_state[:2]

                track_mot_app_joint = dict()
                track_app_det_joint = dict()
                track_motion_prob = dict()
                detection_distance = dict()
                detection_distance_diff_group = dict()
                pot_track_mot_app_joint = dict()
                pot_track_app_det_joint = dict()
                pot_track_motion_prob = dict()
                potential_detections = []
                pot_positions = []
                color_probs = dict()
                pot_color_probs = dict()

                for k in range(len(detections)):
                    x1,x2,y1,y2 = detections[k]
                    coords = np.array([[y1 + (y2-y1)/2.], [x1 + (x2-x1)/2.]])
                    distance = np.sqrt((predicted[0][0]-coords[0][0])**2 + (predicted[1][0]-coords[1][0])**2)

                    if distance <= r_max and player_groups[j]==detection_groups[k][0]: #and len(all_possible_positions[j])==0:
                        color_prob = tracking_util.calcApp(img, reference, hists[k])
                        color_probs[k] = color_prob
                        mot_prob = factor*np.exp(-(distance**2/deviation**2))#calcMotion(deviation, predicted, coords)
                        track_mot_app_joint[k] = color_prob*mot_prob #dodaje se umnozak app x mot za neku koordinatu
                        track_app_det_joint[k] = color_prob*detection_score[k]
                        track_motion_prob[k] = mot_prob
                        all_possible_positions[j].append(detections[k])
                        if (j, tracks[j]) not in possible_tracks[k]:
                            possible_tracks[k].append((j, tracks[j]))
                        continue

                    if distance > r_max and distance <= max_distance and detection_groups[k][0]==player_groups[j]: #and len(detection_distance)==0:
                        if (j, tracks[j]) not in det_dis_tracks[k]:
                            det_dis_tracks.append((j, tracks[j]))
                            detection_distance[k] = distance
                        continue

                    if player_groups[j] != detection_groups[k][0] and distance <= r_max: #and len(pot_positions)==0:
                        color_prob = tracking_util.calcApp(img, reference, hists[k])
                        pot_color_probs[k] = color_prob
                        mot_prob = factor*np.exp(-(distance**2/deviation**2))
                        pot_track_mot_app_joint[k] = color_prob*mot_prob #dodaje se umnozak app x mot za neku koordinatu
                        pot_track_app_det_joint[k] = color_prob*detection_score[k]
                        pot_track_motion_prob[k] = mot_prob
                        pot_positions.append(detections[k])
                        if (j, tracks[j]) not in pot_possible_tracks[k]:
                            pot_possible_tracks[k].append((j, tracks[j]))
                        continue

                    if distance > r_max and distance <= max_distance and detection_groups[k][0]!=player_groups[j]: #and len(detection_distance_diff_group)==0:
                        if (j, tracks[j]) not in det_diff_dis_tracks[k]:
                            detection_distance_diff_group[k] = distance
                            det_diff_dis_tracks.append((j, tracks[j]))
                        continue

                occlusion = False
                if len(all_possible_positions[j]) == 0 and len(pot_positions) > 0 and len(detection_distance) == 0: #and len(detection_distance_diff_group)==0:
                    different_team_assigned[j] = 1
                    mot_app_joint_prob.append(pot_track_mot_app_joint)
                    app_det_joint_prob.append(pot_track_app_det_joint)
                    motion_prob.append(pot_track_motion_prob)
                    tracks_color_prob.append(pot_color_probs)
                    all_possible_positions[j] = copy.deepcopy(pot_positions)
                    for key, value in pot_track_mot_app_joint.items():
                        elements = pot_possible_tracks[key]
                        for track in elements:
                            possible_tracks[key].append(track)
                    occluded[j]+=1
                    occlusion = True
                else:
                    mot_app_joint_prob.append(track_mot_app_joint)
                    app_det_joint_prob.append(track_app_det_joint)
                    motion_prob.append(track_motion_prob)
                    tracks_color_prob.append(color_probs)
                    for key, value in pot_track_mot_app_joint.items():
                        pot_possible_tracks[key].clear()

                if len(detection_distance) > 0:
                    tracks_distance.append(detection_distance)
                else:
                    if occlusion == False:
                        possible_occlusion[j] = 1
                    tracks_distance.append(detection_distance_diff_group)


            new_tracks = list()
            for j in range(len(detections)):
                possible_track = possible_tracks[j] # moguće putanje
                if len(possible_track) > 0:
                    for k in range(len(possible_track)):
                        pos, track = possible_track[k] #trenutna putanja
                        if different_team_assigned[k] == 0:
                            all_positions = all_possible_positions[pos] #sve moguće pozicije s^m na putanji
                            mot_app_track = mot_app_joint_prob[pos] # vjerojatnost app modela i mot modela
                            max_mot_app_track = max(mot_app_track.values())
                            prob = mot_app_track[j]
                            if prob < max_mot_app_track:
                                if detections[j] in all_positions:
                                    all_positions.remove(detections[j]) #[pos for pos in all_positions if not(pos == detect_coord[j]).all()]
                                    all_possible_positions[pos] = copy.deepcopy(all_positions)
                        if different_team_assigned[k] == 1:
                            track_mot = motion_prob[pos] #vjerojatnost pomaka za putanje i pozicije xt i sm
                            all_positions = all_possible_positions[pos] #sve moguće pozicije s^m na putanji
                            max_mot = max(track_mot.values())
                            prob = track_mot[j]
                            if prob < max_mot:
                                if detections[j] in all_positions:
                                    all_positions.remove(detections[j]) #[pos for pos in all_positions if not(pos == detect_coord[j]).all()]
                                    all_possible_positions[pos] = copy.deepcopy(all_positions)

            new_detections = []
            indices_of_tracks = []
            possible_new_detections = []
            for j in range(len(all_possible_positions)):
                all_pos = all_possible_positions[j]
                if len(all_pos) == 0: #and len(tracks_distance[j])>0:
                    detections_distance = tracks_distance[j]
                    if len(detections_distance)!=0:
                        if possible_occlusion[j]==1:
                            occluded[j]+=1
                        vector, reference = tracks[j]
                        joint = dict()
                        for det_key, det_val in detections_distance.items():
                            x1,x2,y1,y2 = detections[det_key]
                            detection_hist = hists[det_key]
                            app = tracking_util.calcApp(img_hsv, reference, detection_hist)
                            motion = tracking_util.calcMotion(deviation, vector, np.array([[y1 + (y2-y1)/2], [x1 + (x2-x1)/2]]))
                            joint[det_key] = app*motion

                        key = max(joint, key=joint.get)
                        max_detect_distance = max(joint.values())
                        detection = detections[key]
                    else:
                        vector, reference = tracks[j]
                        group = player_groups[j]
                        xs = max(0, int(np.round(vector[1][0] - avg_w/2)))
                        ys = max(0, int(np.round(vector[0][0] - avg_h/2)))
                        search_areas = [(xs, xs+avg_w, ys, ys+avg_h), (xs+6, xs+avg_w+6, ys, ys+avg_h), (xs-6, xs+avg_w-6, ys, ys+avg_h), (xs, xs+avg_w, ys+4, ys+avg_h+4),
                        (xs, xs+avg_w, ys-4, ys+avg_h-4), (xs-6, xs+avg_w-6, ys-4, ys+avg_h-4), (xs-6, xs+avg_w-6, ys+4, ys+avg_h+4), (xs+6, xs+avg_w+6, ys-4, ys+avg_h-4),
                        (xs+6, xs+avg_w+6, ys+4, ys+avg_h+4)]

                        search_areas = [search_area for search_area in search_areas if (search_area[1] > 0 and search_area[0] > 0 and search_area[3] > 0 and search_area[2] > 0)
                                        and search_area[1] < 3260 and search_area[0] < 3260 and search_area[3] < 570 and search_area[2] < 570]
                        if len(search_areas) == 0:
                            continue
                        search_hists = tracking_util.calc_hists(search_areas, img_hsv)
                        search_app = []
                        search_motion = []
                        group_sr = [kNN_classifier.predict(np.hstack((search_hist[0], search_hist[1])).reshape(1, -1))[0] for search_hist in search_hists]
                        search_areas = [search_areas[k] for k in range(len(search_areas)) if group_sr[k]==group]
                        search_hists = [search_hists[k] for k in range(len(search_hists)) if group_sr[k]==group]
                        group_sr = [group_num for group_num in group_sr if group_num==group]
                        if len(group_sr)==0:
                            continue
                        for k in range(len(search_hists)):
                            group_num = group_sr[k]
                            search_app.append(tracking_util.calcApp(img_hsv, reference, search_hists[k]))
                            search_motion.append(tracking_util.calcMotion(deviation, vector[:2], search_areas[k]))

                        joint_prob = []
                        for k in range(len(search_app)):
                            joint_prob.append(search_app[k]*search_motion[k])
                        index = joint_prob.index(max(joint_prob))
                        detection = search_areas[index]
                        x1,x2,y1,y2 = detection
                    possible_new_detections.append(detection)
                    all_possible_positions[j].append(detection)

            lost = 0
            for j in range(len(all_possible_positions)):
                if len(all_possible_positions[j]) == 0:
                    lost+=1

            updated_tracks = copy.deepcopy(tracks)
            for j in range(len(tracks)):
                positions = all_possible_positions[j]
                if len(positions)==0:
                    continue
                elif len(positions)==1:
                    x1,x2,y1,y2 = positions[0]
                    measurement = np.array([[y1 + (y2-y1)/2.], [x1 + (x2-x1)/2.]])
                    vector, reference = tracks[j]
                    vy = measurement[0][0] - states[j][0][0]
                    vx = measurement[1][0] - states[j][1][0]
                    new_track_pos = np.array([[measurement[0][0]], [measurement[1][0]], [vy], [vx]])
                    states[j] = new_track_pos
                    updated_tracks[j] = (new_track_pos, reference)


            dist_min = 7
            detects_distance = []
            for j in range(len(detections)):
                detect_dis_values = []
                x1,x2,y1,y2 = detections[j]
                coords = np.array([[y1 + (y2-y1)/2], [x1 + (x2-x1)/2]])
                for k in range(len(updated_tracks)):
                    vector, _ = updated_tracks[k]
                    distance = np.sqrt((coords[0][0]-vector[0][0])**2 + (coords[1][0]-vector[1][0])**2)
                    detect_dis_values.append(distance)
                detects_distance.append(detect_dis_values)


            updated_tracks, ids, player_groups = tracking_util.remove_duplicate_tracks(states, predicted_states, updated_tracks, ids, player_groups)
            updated_tracks, ids, player_groups, occluded = tracking_util.remove_occluded_over_threshold(occluded_thershold, occluded, states, updated_tracks, ids, player_groups)

            new_tracks = list()
            for j in range(len(detections)):
                if min(detects_distance[j]) <= dist_min or lost==0:
                    continue
                possible_track = possible_tracks[j] # moguće putanje
                if len(possible_track)==0: #dodati kNN provjeru jer ako ne pripada niti jednoj bitnoj grupi onda se ne dodaje
                    track_det_dis = det_dis_tracks[j]
                    if len(track_det_dis)==0:
                        track_diff_det_dis = det_diff_dis_tracks[j]
                        if len(track_diff_det_dis)==0:
                            x1,x2,y1,y2 = detections[j]
                            hist = tracking_util.calc_hists([detections[j]], img_hsv)[0]
                            hist_merged = np.hstack((hist[0], hist[1])).reshape(1, -1)
                            if kNN_classifier.predict(hist_merged)[0] < 5:
                                x1,x2,y1,y2 = detections[j]
                                state = tracking_util.initiate_states([detections[j]])[0]
                                states.append(state)
                                track = (np.array([[y1 + (y2-y1)/2], [x1 + (x2-x1)/2], [0], [0]]), hist)
                                new_tracks.append(track)
                                player_groups.append(kNN_classifier.predict(hist_merged)[0])
                                ids.append(max(ids) + 1)

            updated_tracks = updated_tracks + new_tracks
            occluded = occluded + [0]*len(new_tracks)
            possible_occlusion = [0]*len(occluded)

            with open("/home/ivan/Desktop/trajektorije.txt", "a") as file:
                for j in range(len(updated_tracks)):
                    vector, reference = updated_tracks[j]
                    reference_merged = np.hstack((reference[0], reference[1])).reshape (1,-1)
                    group = kNN_classifier.predict(reference_merged)[0]
                    y,x = vector[:2]
                    y1 = int(np.ceil(y[0] - avg_h/2))
                    y2 = int(np.ceil(y[0] + avg_h/2))
                    x1 = int(np.ceil(x[0] - avg_w/2))
                    x2 = int(np.ceil(x[0] + avg_w/2))
                    hist_det = tracking_util.calc_hists([(x1,x2,y1,y2)], img_hsv)[0]
                    hist_det_1, hist_det_2 = hist_det
                    group_det = kNN_classifier.predict(np.hstack((hist_det_1, hist_det_2)).reshape(1,-1))[0]
                    if group==group_det and group_det < 5:
                        hists_train.append(hist_det)
                        player_groups_train.append(group_det)
                        updated_tracks[j] = (vector, hist_det)
                    file.write(str(i) + " " + str(ids[j]) + " " + str(player_groups[j]) + " " + str(y1) + " " + str(y2) + " " + str(x1) + " " + str(x2) + "\n")
            tracks = copy.deepcopy(updated_tracks)
            new_tracks.clear()
            updated_tracks.clear()

    time_end = time.time()
    length = time_end - time_start
    average = length/1000
    print(average)
