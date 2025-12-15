import sys
import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from tracking_utils import calculate_speed, angle_and_quad, calculate_distance
from keypointdiff_and_scalefactor import keypoint_differences_in_metres, scaleFactor, centre_court
from ball_analysis import get_ball_shot_frames
from utils import scene_detect
from scoring import scores
from PlayerMovementMetric import generate_outputPlayer
from video_resolution_resizer import resize_video
from commentary import generate_output
from AI_Insights import generate_outputAI
import math
import argparse
try:
    import torch
except Exception:
    torch = None
import os


def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps


def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img


def main(frames, fps, scenes, bounces, ball_track, df_track_ball, comments, insights, score, homography_matrices, kps_court,
         persons_top, persons_bottom, draw_trace=0, trace=1, Rally_shot=0, player1=0, player2=0):
    """
    :params
        frames: list of original images
        fps: frame per second of the video
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        persons_top: list of person bboxes located in the top of tennis court
        persons_bottom: list of person bboxes located in the bottom of tennis court
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images
    """
    NumRally = 1
    scale1, scale2, scale3 = scaleFactor(kps_court)
    l = keypoint_differences_in_metres(kps_court)
    ratio = l[3]
    scale = [scale1, scale2]
    scale_player = [((0.5645833 / 20) / ratio), (0.5645833 / 20)]
    distancecov = np.zeros(2, dtype=float)
    dist_pos = ((10, 180), (10, 200))
    speed_pos = ((10, 220), (10, 240))
    comment_index = 0
    insights_index = 0
    speed = np.zeros(2, dtype=float)
    last_speed = np.zeros(2, dtype=float)
    last_ball_speed = 0
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices]
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]

        eps = 1e-15
        scene_rate = sum_track / (len_track + eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()

            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i].copy()
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0] is not None and not math.isnan(ball_track[i][0]):
                    if draw_trace == 1:
                        for j in range(0, trace):
                            if i - j >= 0:
                                if ball_track[i - j][0] is not None and not math.isnan(ball_track[i - j][0]):
                                    draw_x = int(ball_track[i - j][0])
                                    draw_y = int(ball_track[i - j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                                         radius=3, color=(0, 255, 0), thickness=2)

                        img_res = cv2.putText(img_res, 'ball',
                                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.8,
                                              thickness=2,
                                              color=(0, 255, 0))

                    else:
                        img_res = cv2.circle(img_res, (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball',
                                              org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.8,
                                              thickness=2,
                                              color=(0, 255, 0))

                    # Tracking Ball Speed
                    if i != 0:
                        if (i % 15 == 0):
                            ball_speed = (calculate_speed(ball_track[i - 1], ball_track[i], fps)) * scale3 * (0.4)
                            last_ball_speed = ball_speed
                        img_res = cv2.putText(img_res, f"speed: {last_ball_speed:.2f} m/s",
                                              org=(int(ball_track[i][0]) + 12, int(ball_track[i][1]) + 25),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.6,
                                              thickness=2,
                                              color=(0, 165, 255))
                        centre = (640, 360)  # centre_court(kps_court)
                        ball_angle, ball_quad = angle_and_quad(centre, ball_track[i])
                        img_res = cv2.putText(img_res, f"{ball_angle:.2f} {ball_quad}",
                                              org=(int(ball_track[i][0]) + 12, int(ball_track[i][1]) + 45),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.6,
                                              thickness=2,
                                              color=(0, 165, 255))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                             radius=0, color=(0, 0, 255), thickness=10)
                        # Draw point number
                        img_res = cv2.putText(img_res, str(j + 1),
                                              org=(int(kps_court[i][j][0, 0]) + 10, int(kps_court[i][j][0, 1]) + 10),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.5,
                                              thickness=1,
                                              color=(0, 0, 255))

                height, width, _ = img_res.shape

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                           radius=0, color=(0, 255, 255), thickness=50)

                minimap = court_img.copy()

                # draw persons
                person1 = persons_top[i]
                person2 = persons_bottom[i]

                # Player 1
                for j, person in enumerate(person1):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # Define the player's label (Player 1, Player 2, etc.)
                        player_label = "Player 1"  # f"Player {j+1}"

                        # Add the label text above the bounding box
                        img_res = cv2.putText(img_res,
                                              player_label,
                                              (int(person_bbox[0]), int(person_bbox[1]) - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.9,
                                              (255, 0, 0),
                                              2)

                        # Draw player speed and direction
                        if i != 0:
                            prev_persons = persons_top[i - 1]
                            for k, prev_person in enumerate(prev_persons):
                                distancecov[0] = distancecov[0] + calculate_distance(prev_person[1], person[1]) * \
                                    scale_player[0]
                                if (i % 15 == 0):
                                    speed[0] = calculate_speed(prev_person[1], person[1], fps) * scale[0] / 2.7
                                    last_speed[0] = speed[0]

                        # Transfer person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                             radius=0, color=(255, 0, 0), thickness=80)

                # Player 2
                for j, person in enumerate(person2):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # Define the player's label (Player 1, Player 2, etc.)
                        player_label = "Player 2"  # f"Player {j+1}"

                        # Add the label text above the bounding box
                        img_res = cv2.putText(img_res,
                                              player_label,
                                              (int(person_bbox[0]), int(person_bbox[1]) - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.9,
                                              (255, 0, 0),
                                              2)

                        # Draw player speed and direction
                        if i != 0:
                            prev_persons = persons_bottom[i - 1]
                            for q, prev_person in enumerate(prev_persons):
                                distancecov[1] = distancecov[1] + calculate_distance(prev_person[1], person[1]) * \
                                    scale_player[1]
                                if (i % 15 == 0):
                                    speed[1] = calculate_speed(prev_person[1], person[1], fps) * scale[1] / 2.7
                                    last_speed[1] = speed[1]

                        # transmit person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                             radius=0, color=(255, 0, 0), thickness=80)

                for j in range(2):
                    img_res = cv2.putText(img_res, f"Total Distance Covered By Player {j+1}: {distancecov[j]:.2f} m",
                                          org=dist_pos[j],
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=0.6,
                                          thickness=2,
                                          color=(0, 165, 255))
                    img_res = cv2.putText(img_res, f"Speed(Player {j+1}): {last_speed[j]:.2f} m/s",
                                          org=speed_pos[j],
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=0.6,
                                          thickness=2,
                                          color=(0, 165, 255))

                img_res = cv2.putText(img_res, f"Number of Rallies: {NumRally}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (0, 165, 255), 2)
                img_res = cv2.putText(img_res, f"Length of Rally: {Rally_shot}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (0, 165, 255), 2)
                img_res = cv2.putText(img_res, f"Number of Shots(Player1): {player1}", (10, 130),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                img_res = cv2.putText(img_res, f"Number of Shots(Player2): {player2}", (10, 150),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                try:
                    if (df_track_ball['ball_hit'][i] == 1):
                        Rally_shot = Rally_shot + 1
                        if (df_track_ball['delta_y'][i] > 0):
                            player2 = player2 + 1
                        if (df_track_ball['delta_y'][i] < 0):
                            player1 = player1 + 1
                except Exception:
                    # If df_track_ball doesn't have expected columns/indices, ignore for now
                    pass

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap

                # Draw frame number
                img_res = cv2.putText(img_res, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Scores box
                img_res = cv2.rectangle(img_res, (1000, 505), (1270, 595), [255, 0, 0], 2)
                img_res = cv2.putText(img_res, "Scores:", (1010, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Use try/except to avoid KeyError/IndexError crashing the whole run
                try:
                    player1_score = score['1'][i + 1]
                except Exception:
                    player1_score = ""
                try:
                    player2_score = score['2'][i + 1]
                except Exception:
                    player2_score = ""
                try:
                    event_score = score['Event'][i + 1]
                except Exception:
                    event_score = ""

                img_res = cv2.putText(img_res, f"Player 1: {player1_score}", (1010, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (0, 255, 0), 2)
                img_res = cv2.putText(img_res, f"Player 2: {player2_score}", (1010, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (0, 255, 0), 2)
                img_res = cv2.putText(img_res, f"Event Detected: {event_score}", (1010, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (0, 255, 0), 2)

                # AI Insights
                if (i != 0):
                    if (i % 90 == 0):
                        try:
                            img_res = cv2.putText(img_res, f"AI Insights: {insights[insights_index]}", (10, 60),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        except Exception:
                            pass
                        insights_index = insights_index + 1
                    else:
                        try:
                            img_res = cv2.putText(img_res, f"AI Insights: {insights[insights_index]}", (10, 60),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        except Exception:
                            pass
                else:
                    try:
                        img_res = cv2.putText(img_res, f"AI Insights: {insights[insights_index]}", (10, 60),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    except Exception:
                        pass

                # Checking if Rally Done
                try:
                    if (score['Rally_Break'][i + 1] == 1):
                        Rally_shot = 0
                        player1 = 0
                        player2 = 0
                        NumRally = NumRally + 1
                except Exception:
                    pass

                # Writing Commentary
                if (i != 0):
                    if (i % 90 == 0):
                        try:
                            img_res = cv2.putText(img_res, f"Aagam: {comments[comment_index]}", (50, 690),
                                                  cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
                        except Exception:
                            pass
                        comment_index = comment_index + 1
                    else:
                        try:
                            img_res = cv2.putText(img_res, f"Aagam: {comments[comment_index]}", (50, 690),
                                                  cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
                        except Exception:
                            pass
                else:
                    try:
                        img_res = cv2.putText(img_res, f"Aagam: {comments[comment_index]}", (50, 690),
                                              cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
                    except Exception:
                        pass

                imgs_res.append(img_res)

        else:
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]]
    return imgs_res


def write(imgs_res, fps, path_output_video):
    if not imgs_res:
        print("No frames to write.")
        return
    height, width = imgs_res[0].shape[:2]
    # Ensure output directory exists
    output_dir = os.path.dirname(path_output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for num in range(len(imgs_res)):
        frame = imgs_res[num]
        out.write(frame)
    out.release()


def process_video(path_input_video, path_output_video):
    path_ball_track_model = 'model/model_best_player_new.pt'
    path_court_model = 'model/model_tennis_court_det.pt'
    path_bounce_model = 'model/ctb_regr_bounce.cbm'
    path_output_frame = 'output_frames'

    if torch is None:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = read_video(path_input_video)
    scenes = scene_detect(path_input_video)

    print('ball detection')
    ball_detector = BallDetector(path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)
    ball_track = ball_detector.interpolate_ball_track(ball_track, 25)
    df_track_ball, Rally_shots, player1, player2 = get_ball_shot_frames(ball_track)

    print('court detection')
    court_detector = CourtDetectorNet(path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('person detection')
    person_detector = PersonDetector(device)
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=True)

    # Generating Comments
    commentary_results = generate_output(path_output_frame, path_input_video)
    comments = commentary_results[0] if commentary_results is not None else []

    # Generating Insights
    insights_results = generate_outputAI(path_output_frame, path_input_video)
    insights = insights_results[0] if insights_results is not None else []

    # Location where the csv is saved
    save_path = 'Player_Metrics_Analysis/metrics.csv'
    # Player Movement Metrics - Storing as a csv in a file
    PlayerMetrics = generate_outputPlayer(path_output_frame, path_input_video, save_path, fps)

    # bounce detection
    bounce_detector = BounceDetector(path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # Score and Event Detection
    score = scores(homography_matrices, kps_court, len(frames), ball_track, df_track_ball, bounces)

    # ensure output dir exists
    out_dir = os.path.dirname(path_output_video)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    imgs_res = main(frames, fps, scenes, bounces, ball_track, df_track_ball, comments, insights, score, homography_matrices,
                    kps_court, persons_top, persons_bottom, draw_trace=1)

    write(imgs_res, fps, path_output_video)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_video_path> <output_video_path>")
        sys.exit(1)
    print("Processing Started")
    path_input_video = sys.argv[1]
    path_output_video = sys.argv[2]
    process_video(path_input_video, path_output_video)
