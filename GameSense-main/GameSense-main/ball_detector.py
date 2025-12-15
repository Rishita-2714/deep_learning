from tracknet import BallTrackerNet
try:
    import torch
except Exception:
    torch = None
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import pandas as pd

class BallDetector:
    def __init__(self, path_model=None, device='cuda'):
        self.model = None
        # Only create/load model if torch available
        if torch is not None:
            self.model = BallTrackerNet(input_channels=9, out_channels=256)
            self.device = device
            if path_model:
                self.model.load_state_dict(torch.load(path_model, map_location=device))
                self.model = self.model.to(device)
                self.model.eval()
        else:
            self.device = 'cpu'
        self.width = 640
        self.height = 360
        
    def interpolate_ball_track(self, ball_track, max_gap):
        # Extract the ball coordinates from the list of tuples
        ball_track = [list(x) for x in ball_track]
        
        # Convert the list into a pandas DataFrame
        df_ball_track = pd.DataFrame(ball_track, columns=['x', 'y'])
        Break_Seq = df_ball_track[df_ball_track.x.isnull()==False].index.tolist()
        # Interpolate the missing values
        temp = pd.DataFrame()
        for i in range(len(Break_Seq)-1):
            if(Break_Seq[i+1]-Break_Seq[i]<max_gap):
                temp = df_ball_track.iloc[Break_Seq[i]:Break_Seq[i+1],:].interpolate().bfill()
            else:
                temp = df_ball_track.iloc[Break_Seq[i]:Break_Seq[i+1],:]
            df_ball_track.iloc[Break_Seq[i]:Break_Seq[i+1],:] = temp
        # df_ball_track = df_ball_track.interpolate()
        # df_ball_track = df_ball_track.bfill()
        # Convert the DataFrame back to a list of tuples
        ball_track = [tuple(x) for x in df_ball_track.to_numpy().tolist()]

        return ball_track



    
    def infer_model(self, frames):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        # If no model available, return placeholder track
        if self.model is None:
            for num in range(2, len(frames)):
                ball_track.append((None, None))
            return ball_track

        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track
    

    # def get_ball_shot_frames(self, ball_track):
    #     # Extract the ball coordinates from the list of tuples
    #     ball_track = [list(x) for x in ball_track]
        
    #     # Convert the list into a pandas DataFrame
    #     df_ball_track = pd.DataFrame(ball_track, columns=['x', 'y'])

    #     df_ball_track['rolling_mean_y'] = df_ball_track['y'].rolling(window=5,min_periods=1, center=False).mean()
    #     df_ball_track['delta_y'] = df_ball_track['rolling_mean_y'].diff()
    #     df_ball_track['ball_hit']=0
    #     minimum_change_frames_for_hit = 25
    #     for i in range(1,len(df_ball_track)- int(minimum_change_frames_for_hit*1.2) ):
    #         negative_position_change = df_ball_track['delta_y'].iloc[i] >0 and df_ball_track['delta_y'].iloc[i+1] <0
    #         positive_position_change = df_ball_track['delta_y'].iloc[i] <0 and df_ball_track['delta_y'].iloc[i+1] >0
            
    #         if negative_position_change or positive_position_change:
    #             change_count = 0 
    #             for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
    #                 negative_position_change_following_frame = df_ball_track['delta_y'].iloc[i] >0 and df_ball_track['delta_y'].iloc[change_frame] <0
    #                 positive_position_change_following_frame = df_ball_track['delta_y'].iloc[i] <0 and df_ball_track['delta_y'].iloc[change_frame] >0

    #                 if negative_position_change and negative_position_change_following_frame:
    #                     change_count+=1
    #                 elif positive_position_change and positive_position_change_following_frame:
    #                     change_count+=1
    
    #     if change_count>minimum_change_frames_for_hit-1:
    #         df_ball_track['ball_hit'].iloc[i] = 1
    #     df_hit = df_ball_track[df_ball_track['ball_hit']==1]
    #     frame_nums_with_ball_hits = df_ball_track[df_ball_track['ball_hit']==1].index.tolist()
    #     player1 = df_hit[df_hit['delta_y']<0].index.tolist()
    #     player2 = df_hit[df_hit['delta_y']>0].index.tolist()
    #     print(frame_nums_with_ball_hits)
    #     print(player1)
    #     print(player2)
    #     print(df_ball_track)
    #     return df_ball_track, len(frame_nums_with_ball_hits), len(player1), len(player2)
    
    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
