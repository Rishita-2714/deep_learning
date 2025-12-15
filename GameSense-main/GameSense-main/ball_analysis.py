import pandas as pd


def get_ball_shot_frames(ball_track):
        # Extract the ball coordinates from the list of tuples
        ball_track = [list(x) for x in ball_track]
        
        # Convert the list into a pandas DataFrame
        df_ball_track = pd.DataFrame(ball_track, columns=['x', 'y'])

        df_ball_track['rolling_mean_y'] = df_ball_track['y'].rolling(window=5,min_periods=1, center=False).mean()
        df_ball_track['delta_y'] = df_ball_track['rolling_mean_y'].diff()
        df_ball_track['ball_hit']=0
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_track)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_track['delta_y'].iloc[i] >0 and df_ball_track['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_track['delta_y'].iloc[i] <0 and df_ball_track['delta_y'].iloc[i+1] >0
            
            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_track['delta_y'].iloc[i] >0 and df_ball_track['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_track['delta_y'].iloc[i] <0 and df_ball_track['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_track['ball_hit'].iloc[i] = 1
        df_hit = df_ball_track[df_ball_track['ball_hit']==1]
        frame_nums_with_ball_hits = df_ball_track[df_ball_track['ball_hit']==1].index.tolist()
        player1 = df_hit[df_hit['delta_y']<0].index.tolist()
        player2 = df_hit[df_hit['delta_y']>0].index.tolist()
        print(frame_nums_with_ball_hits)
        print(player1)
        print(player2)
        print(df_ball_track)
        return df_ball_track, len(frame_nums_with_ball_hits), len(player1), len(player2)