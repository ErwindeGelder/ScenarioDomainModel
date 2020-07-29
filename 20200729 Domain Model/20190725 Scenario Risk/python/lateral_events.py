"""
Small description.
"""

__author__ = 'Jeroen Manders'

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

def get_ego(settings, ego_df, plot=False):
    ego_df['line_center_y'] = (ego_df['line_left_y']+ego_df['line_right_y'])/2
    
    ego_df['line_left_y_conf'] = ego_df['line_left_y']
    ego_df.loc[ego_df['lines_0_confidence']!=3, ('line_left_y_conf')] = np.nan
    ego_df['line_right_y_conf'] = ego_df['line_right_y']
    ego_df.loc[ego_df['lines_1_confidence']!=3, ('line_right_y_conf')] = np.nan
    ego_df['line_center_y_conf'] = ego_df['line_center_y']
    ego_df.loc[(ego_df['lines_0_confidence']+ego_df['lines_1_confidence'])!=6, ('line_center_y_conf')] = np.nan
    
    ego_df['line_left_y_conf_down'] = ego_df['line_left_y_conf'] - ego_df['line_left_y_conf'].rolling(window=settings['fps']).max()
    ego_df['line_right_y_conf_up'] = ego_df['line_right_y_conf'] - ego_df['line_right_y_conf'].rolling(window=settings['fps']).min()
    
    # FOR METRICS
    ego_df['line_center_y_conf_savgol'] = savgol_filter(ego_df['line_center_y_conf'],settings['fps']+1,0)
    ego_df['line_center_y_conf_savgol_lateral_speed'] = (ego_df['line_center_y_conf_savgol']-ego_df['line_center_y_conf_savgol'].shift(1))*settings['fps']
    # delete around mobileye lane resets
    for i in ego_df.index[1:]:
        if abs(ego_df['line_center_y_conf'][i]-ego_df['line_center_y_conf'][np.round(i-1/settings['fps'],2)])>=1:
            min_i = np.max((ego_df.index[0],i-0.5))
            max_i = np.min((ego_df.index[-1],i+0.5))
            ego_df.loc[min_i:max_i,'line_center_y_conf_savgol'] = 0
            ego_df.loc[min_i:max_i,'line_center_y_conf_savgol_lateral_speed'] = 0
    
    events = [(ego_df.index[0],'fl')]
    event = 'fl'
    for i in ego_df.index[1:]:
        # Left lane change (out of ego lane)
        if event!='l' and ego_df['line_left_y'][i]<0<=ego_df['line_left_y'][np.round(i-1/settings['fps'],2)] and abs(ego_df['line_left_y'][i]-ego_df['line_left_y'][np.round(i-1/settings['fps'],2)])<1 and ego_df['line_left_y_conf_down'][i]<-0.25:
            begin_i = np.max((i-10, events[-1][0]))
            begin_i = ego_df[begin_i:i][ego_df.loc[begin_i:i,('line_left_y_conf_down')].gt(-0.25)]
            if len(begin_i)>0:
                begin_i = begin_i.index[-1]
                begin_i = ego_df.loc[np.max((begin_i-1, ego_df.index[0])):begin_i,'line_left_y'].idxmax()
                if abs(ego_df['line_left_y'][begin_i]-ego_df['line_left_y'][np.round(begin_i+1/settings['fps'],2)])<1:
                    event = 'l'
                    events.append((begin_i, event))
        # Right lane change (out of ego lane)
        elif event!='r' and ego_df['line_right_y'][np.round(i-1/settings['fps'],2)]<=0<ego_df['line_right_y'][i] and abs(ego_df['line_right_y'][i]-ego_df['line_right_y'][np.round(i-1/settings['fps'],2)])<1 and ego_df['line_right_y_conf_up'][i]>0.25:
            begin_i = np.max((i-10, events[-1][0]))
            begin_i = ego_df[begin_i:i][ego_df.loc[begin_i:i,('line_right_y_conf_up')].lt(0.25)]
            if len(begin_i)>0:
                begin_i = begin_i.index[-1]
                begin_i = ego_df.loc[np.max((begin_i-1, ego_df.index[0])):begin_i,'line_right_y'].idxmin()
                if abs(ego_df['line_right_y'][begin_i]-ego_df['line_right_y'][np.round(begin_i+1/settings['fps'],2)])<1:
                    event = 'r'
                    events.append((begin_i, event))
        # Follow-Lane
        elif event!='fl' and ego_df['line_center_y_conf'][i]!=0 and ego_df['line_right_y_conf_up'][i]<0.25 and ego_df['line_left_y_conf_down'][i]>-0.25:
            event = 'fl'
            events.append((i,event))

    if plot:
        fig, axarr = plt.subplots(1,3, sharex=True, figsize=(19.2,6.4))
        fig.suptitle('Lateral target events')
        for event in events:
            axarr[0].vlines(event[0], -4, 4, linestyles='dashed', colors='grey')
            axarr[1].vlines(event[0], -0.5, 0.25, linestyles='dashed', colors='grey')
            axarr[2].vlines(event[0], -0.25, 0.5, linestyles='dashed', colors='grey')
            axarr[0].text(event[0], -4, event[1])
            axarr[1].text(event[0], -0.5, event[1])
            axarr[2].text(event[0], -0.25, event[1])
        axarr[0].plot(ego_df.index, ego_df['line_left_y_conf'], color='blue', label='left line')
        axarr[0].plot(ego_df.index, ego_df['line_right_y_conf'], color='red', label='right line')
        axarr[0].plot(ego_df.index, ego_df['line_center_y_conf'], color='magenta', label='center of lane')
        axarr[0].legend()
        axarr[0].plot(ego_df.index, ego_df['line_left_y'], linestyle=':', color='blue')
        axarr[0].plot(ego_df.index, ego_df['line_right_y'], linestyle=':', color='red')
        axarr[0].plot(ego_df.index, ego_df['line_center_y'], linestyle=':', color='magenta')
        axarr[0].hlines(y=[0], xmin=ego_df.index[0], xmax=ego_df.index[-1])
        axarr[0].set_xlabel('time (seconds)')
        axarr[0].set_ylabel('distance (meters)')
        axarr[1].plot(ego_df.index, ego_df['line_left_y_conf_down'], label='left line decrease in last second')
        axarr[1].hlines(y=[0, -0.25], xmin=ego_df.index[0], xmax=ego_df.index[-1])
        axarr[1].set_xlabel('time (seconds)')
        axarr[1].set_ylabel('distance (meters)')
        axarr[1].legend()
        axarr[2].plot(ego_df.index, ego_df['line_right_y_conf_up'], label='right line increase in last second')
        axarr[2].hlines(y=[0, 0.25], xmin=ego_df.index[0], xmax=ego_df.index[-1])
        axarr[2].set_xlabel('time (seconds)')
        axarr[2].set_ylabel('distance (meters)')
        axarr[2].legend()

    return ego_df, events

def get_target(settings, ego_df, target_df, plot=False):
    target_df['line_left_y'] = ego_df['line_left_y'] + ego_df['lines_0_c1']*target_df['pose_x'] + ego_df['lines_0_c2']*np.power(target_df['pose_x'],2) + ego_df['lines_0_c3']*np.power(target_df['pose_x'],3) - target_df['pose_y']
    target_df['line_right_y'] = ego_df['line_right_y'] + ego_df['lines_1_c1']*target_df['pose_x'] + ego_df['lines_1_c2']*np.power(target_df['pose_x'],2) + ego_df['lines_1_c3']*np.power(target_df['pose_x'],3) - target_df['pose_y']

    target_df['line_center_y'] = (target_df['line_left_y']+target_df['line_right_y'])/2
    target_df['line_left_y_conf'] = target_df['line_left_y']
    target_df.loc[ego_df['lines_0_confidence']!=3, ('line_left_y_conf')] = np.nan
    target_df['line_right_y_conf'] = target_df['line_right_y']
    target_df.loc[ego_df['lines_1_confidence']!=3, ('line_right_y_conf')] = np.nan
    target_df['line_center_y_conf'] = target_df['line_center_y']
    target_df.loc[(ego_df['lines_0_confidence']+ego_df['lines_1_confidence'])!=6, ('line_center_y_conf')] = np.nan

    target_df['line_left_y_conf_down'] = target_df['line_left_y_conf'] - target_df['line_left_y_conf'].rolling(window=settings['fps']).max()
    target_df['line_left_y_conf_up'] = target_df['line_left_y_conf'] - target_df['line_left_y_conf'].rolling(window=settings['fps']).min()
    target_df['line_right_y_conf_down'] = target_df['line_right_y_conf'] - target_df['line_right_y_conf'].rolling(window=settings['fps']).max()
    target_df['line_right_y_conf_up'] = target_df['line_right_y_conf'] - target_df['line_right_y_conf'].rolling(window=settings['fps']).min()

    # FOR METRICS
    target_df['line_center_y_conf_savgol'] = savgol_filter(target_df['line_center_y_conf'],settings['fps']+1,0)
    target_df['line_center_y_conf_savgol_lateral_speed'] = (target_df['line_center_y_conf_savgol']-target_df['line_center_y_conf_savgol'].shift(1))*settings['fps']
    # delete around mobileye lane resets
    for i in target_df.index[1:]:
        if abs(ego_df['line_center_y_conf'][i]-ego_df['line_center_y_conf'][np.round(i-1/settings['fps'],2)])>=1:
            min_i = np.max((target_df.index[0],i-0.5))
            max_i = np.min((target_df.index[-1],i+0.5))
            target_df.loc[min_i:max_i,'line_center_y_conf_savgol'] = 0
            target_df.loc[min_i:max_i,'line_center_y_conf_savgol_lateral_speed'] = 0

    events = [(target_df.index[0], 'fl')]
    event = 'fl'
    prev_i = target_df.index[0]
    follow_lane_switch = 0
    for i in target_df.index[1:]:
        follow_lane_switch -= 1
        # Left lane change out of ego lane: cross left_y down
        if event!='lo' and event!='li' and target_df.loc[i,'line_left_y_conf']<0<=target_df.loc[prev_i,'line_left_y_conf'] and abs(target_df.loc[i,'line_left_y_conf']-target_df.loc[prev_i,'line_left_y_conf'])<1 and target_df.loc[i,'line_left_y_conf_down']<-0.25 and target_df.loc[i,'line_center_y_conf']!=np.nan and target_df.loc[i,'line_left_y_conf']>target_df.loc[i,'line_right_y_conf']:
            from_y = (1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            goal_y = -(1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            begin_i = np.max((i-6, events[-1][0]))
            begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf')].gt(from_y)]
            if len(begin_i)>0 and sum(target_df.loc[begin_i.index[-1]:np.round(i-1/settings['fps'],2),'line_left_y_conf'].lt(goal_y/2))==0 and (begin_i.index[-1]==target_df.index[0] or abs(target_df.loc[begin_i.index[-1],'line_left_y_conf']-target_df.loc[np.round(begin_i.index[-1]-1/settings['fps'],2),'line_left_y_conf'])<1):
                begin_i = begin_i.index[-1]
                if len(target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf_down')]==0])>0:
                    begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf_down')]==0].index[-1]
                end_i = np.min((i+6, target_df.index[-1]))
                end_i = target_df[i:end_i][target_df.loc[i:end_i,('line_left_y_conf')].lt(goal_y)]
                if len(end_i)>0 and sum(target_df.loc[i:end_i.index[0],'line_left_y_conf'].gt(0))==0 and abs(target_df.loc[end_i.index[0],'line_left_y_conf']-target_df.loc[np.round(end_i.index[0]-1/settings['fps'],2),'line_left_y_conf'])<1:
                    end_i = end_i.index[0]
                    event = 'lo'
                    events.append((begin_i, event))
                    follow_lane_switch = (end_i-i)*settings['fps']
        # Left lane change into ego lane: cross right_y down
        elif event!='li' and event!='lo' and target_df.loc[i,'line_right_y_conf']<0<=target_df.loc[prev_i,'line_right_y_conf'] and abs(target_df.loc[i,'line_right_y_conf']-target_df.loc[prev_i,'line_right_y_conf'])<1 and target_df.loc[i,'line_right_y_conf_down']<-0.25 and target_df.loc[i,'line_center_y_conf']!=np.nan and target_df.loc[i,'line_left_y_conf']>target_df.loc[i,'line_right_y_conf']:
            from_y = (1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            goal_y = -(1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            begin_i = np.max((i-6, events[-1][0]))
            begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf')].gt(from_y)]
            if len(begin_i)>0 and sum(target_df.loc[begin_i.index[-1]:np.round(i-1/settings['fps'],2),'line_right_y_conf'].lt(goal_y/2))==0 and (begin_i.index[-1]==target_df.index[0] or abs(target_df.loc[begin_i.index[-1],'line_right_y_conf']-target_df.loc[np.round(begin_i.index[-1]-1/settings['fps'],2),'line_right_y_conf'])<1):
                begin_i = begin_i.index[-1]
                if len(target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf_down')]==0])>0:
                    begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf_down')]==0].index[-1]
                end_i = np.min((i+6, target_df.index[-1]))
                end_i = target_df[i:end_i][target_df.loc[i:end_i,('line_right_y_conf')].lt(goal_y)]
                if len(end_i)>0 and sum(target_df.loc[i:end_i.index[0],'line_right_y_conf'].gt(0))==0 and abs(target_df.loc[end_i.index[0],'line_right_y_conf']-target_df.loc[np.round(end_i.index[0]-1/settings['fps'],2),'line_right_y_conf'])<1:
                    end_i = end_i.index[0]
                    event = 'li'
                    events.append((begin_i, event))
                    follow_lane_switch = (end_i-i)*settings['fps']
        # Right lane change into ego lane: cross left_y up
        elif event!='ri' and event!='ro' and target_df.loc[i,'line_left_y_conf']>0>=target_df.loc[prev_i,'line_left_y_conf'] and abs(target_df.loc[i,'line_left_y_conf']-target_df.loc[prev_i,'line_left_y_conf'])<1 and target_df.loc[i,'line_left_y_conf_up']>0.25 and target_df.loc[i,'line_center_y_conf']!=np.nan and target_df.loc[i,'line_left_y_conf']>target_df.loc[i,'line_right_y_conf']:
            from_y = -(1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            goal_y = (1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            begin_i = np.max((i-6, events[-1][0]))
            begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf')].lt(from_y)]
            if len(begin_i)>0 and sum(target_df.loc[begin_i.index[-1]:np.round(i-1/settings['fps'],2),'line_left_y_conf'].gt(goal_y/2))==0 and (begin_i.index[-1]==target_df.index[0] or abs(target_df.loc[begin_i.index[-1],'line_left_y_conf']-target_df.loc[np.round(begin_i.index[-1]-1/settings['fps'],2),'line_left_y_conf'])<1):
                begin_i = begin_i.index[-1]
                if len(target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf_up')]==0])>0:
                    begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_left_y_conf_up')]==0].index[-1]
                end_i = np.min((i+6, target_df.index[-1]))
                end_i = target_df[i:end_i][target_df.loc[i:end_i,('line_left_y_conf')].gt(goal_y)]
                if len(end_i)>0 and sum(target_df.loc[i:end_i.index[0],'line_left_y_conf'].lt(0))==0 and abs(target_df.loc[end_i.index[0],'line_left_y_conf']-target_df.loc[np.round(end_i.index[0]-1/settings['fps'],2),'line_left_y_conf'])<1:
                    end_i = end_i.index[0]
                    event = 'ri'
                    events.append((begin_i, event))
                    follow_lane_switch = (end_i-i)*settings['fps']
        # Right lane change out of ego lane: cross right_y up
        elif event!='ro' and event!='ri' and target_df.loc[i,'line_right_y_conf']>0>=target_df.loc[prev_i,'line_right_y_conf'] and abs(target_df.loc[i,'line_right_y_conf']-target_df.loc[prev_i,'line_right_y_conf'])<1 and target_df.loc[i,'line_right_y_conf_up']>0.25 and target_df.loc[i,'line_center_y_conf']!=np.nan and target_df.loc[i,'line_left_y_conf']>target_df.loc[i,'line_right_y_conf']:
            from_y = -(1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            goal_y = (1/4)*(target_df.loc[i,'line_left_y_conf']-target_df.loc[i,'line_right_y_conf'])
            begin_i = np.max((i-6, events[-1][0]))
            begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf')].lt(from_y)]
            if len(begin_i)>0 and sum(target_df.loc[begin_i.index[-1]:np.round(i-1/settings['fps'],2),'line_right_y_conf'].gt(goal_y/2))==0 and (begin_i.index[-1]==target_df.index[0] or abs(target_df.loc[begin_i.index[-1],'line_right_y_conf']-target_df.loc[np.round(begin_i.index[-1]-1/settings['fps'],2),'line_right_y_conf'])<1):
                begin_i = begin_i.index[-1]
                if len(target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf_up')]==0])>0:
                    begin_i = target_df[begin_i:i][target_df.loc[begin_i:i,('line_right_y_conf_up')]==0].index[-1]
                end_i = np.min((i+6, target_df.index[-1]))
                end_i = target_df[i:end_i][target_df.loc[i:end_i,('line_right_y_conf')].gt(goal_y)]
                if len(end_i)>0 and sum(target_df.loc[i:end_i.index[0],'line_right_y_conf'].lt(0))==0 and abs(target_df.loc[end_i.index[0],'line_right_y_conf']-target_df.loc[np.round(end_i.index[0]-1/settings['fps'],2),'line_right_y_conf'])<1:
                    end_i = end_i.index[0]
                    event = 'ro'
                    events.append((begin_i, event))
                    follow_lane_switch = (end_i-i)*settings['fps']
        elif event != 'fl' and follow_lane_switch <= 0:
            event = 'fl'
            events.append((i,event))
        prev_i = i

    #if (plot or save_path!='') and len(events)>1:
    #if int(target_df['id'].values[0]) in [848, 2365, 4546, 5118, 5134, 5155, 5220, 5822]:
    if plot:
        fig, axarr = plt.subplots(1,3, sharex=True, figsize=(19.2,6.4))
        fig.suptitle('Lateral target events')
        for event in events:
            event_name = event[1]
            if event_name != 'fl':
                event_name = event_name[0]
            axarr[0].vlines(event[0], -4, 4, linestyles='dashed', colors='grey')
            axarr[1].vlines(event[0], -2, 0.5, linestyles='dashed', colors='grey')
            axarr[2].vlines(event[0], -2, 0.5, linestyles='dashed', colors='grey')
            axarr[0].text(event[0], -4, event_name)
            axarr[1].text(event[0], -2, event_name)
            axarr[2].text(event[0], -2, event_name)
        axarr[0].plot(target_df.index, target_df['line_left_y_conf'], color='blue', label='left line')
        axarr[0].plot(target_df.index, target_df['line_right_y_conf'], color='red', label='right line')
        axarr[0].plot(target_df.index, target_df['line_center_y_conf'], color='magenta', label='center of lane')
        axarr[0].legend()
        axarr[0].plot(target_df.index, 0.25*(target_df['line_left_y']-target_df['line_right_y']), color='black')
        axarr[0].plot(target_df.index, -0.25*(target_df['line_left_y']-target_df['line_right_y']), color='black')
        axarr[0].plot(target_df.index, target_df['line_left_y'], linestyle=':', color='blue')
        axarr[0].plot(target_df.index, target_df['line_right_y'], linestyle=':', color='red')
        axarr[0].plot(target_df.index, target_df['line_center_y'], linestyle=':', color='magenta')
        axarr[0].hlines(y=[0], xmin=target_df.index[0], xmax=target_df.index[-1])
        axarr[0].set_xlabel('time (seconds)')
        axarr[0].set_ylabel('distance (meters)')
        axarr[1].plot(target_df.index, target_df['line_right_y_conf_down'], label='right line decrease in last second')
        axarr[1].hlines(y=[0, -0.25], xmin=target_df.index[0], xmax=target_df.index[-1])
        axarr[1].set_xlabel('time (seconds)')
        axarr[1].set_ylabel('distance (meters)')
        axarr[1].legend()
        axarr[2].plot(target_df.index, target_df['line_left_y_conf_down'], label='left line decrease in last second')
        axarr[2].hlines(y=[0, -0.25], xmin=target_df.index[0], xmax=target_df.index[-1])
        axarr[2].set_xlabel('time (seconds)')
        axarr[2].set_ylabel('distance (meters)')
        axarr[2].legend()
    
    return target_df, events