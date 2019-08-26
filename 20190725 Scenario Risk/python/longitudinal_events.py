"""
Small description.
"""

__author__ = 'Jeroen Manders'

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

def get(settings, data_df, plot=False):
    ##########
    # FEATURES
    ##########
    # Acceleration in the next second (value of t+1 second - minimum between t and t+1 seconds)
    data_df.loc[:,'odom_vx_up'] = data_df.loc[:,'odom_vx'].shift(-settings['fps']) - data_df.loc[:,'odom_vx'].shift(-settings['fps']).rolling(window=settings['fps']).min()
    # _start removes values from speed_a where t != minimum between t and t+1
    data_df.loc[:,'odom_vx_up_start'] = data_df.loc[:,'odom_vx_up'].copy()
    data_df.loc[data_df.loc[:,'odom_vx']!=data_df.loc[:,'odom_vx'].shift(-settings['fps']).rolling(window=settings['fps']).min(),'odom_vx_up_start'] = 0
    
    # Decceleration in the next second (value of t+1 second - maximum between t and t+1 seconds)
    data_df.loc[:,'odom_vx_down'] = data_df.loc[:,'odom_vx'].shift(-settings['fps']) - data_df.loc[:,'odom_vx'].shift(-settings['fps']).rolling(window=settings['fps']).max()
    # _start removes values from speed_a where t != maximum between t and t+1
    data_df.loc[:,'odom_vx_down_start'] = data_df.loc[:,'odom_vx_down'].copy()
    data_df.loc[data_df.loc[:,'odom_vx'] != data_df.loc[:,'odom_vx'].shift(-settings['fps']).rolling(window=settings['fps']).max(),'odom_vx_down_start'] = 0
    
    # For TTC1/2/3 and THW1/2/3:
    data_df['odom_vx_savgol'] = savgol_filter(data_df['odom_vx'],settings['fps']+1, 0)/3.6
    data_df['odom_vx_savgol_acc'] = savgol_filter((data_df['odom_vx_savgol'] - data_df['odom_vx_savgol'].shift(1))*settings['fps'], 2*settings['fps']+1, 0)
    data_df['odom_vx_savgol_jerk'] = savgol_filter((data_df['odom_vx_savgol_acc'] - data_df['odom_vx_savgol_acc'].shift(1))*settings['fps'], 2*settings['fps']+1, 0)
    
    ##########
    # ALGORITHM
    ##########
    all_events = [(data_df.index[0], 'c')]
    event = 'c'
    cruise_switch = 0
    for i in data_df.index:
        cruise_switch -= 1
        # Potential acceleration signal when in minimum wrt next second, acceleration and not standing still
        if event != 'a' and data_df.loc[i,'odom_vx_up_start']>=0.5 and data_df.loc[i,'odom_vx']>0.25:
            end_i = np.min((i+300, data_df.index[-1]))
            if len(data_df.loc[i:end_i,'odom_vx_up'].loc[data_df.loc[i:end_i,'odom_vx_up'].lt(0.25)]) > 0:
                end_i = data_df.loc[i:end_i,'odom_vx_up'].loc[data_df.loc[i:end_i,'odom_vx_up'].lt(0.25)].index[0]
            # OR: Definitive accelerate signal when: (before we fall below 50% of start acceleration (0.25))
            # - minimum of 4km/u speed increase
            if np.sum(data_df.loc[i:end_i,'odom_vx_up'])/settings['fps']>=4:
                event = 'a'
                all_events.append((i,event))
                cruise_switch = (end_i-i)*settings['fps']
        # Deccelerate (inverse of accelerate)
        elif event != 'd' and data_df.loc[i,'odom_vx_down_start']<=-0.5:
            end_i = np.min((i+300, data_df.index[-1]))
            if len(data_df.loc[i:end_i,'odom_vx_down'].loc[data_df.loc[i:end_i,'odom_vx_down'].gt(-0.25)]) > 0:
                end_i = data_df.loc[i:end_i,'odom_vx_down'].loc[data_df.loc[i:end_i,'odom_vx_down'].gt(-0.25)].index[0]
            if np.sum(data_df.loc[i:end_i,'odom_vx_down'])/settings['fps']<=-4:
                event = 'd'
                all_events.append((i,event))
                cruise_switch = (end_i-i)*settings['fps']
        # Cruise when not accelerating/deccelerating
        elif event != 'c' and cruise_switch <= 0:
            event = 'c'
            all_events.append((i,event))
            
    # Remove small cruise activities (<4sec)     
    events = []
    i = 0
    while i < len(all_events):
        if i == 0 or i == len(all_events)-1 or all_events[i][1]=='a' or all_events[i][1]=='d' or (all_events[i+1][0]-all_events[i][0]) >= 4:
            events.append(all_events[i])
            i += 1
        elif all_events[i-1][1] == all_events[i+1][1]:
            i += 2
        elif all_events[i+1][1]=='a':
            events.append((data_df.loc[all_events[i][0]:all_events[i+1][0],'odom_vx'].eq(min(data_df.loc[all_events[i][0]:all_events[i+1][0],'odom_vx']))[::-1].idxmax(),'a'))
            i+= 2
        else:
            events.append((data_df.loc[all_events[i][0]:all_events[i+1][0],'odom_vx'].eq(max(data_df.loc[all_events[i][0]:all_events[i+1][0],'odom_vx']))[::-1].idxmax(),'d'))
            i+= 2
    
    if plot:   
        fig, axarr = plt.subplots(1,3, sharex=True, figsize=(19.2,6.4))
        fig.suptitle('Longitudinal events')
        for event in events:
            axarr[0].vlines(event[0], -2, 120, linestyles='dashed', colors='grey')
            axarr[1].vlines(event[0], -0.5, 6, linestyles='dashed', colors='grey')
            axarr[2].vlines(event[0], -6, 0.5, linestyles='dashed', colors='grey')
            axarr[0].text(event[0], -2, event[1])
            axarr[1].text(event[0], -0.5, event[1])
            axarr[2].text(event[0], -6, event[1])
        axarr[0].plot(data_df.index, data_df.loc[:,'odom_vx'], label='vehicle speed')
        axarr[0].hlines(y=[0], xmin=data_df.index[0], xmax=data_df.index[-1])
        axarr[0].set_ylabel('vehicle speed (kph)')
        axarr[0].set_xlabel('time (seconds)')
        axarr[0].legend()
        axarr[1].plot(data_df.index, data_df.loc[:,'odom_vx_up'], label='acceleration within next second')
        axarr[1].hlines(y=[0, 0.5], xmin=data_df.index[0], xmax=data_df.index[-1])
        axarr[1].set_ylabel('speed change within next second (kph)')
        axarr[1].set_xlabel('time (seconds)')
        axarr[1].legend()
        axarr[2].plot(data_df.index, data_df.loc[:,'odom_vx_down'], label='deceleration within next second')
        axarr[2].hlines(y=[0, -0.5], xmin=data_df.index[0], xmax=data_df.index[-1])
        axarr[2].set_ylabel('speed change within next second (kph)')
        axarr[2].set_xlabel('time (seconds)')
        axarr[2].legend()        
        
    return data_df, events

def get_ego(settings, ego_df, plot=False):
    return get(settings, ego_df, plot)

def get_target(settings, target_df, plot=False):
    return get(settings, target_df, plot)