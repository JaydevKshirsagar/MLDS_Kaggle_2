#!/usr/bin/env python
"""
Code for competition 2.

The competition involves attempting to find the location of a robot.
"""


import sys
import math
import colorsys
import matplotlib.pyplot as plt
import numpy as np


# File name of the label file
LABEL_FILE = 'Label'


# File name of the adjusted labels
LABEL_ADJUSTED_FILE = 'labels-adjusted.csv'


# Labels, none if not yet loaded, otherwise a list of size 10,000 x 4,
# where each entry has the form [r, t, x, y]
_LABELS = None


# File name of the observations file
OBSERVATIONS_FILE = 'Observations.csv'


def load_obs():
    """
    Returns the observations from OBSERVATIONS_FILE as a numpy array.

    There are 10,000 runs, each consisting of 1,000 timesteps.

    Returns:
        2d np array of runs and timesteps, 10000 x 1000.
    """
    return np.genfromtxt(OBSERVATIONS_FILE, delimiter=',')


def load_labels_adj():
    """
    Returns the adjusted labels as a list of lists.

    Returns:
        A list of size 10,000 x 4, where each entry has the form [r, t, x, y].
    """
    global _LABELS # need to modify to save loaded labels
    # labels already loaded, just return, don't load again
    if _LABELS:
        return _LABELS

    result = []
    with open(LABEL_ADJUSTED_FILE) as label_file:
        for line in label_file:
            if line:
                entries = line.strip().split(',')
                entries[0] = int(entries[0])
                entries[1] = int(entries[1])
                entries[2] = float(entries[2])
                entries[3] = float(entries[3])
                result.append(entries)
    _LABELS = result # save loaded labels
    return result


# https://stackoverflow.com/questions/23861680/convert-spreadsheet-number-to-column-letter
def col_to_letters(col_num):
    """
    Returns the string representing the column given by col_num.

    Parameters:
        col_num: The number of the column to find, nonnegative, 0 indexed.
    Returns:
        The string representing col_num.
    """
    string = ''
    num = col_num + 1 # adjust to be 0 indexed
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        string = chr(65 + remainder) + string
    return string


def print_observer_calc_angles():
    """
    Prints the angles corresponding to run 1, timesteps 224 and 317.and

    With 0 indexed notation, these are (0, 223) and (0, 316).
    """
    obs = load_obs()
    print('(0, 213): {}, (0, 316): {}'.format(
        theta(obs, 0, 213), theta(obs, 0, 316)))


def print_angle(run, time):
    """
    Prints the angle corresponding to run's timestep time.
    """
    obs = load_obs()
    print('({}, {}): {}'.format(run, time, theta(obs, run, time)))


def theta(obs, run, time):
    """
    Returns the angle of run's timestep t.

    Parameters:
        obs: 2d np array of runs and timesteps, 10000 x 1000.
            Each entry is the angle theta made at the run's time step.
        run: Run number, 0 <= run < 10,000.
        time: Time step, 0 <= time < 1,000.
    Returns:
        The angle theta made with the x-axis in run' timestep time.
    """
    return obs[run, time]


def adjust_labels():
    """
    Saves in LABEL_ADJUSTED_FILE a 0-indexed and sorted version of LABEL_FILE.

    LABEL_FILE consists of 600,000 rows, then a final blank line,
    of data in the form "run, timestep, x, y".
    The run and timestep data columns are 1 indexed.
    This saves the data "run - 1, timestep - 1, x, y".

    Further, time steps are not sorted per run in the given label file.
    The output file is sorted by run, then by time step in each run.
    """
    # TODO clean repeat labels?
    run_ind = 0
    t_ind = 1
    x_ind = 2
    y_ind = 3
    with open(LABEL_FILE) as src_file:
        lines = [line for line in src_file if line]
    split_data = [line.strip().split(',') for line in lines]
    entries = sorted(split_data, key=lambda x: (int(x[run_ind]), int(x[t_ind])))
    for entry in entries:
        entry[run_ind] = str(int(entry[run_ind]) - 1)
        entry[t_ind] = str(int(entry[t_ind]) - 1)
        entry[x_ind] = str(float(entry[x_ind]) + 1.5)
        entry[y_ind] = str(float(entry[y_ind]) + 1.5)
    with open(LABEL_ADJUSTED_FILE, 'w') as output_file:
        for line in entries:
            output_file.write(','.join(line) + '\n')


def print_max_error():
    """
    Prints the max error in calculating the angles from the adjusted labels.
    """
    obs = load_obs()
    labels = load_labels_adj()
    max_error = 0.0
    for entry in labels:
        run, time, x_coord, y_coord = entry
        angle = theta(obs, run, time)
        error = abs(angle - math.atan2(y_coord, x_coord))
        max_error = max(error, max_error)
    print(max_error) # 8.35016810334e-05


def plot_locations(run):
    """
    Plots the labeled locations for the given run.

    Parameters:
        run: The run to plot, 0 <= run < 6,000.
    """
    all_labels = load_labels_adj()
    run_labels = [entry for entry in all_labels if entry[0] == run]
    x_coords = []
    y_coords = []
    for entry in run_labels:
        x_coords.append(entry[2])
        y_coords.append(entry[3])
    # colors = []
    npts = len(x_coords)
    step = 170.0 / npts
    for i in range(npts):
        # colors.append((i * step, 0.0, 0.0))
        plt.plot(x_coords[i], y_coords[i],
                 c=colorsys.hsv_to_rgb(170 - i * step, 0.9, 0.9), marker='o',
                 linestyle='None')
    # plt.plot(x_coords, y_coords, c='r', marker='.', linestyle='None')
    circle = plt.Circle((1.5, 1.5), 1.0, fill=False, color='b')
    axis = plt.gca()
    axis.add_artist(circle)
    plt.axis('equal')
    plt.show()


def plot_all_locations():
    """
    Plots the labeled locations for all runs.
    """
    labels = load_labels_adj()
    x_coords = []
    y_coords = []
    for entry in labels:
        x_coords.append(entry[2])
        y_coords.append(entry[3])
    plt.plot(x_coords, y_coords, 'r.', ms=0.1)
    circle = plt.Circle((1.5, 1.5), 1.0, fill=False, color='b')
    axis = plt.gca()
    axis.add_artist(circle)
    plt.axis('equal')
    plt.show()


def plot_final_locations():
    """
    Plots the labeled locations for the final 60 points of all runs.
    """
    labels = load_labels_adj()
    x_coords = []
    y_coords = []
    for entry in labels:
        if entry[1] >= 999 - 60:
            x_coords.append(entry[2])
            y_coords.append(entry[3])
    plt.plot(x_coords, y_coords, 'r.', ms=0.1)
    circle = plt.Circle((1.5, 1.5), 1.0, fill=False, color='b')
    axis = plt.gca()
    axis.add_artist(circle)
    plt.axis('equal')
    plt.show()


def plot_angles(run):
    """
    Plots the angle as a function of time step for the given run.

    Parameters:
        run: The run to plot, 0 <= run < 6,000.
    """
    obs = load_obs()
    time = np.arange(1000)
    plt.plot(time, obs[run, :])
    plt.show()


def avg_radius():
    """
    Returns the average distance of all labeled points to the observer.

    Returns:
        The average distance between each labeled point and the observer
        located at (1.5, 1.5).
    """
    labels = load_labels_adj()
    total = 0.0
    for entry in labels:
        x_coord = entry[2]
        y_coord = entry[3]
        dist = math.sqrt((x_coord - 1.5)**2 + (y_coord - 1.5)**2)
        total += dist
    return total / len(labels)


def max_labelled_distance():
    """
    Returns the maximum distance of all labeled points to the observer.

    Returns:
        The maximum distance of all labeled points between the point and the
        observer located at (1.5, 1.5).
    """
    labels = load_labels_adj()
    max_dist = 0.0
    for entry in labels:
        x_coord = entry[2]
        y_coord = entry[3]
        dist = math.sqrt((x_coord - 1.5)**2 + (y_coord - 1.5)**2)
        max_dist = max(max_dist, dist)
    return max_dist


def min_labelled_distance():
    """
    Returns the minimum distance of all labeled points to the observer.

    Returns:
        The minimum distance of all labeled points between the point and the
        observer located at (1.5, 1.5).
    """
    labels = load_labels_adj()
    min_dist = math.inf
    for entry in labels:
        x_coord = entry[2]
        y_coord = entry[3]
        dist = math.sqrt((x_coord - 1.5)**2 + (y_coord - 1.5)**2)
        min_dist = min(min_dist, dist)
    return min_dist


def min_label_time():
    """
    Returns the label with the lowest time step.
    """
    labels = load_labels_adj()
    min_time = math.inf
    for entry in labels:
        min_time = min(min_time, entry[1])
    return min_time


def hsv_to_rgb(hue, sat, val):
    """
    Returns the [0, 1] rgb version of hsv.

    Parameters:
        hue: The hue, in [0.0, 360.0).
        sat: The saturation, in [].
        val: The value, in .
    Returns:
        A tuple (r, g, b), with each value in [0.0, 1.0], representing the
        value of (h, s, v).
    """
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
    return red / 255.0, green / 255.0, blue / 255.0


def simple_transition(num_states, prob_stay, prob_move):
    """
    Returns an np array representing a num_states x num_states matrix.

    Parameters:
        num_states: The number of states, num_states >= 4.
        prob_state: The probability of staying in the same state, in [0.0, 1.0].
        prob_move: The probability of moving to the next state, in [0.0, 1.0].
    Returns:
        A num_states x num_states matrix. Each state has a probability of
        prob_stay of staying in the same state and a probability of prob_move
        of advancing to the next state.
    """
    mat = np.zeros((num_states, num_states))
    for i in range(num_states):
        mat[i, i] = prob_stay
        mat[i, (i + 1) % num_states] = prob_move
    return mat


def save_simple_transition(num_states, prob_stay, prob_move, fname):
    """
    Saves a simple transition matrix to fname.

    Parameters:
        fname: The filename to which to save the matrix.
        num_states: The number of states, num_states >= 4.
        prob_state: The probability of staying in the same state, in [0.0, 1.0].
        prob_move: The probability of moving to the next state, in [0.0, 1.0].
    """
    np.savetxt(fname, simple_transition(num_states, prob_stay, prob_move),
        delimiter = ',')


def main(args):
    """
    Runs the script.

    Parameters:
        args: Determines what test to run.
    """
    if not args or args[0] == 'help':
        print("Enter a command (see script's documentation).")
    elif args[0] == 'obs_calc':
        print_observer_calc_angles()
    elif args[0] == 'angle_of':
        print_angle(int(args[1]), int(args[2]))
    elif args[0] == 'column_of':
        print(col_to_letters(int(args[1])))
    elif args[0] == 'adjust_labels':
        adjust_labels()
    elif args[0] == 'max_error':
        print_max_error()
    elif args[0] == 'plot_locations':
        if args[1] == 'all':
            plot_all_locations()
        elif args[1] == 'final':
            plot_final_locations()
        else:
            plot_locations(int(args[1]))
    elif args[0] == 'plot_angles':
        plot_angles(int(args[1]))
    elif args[0] == 'print_label_distances':
        print('average radius: {}'.format(avg_radius()))
        print('min distance: {}'.format(min_labelled_distance()))
        print('max distance: {}'.format(max_labelled_distance()))
    elif args[0] == 'min_time':
        print(min_label_time())
    elif args[0] == 'simple_transition':
        num_states = int(args[1])
        prob_stay = float(args[2])
        prob_move = float(args[3])
        if len(args) == 4:
            print(simple_transition(num_states, prob_stay, prob_move))
        else:
            fname = args[4]
            save_simple_transition(num_states, prob_stay, prob_move, fname)
    else:
        print("No valid command entered.")


if __name__ == '__main__':
    main(sys.argv[1:]) # strip off the script name
