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


# Labels, None if not yet loaded, otherwise a list of size 10,000 x 4,
# where each entry has the form [r, t, x, y]
_LABELS = None


# Labels of the final 60 points of each run. None if not yet loaded, otherwise
# a list of size 10,000 x 4, where each entry has the form [r, t, x, y].
_LABELS_60 = None


# File name of the observations file
OBSERVATIONS_FILE = 'Observations.csv'


# The final 60 observations of all runs
OBSERVATIONS_FINAL_60 = 'obs-60.csv'


# The final 500 observations of all runs
OBSERVATIONS_FINAL_500 = 'obs-500.csv'


# The labels corresponding to the final 60 observations, without repeats
LABELS_ADJUSTED_60 = 'labels-adj-60.csv'


# The labels corresponding to the final 500 observations, without repeats
LABELS_ADJUSTED_500 = 'labels-adj-500.csv'


# File of seemingly good results
GOOD_STATES = 'good-STATES_Matlab-24.csv'


# Labels [r, t, x, y, theta, state] with r and t indexed by 1
ANTOINE_LABEL = 'antoine-labels.csv'


# Labels [r, t, x+1.5, y+1.5, theta, state] with r and t
VISH_LABEL = 'vish-labels.csv'


def load_obs():
    """
    Returns the observations from OBSERVATIONS_FILE as a numpy array.

    There are 10,000 runs, each consisting of 1,000 timesteps.

    Returns:
        2d np array of runs and timesteps, 10000 x 1000.
    """
    return np.genfromtxt(OBSERVATIONS_FILE, delimiter=',')


def load_obs_tail():
    """
    Returns the tails of the observations.

    There are 10,000 runs, each of 1,000 time steps. This function returns the
    final 60 time steps of each run.

    Returns:
        2d np array of runs and time steps, 10,000 x 60.
    """
    return np.genfromtxt(OBSERVATIONS_FINAL_60, delimiter=',')


def load_obs_500():
    return np.genfromtxt(OBSERVATIONS_FINAL_500, delimiter=',')


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



def load_tail_labels():
    """
    Returns the labels of the final 60 points in each run.

    Returns:
        A list of lists, where each inner list has length 4 and store
        the information [r, t, x, y], where r and t are 0 indexed and
        the coordinates x and y have been shifted by 1.5 to lie in the middle.
    """
    global _LABELS_60 # need to modify to save loaded labels
    # if labels have already been loaded, just return, don't load again
    if _LABELS_60:
        return _LABELS_60

    result = []
    with open(LABELS_ADJUSTED_60) as label_file:
        for line in label_file:
            if line:
                entries = line.strip().split(',')
                entries[0] = int(entries[0])
                entries[1] = int(entries[1])
                entries[2] = float(entries[2])
                entries[3] = float(entries[3])
                result.append(entries)
    _LABELS_60 = result # save loaded labels
    return result


def load_labels_500():
    result = []
    with open(LABELS_ADJUSTED_500) as label_file:
        for line in label_file:
            if line:
                entries = line.strip().split(',')
                entries[0] = int(entries[0])
                entries[1] = int(entries[1])
                entries[2] = float(entries[2])
                entries[3] = float(entries[3])
                result.append(entries)
    return result


def create_final_60():
    """
    Creates the files of each run's final 60 observations and labels.
    """
    obs = load_obs()
    np.savetxt(OBSERVATIONS_FINAL_60, obs[:, -60:], delimiter=',')

    labels = load_labels_adj()
    new_labels = []
    for entry in labels:
        time = entry[1]
        if time >= 940 and (len(labels) == 0 or labels[-1][1] != time):
            new_labels.append([str(x) for x in entry])
    with open(LABELS_ADJUSTED_60, 'w') as output_file:
        for line in new_labels:
            output_file.write(','.join(line) + '\n')


def create_final_500():
    obs = load_obs()
    np.savetxt(OBSERVATIONS_FINAL_500, obs[:, -500:], delimiter=',')

    labels = load_labels_adj()
    new_labels = []
    for entry in labels:
        time = entry[1]
        if time >= 500 and (len(labels) == 0 or labels[-1][1] != time):
            new_labels.append([str(x) for x in entry])
    with open(LABELS_ADJUSTED_500, 'w') as output_file:
        for line in new_labels:
            output_file.write(','.join(line) + '\n')



# https://stackoverflow.com/questions/23861680/convert-spreadsheet-number-to-column-letter
def col_to_letters(col_num):
    """
    Returns the string representing the column given by col_num.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
        fname: The filename to which to save the matrix.
        num_states: The number of states, num_states >= 4.
        prob_state: The probability of staying in the same state, in [0.0, 1.0].
        prob_move: The probability of moving to the next state, in [0.0, 1.0].
    """
    np.savetxt(fname, simple_transition(num_states, prob_stay, prob_move),
               delimiter = ',')


def print_tail_angle_info():
    """
    Prints information about the final 60 runs.
    """
    # obs = load_obs_tail()
    obs = load_obs_500()
    min_theta = obs.min() # 0.14762
    max_theta = obs.max() # 1.4168
    print('min angle: {}'.format(min_theta))
    print('max_angle: {}'.format(max_theta))


def plot_angles_on_tail(num_sections):
    """
    Draws sections between the min and max theta values.
    Args:
        num_sections: The number of sections, positive.
    """
    labels = load_tail_labels()
    # get angles for colors
    obs_nums = calculate_obs_num(num_sections)
    colors = [] # calculate color based on label's angle
    x_coords = []
    y_coords = []
    for entry in labels:
        x_coords.append(entry[2])
        y_coords.append(entry[3])
        class_num = obs_nums[entry[0], entry[1] - 940]
        colors.append(colorsys.hsv_to_rgb(class_num / 360.0, 0.9, 0.9))
    plt.scatter(x_coords, y_coords, marker='.', s=0.1, c=colors)
    circle = plt.Circle((1.5, 1.5), 1.0, fill=False, color='b')
    axis = plt.gca()
    axis.add_artist(circle)

    min_theta = 0.14762
    max_theta = 1.4168
    total_range = max_theta - min_theta
    step = total_range / num_sections
    length = 5.0 # length of lines, simply make longer than the area displayed
    for i in range(num_sections + 1):
        angle = min_theta + i * step
        plt.plot([0.0, length * math.cos(angle)],
                 [0.0, length * math.sin(angle)],
                 linewidth=0.25, c='k')

    centers = calc_initial_state_centers(100) # TODO make parameter or load
    plt.scatter(centers[:, 0], centers[:, 1])

    plt.axis([0.0, 3.0, 0.0, 3.0])
    plt.show()


def calculate_obs_num(num_sections):
    """
    Calculates a section number (observatio number) for each observation angle.

    Args:
        num_sections: The number of sections to place angles, positive.
    Returns:
        A 10,000 x 60 array of the tail converted to integer numbers in
        the inclusive range [1, num_sections].
    """
    # obs_angles = load_obs_tail()
    obs_angles = load_obs_500()
    obs_classes = np.zeros_like(obs_angles, dtype=np.int)
    # min_theta = 0.14762 # for final 60
    # max_theta = 1.4168 # for final 60
    min_theta = 0.12031
    max_theta = 1.4415
    total_range = max_theta - min_theta
    step = total_range / num_sections

    num_runs, num_time_steps = obs_angles.shape
    for i in range(num_runs):
        for j in range(num_time_steps):
            angle = obs_angles[i, j]
            # find the angle's observation number (maybe could use mod here)
            index = 0
            total = min_theta
            while total <= angle:
                index += 1
                total += step
            obs_classes[i, j] = min(index, num_sections)
    assert np.min(obs_classes) >= 1
    assert np.max(obs_classes) <= num_sections
    return obs_classes


def assign_obs_num(num_sections, fname):
    """
    Writes to fname an assignment of angles to an integer observation number.

    Args:
        num_sections: The number of sections to place angles, positive.
        fname: Name of the file which to save.
    """
    np.savetxt(fname, calculate_obs_num(num_sections), delimiter=',', fmt='%i')


def calc_initial_state_centers(num_states):
    """
    Returns the initial centroids of num_states.

    Args:
        num_states: The number of states, num_states >= 8, divisible by 4.
    Returns:
        2d np array num_states x 2 that is the x and y coordinates of
        each of the state initial centroids.
    """
    assert num_states >= 8, 'Need at least 8 states, had {}'.format(num_states)
    assert num_states % 4 == 0, 'Must be divisible by 4'
    values = np.zeros((num_states, 2))
    halfpi = math.pi / 2.0
    quarter_num_states = num_states // 4
    index = 0
    nudge = math.pi / 36 # hack ;)
    for i in range(quarter_num_states):
        t = i / (quarter_num_states - 1)
        dist = center_radius(t)
        # t -= nudge / (quarter_num_states)
        values[index, 0] = dist * math.cos(nudge + t * halfpi) + 1.5
        values[index, 1] = dist * math.sin(nudge + t * halfpi) + 1.5
        index += 1
    for i in range(quarter_num_states):
        t = i / (quarter_num_states - 1)
        dist = center_radius(t)
        values[index, 0] = dist * math.cos(nudge + halfpi + t * halfpi) + 1.5
        values[index, 1] = dist * math.sin(nudge + halfpi + t * halfpi) + 1.5
        index += 1
    for i in range(quarter_num_states):
        t = i / (quarter_num_states - 1)
        dist = center_radius(t)
        values[index, 0] = dist * math.cos(nudge + math.pi + t * halfpi) + 1.5
        values[index, 1] = dist * math.sin(nudge + math.pi + t * halfpi) + 1.5
        index += 1
    for i in range(quarter_num_states):
        t = i / (quarter_num_states - 1)
        dist = center_radius(t)
        values[index, 0] = dist * math.cos(nudge + 3.0 * halfpi + t * halfpi) + 1.5
        values[index, 1] = dist * math.sin(nudge + 3.0 * halfpi + t * halfpi) + 1.5
        index += 1
    return values


def center_radius(t):
    """
    Calculates the distance of the centroid cluster given a t parameter.

    Args:
        t: Parameter in [0.0, 1.0].
    Returns:
        The distance from the center given parameter value t, in [0.9, 1.1].
    """
    return 0.2 * t + 0.9


def init_matrices(num_states, num_observations, prob_stay, prob_move):
    """
    Initializes the transition and emission initial matrices.

    Args:
        num_states: at least 8, multiple of 4
        num_observations: positive
        prob_stay: In [0.0, 1.0], sums to 1.0 with prob_move.
        prob_move: In [0.0, 1.0], sums to 1.0 with prob_stay.
    """
    # labels = load_tail_labels()
    # labels = load_labels_500()
    # obs_nums = calculate_obs_num(num_observations)
    # centers = calc_initial_state_centers(num_states)

    # counts = np.zeros(num_states, dtype=np.int) # number of points in each state
    # emission = np.zeros((num_states, num_observations))
    # for entry in labels:
    #     x_coord = entry[2]
    #     y_coord = entry[3]
    #     min_dist = math.inf
    #     min_ind = -1
    #     for i in range(num_states):
    #         dist = distance(x_coord, y_coord, centers[i, 0], centers[i, 1])
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_ind = i
    #     # class_num = obs_nums[entry[0], entry[1] - 940]
    #     class_num = obs_nums[entry[0], entry[1] - 500]
    #     counts[min_ind] += 1
    #     emission[min_ind, class_num - 1] += 1 # make class number 0 indexed

    # # normalize emissions
    # for i in range(num_states):
    #     emission[i, :] /= counts[i]

    save_simple_transition(num_states, prob_stay, prob_move,
                           'init-transition.csv')
    # np.savetxt('init-emission.csv', emission, delimiter=',')
    # assign_obs_num(num_observations, 'init-observation-classes.csv')


def distance(x1, y1, x2, y2):
    """Returns the distance between (x1, y1) and (x2, y2)."""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def state_plot():
    """
    Plots the states Vish sent to me.
    """
    labels = load_tail_labels()
    state_array = np.genfromtxt('STATES_Matlab.csv', delimiter=',')
    x_coords = []
    y_coords = []
    colors = []
    for entry in labels:
        x_coords.append(entry[2])
        y_coords.append(entry[3])
        state = state_array[entry[0], entry[1] - 940]
        colors.append(colorsys.hsv_to_rgb(state / 360.0, 0.9, 0.9))
    plt.scatter(x_coords, y_coords, c=colors, marker='.', s=0.1)
    plt.show()


def load_good_states():
    """
    Returns a numpy array representing the good states file.

    Returns:
        A 10,000 x 800 np array representing the assignment of state to point.
        The r, t entry represents run r's step t+200, with t starting at 0.
    """
    return np.genfromtxt(GOOD_STATES, delimiter=',')


def make_antoine_file():
    """
    Makes the label file for antonie.
    """
    good_states = load_good_states()
    all_obs = load_obs()
    labels = []
    with open('Label') as label_file:
        for line in label_file:
            if line:
                entries = line.strip().split(',')
                run = int(entries[0])
                entries[0] = run
                time = int(entries[1])
                entries[1] = time
                entries[2] = float(entries[2])
                entries[3] = float(entries[3])
                angle = all_obs[run - 1, time - 1]
                entries.append(angle)
                state = good_states[run - 1, time - 201]
                entries.append(int(state))
                labels.append(entries)
    with open(ANTOINE_LABEL, 'w') as output_file:
        for label in labels:
            output_file.write(','.join(str(x) for x in label) + '\n')


def make_vish_file():
    """
    Makes the label file for vish.
    """
    good_states = load_good_states()
    all_obs = load_obs()
    labels = []
    with open('Label') as label_file:
        for line in label_file:
            if line:
                entries = line.strip().split(',')
                run = int(entries[0])
                entries[0] = run
                time = int(entries[1])
                entries[1] = time
                entries[2] = float(entries[2]) + 1.5
                entries[3] = float(entries[3]) + 1.5
                angle = all_obs[run - 1, time - 1]
                entries.append(angle)
                state = good_states[run - 1, time - 201]
                entries.append(int(state))
                labels.append(entries)
    labels = sorted(labels, key=lambda x: (int(x[0]), int(x[1])))
    with open(VISH_LABEL, 'w') as output_file:
        for label in labels:
            output_file.write(','.join(str(x) for x in label) + '\n')


def proj(x, y, angle):
    """
    Returns the projection of (x, y) onto the line determined by angle.

    Args:
        x: x coordinate of the point to project.
        y: y coordinate of the point to project.
        angle: The angle the projection line makes to the x axis, in (0, pi/2).
    Returns:
        A tuple (x', y') that is the projection of (x, y) onto the line
        through the origin at an angle of 'angle' from the x axis.
    """
    to_proj = np.array([x, y])
    proj_line = np.array([math.cos(angle), math.sin(angle)])
    scalar = to_proj.dot(proj_line) / proj_line.dot(proj_line)
    projection = scalar * proj_line
    return projection[0], projection[1]


def main(args):
    """
    Runs the script.

    Args:
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
    elif args[0] == 'create_final_60':
        create_final_60()
    elif args[0] == 'tail_info':
        print_tail_angle_info()
    elif args[0] == 'plot_tail_segments':
        plot_angles_on_tail(int(args[1]))
    elif args[0] == 'assign_obs_num':
        num_observations = int(args[1])
        fname = args[2]
        assign_obs_num(num_observations, fname)
    elif args[0] == 'calc_centers':
        print(calc_initial_state_centers(int(args[1])))
    elif args[0] == 'init_matrices':
        num_states = int(args[1])
        num_observations = int(args[2])
        prob_stay = float(args[3])
        prob_move = float(args[4])
        init_matrices(num_states, num_observations, prob_stay, prob_move)
    elif args[0] == 'state_plot':
        state_plot()
    elif args[0] == 'create_500':
        create_final_500()
    elif args[0] == 'make_antoine_file':
        make_antoine_file()
    elif args[0] == 'make_vish_file':
        make_vish_file()
    else:
        print("No valid command entered.")


if __name__ == '__main__':
    main(sys.argv[1:]) # strip off the script name
