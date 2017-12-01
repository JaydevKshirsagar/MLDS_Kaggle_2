

max_probable_transition = []
max_probable_transition.append(0)

with open('Trans_Matrix_Matlab_24.csv', 'r') as trans_file:
    for line in trans_file:
        trans = line.split(',')
        max_prob = 0
        next_state = 0
        for j in range(len(trans)):
            val = float(trans[j])
            if val > max_prob:
                max_prob = val
                next_state = j+1
        max_probable_transition.append(next_state)


max_probable_emission = []
max_probable_emission.append(0)

with open('Emis_Matrix_Matlab_24.csv', 'r') as emis_file:
    for line in emis_file:
        emis = line.split(',')
        max_prob = 0
        possible_emission = 0
        for j in range(len(emis)):
            val = float(emis[j])
            if val > max_prob:
                max_prob = val
                possible_emission = j+1
        max_probable_emission.append(possible_emission)


next_state_list = []
next_emission_list = []

with open('STATES_Matlab.csv', 'r') as state_seq_file:
    for line in state_seq_file:
        final_state = int(line.split(',')[-1])
        max_probable_next_state = max_probable_transition[final_state]
        max_probable_next_emission = max_probable_emission[max_probable_next_state]
        next_state_list.append(max_probable_next_state)
        next_emission_list.append(max_probable_next_emission)
        print '%d,%d' %(max_probable_next_state, max_probable_next_emission)

#i = 0
#with open('STATES_Matlab_new.csv', 'w') as state_seq_file_new:
#    with open('STATES_Matlab.csv', 'r') as state_seq_file:
#        for line in state_seq_file:
#            line[-1] = ','
#            line = line + str(next_state_list[i])
#            print line
#            i += 1
            #state_seq_file_new.write(line)

