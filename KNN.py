import pandas as pd
import numpy as np
import operator

output = []
output_index = []
train_output = []
# read training dataset
input = pd.read_csv('train_input.csv')

file = np.load('train_output.npz')
for key in file:
    train_output.append(file[key])

# load test input dataset
test_input = pd.read_csv('test_input.csv')


train_input_s = input['sequence']
train_input_q = input['q8']
test_input_s = test_input['sequence']
test_input_q = test_input['q8']

# append a value for test data cannot find a match(same length) in training data
train_output.append([1])

# find the training sample with the same length
# find the longest common sequence of both q8 and sequence in every training sample with same length
# add them up and then find the maximum
# return the index of the sequence, finally output the corresponding matrix


for b in range(len(test_input_s)):
    # find common sequence in 'sequence'
    T = test_input_s[b]
    long = pd.DataFrame(columns=['common1', 'length1','common2','length2','total_length'])
    for a in range(len(train_input_s)):
        S=train_input_s[a]
        m = len(S)
        n = len(T)
        if m == n:
            counter = [[0] * (n + 1) for x in range(m + 1)]
            longest = 0
            lcs_set = set()
            for i in range(m):
                for j in range(n):
                    if S[i] == T[j]:
                        c = counter[i][j] + 1
                        counter[i + 1][j + 1] = c
                        if c > longest:
                            lcs_set = set()
                            longest = c
                            lcs_set.add(S[i - c + 1:i + 1])
                        #elif c == longest:
                            #lcs_set.add(S[i - c + 1:i + 1])
            print(a, list(lcs_set), len(list(lcs_set)[0]))
            # find the common sequence in q8
            T2 = test_input_q[b]
            S2 = train_input_q[a]
            counter = [[0] * (n + 1) for x in range(m + 1)]
            longest = 0
            lcs_set2 = set()
            for i in range(m):
                for j in range(n):
                    if S2[i] == T2[j]:
                        c = counter[i][j] + 1
                        counter[i + 1][j + 1] = c
                        if c > longest:
                            lcs_set2 = set()
                            longest = c
                            lcs_set2.add(S2[i - c + 1:i + 1])
            long.loc[a] = [list(lcs_set)[0], len(list(lcs_set)[0]), list(lcs_set2)[0],len(list(lcs_set2)[0]),0]

            long['length1'] = long['length1'].astype('float64')
            long['length2'] = long['length2'].astype('float64')
            long['total_length'] = long['length1']+long['length2']
            print(long)
    if len(long)>0:
        sequence_location= long.total_length.idxmax()
    else:
        sequence_location=4554
    output_index.append(sequence_location)
    print(output_index)
for i in output_index:
    output.append(train_output[i])
    print(train_output[i])

#write the output to the npz file
np.savez('output_seperate.npz', *output)
