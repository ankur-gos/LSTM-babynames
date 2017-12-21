'''
lstm.py
Basic lstm implementation
Ankur Goswmai, agoswam3@ucsc.edu
'''

import numpy as np

def sigmoid(value):
    exp = np.e ** value
    return exp / (1 + exp)

sigmoid_vec = np.vectorize(sigmoid)

def encode_input(letter):
    # convert to ascii
    x = ord(letter)    
    # input must be within bounds
    if(x < 65 or x > 122)
        assert("input is not an accepted letter")
    arr = np.zeros(58)
    arr[x - 65] = 0
    return arr

def alphabetize_output(number):
    return chr(number)

cell_state = np.zeros(256)
current_output = np.zeros(58)
inp = 'bobby'
bias = np.ones(1)
current_input = inp[0]
forget_gate_weights = np.zeros(shape=(117, 256))
input_sigmoid_gate_weights = np.zeros(shape(117, 256))
input_tanh_gate_weights = np.zeros(shape(117, 256))
output_sigmoid_gate_weights = np.zeros(shape(117, 256))
for ch in inp[1:]:
    forget_gate_input = np.concatenate(current_output, current_input, bias)
    forget_gate_dot = np.dot(forget_gate_input, forget_gate_weights)

    forget_gate_output = sigmoid_vec(forget_gate_dot)

    cell_state = np.multiply(cell_state, forget_gate_output)
    input_sigmoid_gate_dot = np.dot(forget_gate_input, input_sigmoid_gate_weights)
    input_sigmoid_gate_output = sigmoid_vec(input_sigmoid_gate_dot)
    input_tanh_gate_dot = np.dot(forget_gate_input, input_tanh_gate_weights)
    input_tanh_gate_output = np.tanh(input_tanh_gate_dot)

    merge_inputs = np.multiply(input_tanh_gate_output, input_sigmoid_gate_output)
    cell_state = np.add(cell_state, merge_inputs)

    cell_state_output_gate = np.tanh(cell_state)

    output_sigmoid_gate_dot = np.dot(forget_gate_input, output_sigmoid_gate_weights)
    output_sigmoid_gate_output = sigmoid_vec(output_sigmoid_gate_dot)
    cell_state_output_gate = np.multiply(cell_state_output_gate, output_sigmoid_gate_output)


input_layer =