import ast
import sys
sys.path.append('/home/hd/TransformerPrograms')
import time
import os
import json
import numpy as np
import coverage
import trace
from output.dyck1 import dyck1

def reverse_func(s: str) -> str:
    return s[::-1]

def hist_func(s: str) -> str:
    return ''.join([str(s.count(char)) for char in s])

def hist2_func(s: str) -> str:
    # Calculate the histogram for the string
    histogram = {char: s.count(char) for char in set(s)}
    
    # For each character in the string, count the number of unique characters 
    # that have the same histogram value.
    result = [str(sum(1 for _, count in histogram.items() if count == s.count(char))) for char in s]
    
    return ''.join(result)

def most_freq_func(s: str) -> str:
    # Step 1: Create a frequency dictionary
    freq_dict = {}
    for char in s:
        freq_dict[char] = freq_dict.get(char, 0) + 1

    # Step 2: Sort the dictionary based on frequency. If frequency is the same, then position in the string is used as a tie-breaker.
    sorted_chars = sorted(freq_dict.keys(), key=lambda x: (-freq_dict[x], s.index(x)))

    # Step 3: Extract the sorted keys to form the result string.
    return ''.join(sorted_chars)


def dyck1_func(s: str) -> str:
    open_count = 0
    result = []
    invalid = False  # to keep track if we've entered an invalid state

    for char in s:
        if invalid:  # if we've previously determined it's invalid, all further chars will be "F"
            result.append('F')
            continue

        if char == '(':
            open_count += 1
        elif char == ')':
            open_count -= 1

        if open_count == 0:
            result.append('T')
        elif open_count > 0:
            result.append('P')
        else:
            result.append('F')
            invalid = True

    return result


def dyck2_func(s: str) -> str:
    stack = []
    result = []

    for char in s:
        if char in ['(', '{']:
            stack.append(char)
            result.append('P')  # assume it's a prefix until proven otherwise
        else:
            if not stack:
                result.append('F')
                continue
            
            top = stack[-1]
            if char == ')' and top == '(':
                stack.pop()
            elif char == '}' and top == '{':
                stack.pop()
            else:
                result.append('F')
                continue

            # Check for a balanced state
            if not stack:
                result[-len(stack):] = ['T'] * len(result[-len(stack):])
            else:
                result[-1] = 'P'

    # If the stack is not empty at the end, remaining characters are prefixes
    while stack:
        result.append('P')
        stack.pop()

    return ''.join(result)



def load_txt_to_lists(filename):
    lists = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove whitespace and newline characters
            line = line.strip()
            
            # Convert the string representation of a list into an actual list
            try:
                current_list = ast.literal_eval(line)
                if isinstance(current_list, list):
                    lists.append(current_list)
            except (SyntaxError, ValueError):
                print(f"Failed to parse line: {line}")
                continue
    return lists


for step in [60,300,600,3600]:
    filename = "./result/dyck1/trace_test_cases_"+str(step)+".txt"
    lists_from_file = load_txt_to_lists(filename)
    error = 0
    for l in lists_from_file:
        l = l[1:]
        if len(l)==0:
            continue
        result = dyck1.run(l)
        dyck1_result = dyck1_func(l)
        # converted_lst = [int(item) if item.isdigit() else item for item in result]
        if dyck1_result != result:
            error+=1

    print(error)
