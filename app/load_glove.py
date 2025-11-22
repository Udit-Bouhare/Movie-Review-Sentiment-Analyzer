# Load Glove Model 
import numpy as np 

def load_glove(path): 
    glove = {}

    with open(path,'r', encoding='utf8') as f: 
        for line in f: 
            parts = line.split() 

            word = parts[0]
            numbers = parts[1:]
            vector = np.array(numbers, dtype=float)

            glove[word] = vector 

    return glove