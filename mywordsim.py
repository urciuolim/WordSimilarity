import sys
import numpy
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr as correlation
import time
import random


class NotInGloVeException(Exception):
    def __init__(self, msg):
        self.msg = msg

def word2WE(word, glove):
    if word in glove:
        return glove[word]
    raise NotInGloVeException("'" + word + "' not in gloVe.")

def cosine(vec1, vec2):
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def findMostSimilar(WE, glove, exclude):
    maxsim = (None, -2)
    for otherword in glove:
        # When processing analogies I use this line to filter against
        # words already in the analogy (not against plurals though)
        if otherword in exclude:
            continue
        cos_sim = cosine(WE, glove[otherword])
        if cos_sim > maxsim[1]:
            maxsim = (otherword, cos_sim)
    return maxsim

def generateParallel(wordsim, glove):
    pairs = []
    for tupl in wordsim:
        try:
            WE1 = word2WE(tupl[0], glove)
            WE2 = word2WE(tupl[1], glove)
            pairs.append(((tupl[0], tupl[1]), (tupl[2], cosine(WE1, WE2))))
        except NotInGloVeException as e:
            print(e.msg)
    return pairs

def outputParallel(pairs):
    with open(sys.argv[3], 'w') as output:
        for pair in pairs:
            output.write(pair[0][0] + '\t' + pair[0][1] + '\t' + str(pair[1][0]) + '\t' + str(pair[1][1])[:6] + '\n')
    print("Output written to " + sys.argv[3])

def computeCorrelation(pairs):
    X = []
    Y = []
    for pair in pairs:
        X.append(pair[1][0])
        Y.append(pair[1][1])

    return correlation(X, Y)

def main():
    neednumargs = 4
    if len(sys.argv) < neednumargs:
           print("Need " + str(neednumargs-len(sys.argv)) + " more arguments")
           print("python3 mywordsim.py <gloVe file path> <wordsim file path> <output file path>")
           exit(1)
    glove = dict()
    with open(sys.argv[1], 'r') as gin:
        lines = gin.readlines()
        counter = 0
        for line in lines:
            vector = []
            first = True
            for item in line.split(" "):
                if first:
                    vector.append(item.lower())
                    first = False
                else:
                    vector.append(float(item.replace('\n','')))
            glove[vector[0]] = numpy.array(vector[1:])
            counter += 1
            if counter % 10000 == 0:
                print('.', end='', flush=True)
        print('')
        
    print("gloVe loaded")
    wordsim = []
    with open(sys.argv[2], 'r') as ws:
        lines = ws.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            split = line.split('\t')
            tupl = (split[1].lower(), split[2].lower(), float(split[3].replace('\n','')))
            wordsim.append(tupl)
    print("wordsim loaded")
    
    pairs = generateParallel(wordsim, glove)
    outputParallel(pairs)

    with open(sys.argv[3], 'a') as output:
        output.write("CORR\t" + str(computeCorrelation(pairs)[0]) + "\n\n")
        output.write("Manually found analogies:\n")
        analogy = ["strong", "stronger", "dark"]
        analogystring = analogy[0] + ':' + analogy[1] + "::" + analogy[2] + ":"
        output.write(analogystring)
        A = word2WE(analogy[0], glove)
        B = word2WE(analogy[1], glove)
        C = word2WE(analogy[2], glove)
        D = numpy.add(numpy.subtract(B,A),C)
        answer = findMostSimilar(D, glove, analogy)
        output.write(answer[0] + " with similarity " + str(answer[1])[:6] + "\n")

        analogy = ["vodka", "whiskey", "soda"]
        analogystring = analogy[0] + ":" + analogy[1] + "::" + analogy[2] + ":"
        output.write(analogystring)
        A = word2WE(analogy[0], glove)
        B = word2WE(analogy[1], glove)
        C = word2WE(analogy[2], glove)
        D = numpy.add(numpy.subtract(B,A),C)
        answer = findMostSimilar(D, glove, analogy)
        output.write(answer[0] + " with similarity " + str(answer[1])[:6] + "\n")

        analogy = ["money", "bank", "beer"]
        analogystring = analogy[0] + ":" + analogy[1] + "::" + analogy[2] + ":"
        output.write(analogystring)
        A = word2WE(analogy[0], glove)
        B = word2WE(analogy[1], glove)
        C = word2WE(analogy[2], glove)
        D = numpy.add(numpy.subtract(B,A),C)
        answer = findMostSimilar(D, glove, analogy)
        output.write(answer[0] + " with similarity " + str(answer[1])[:6] + "\n")

    print("Example analogies printed to " + sys.argv[3])

    while True:
        try:
            print("Please input an analogy in format\nA:B::C:")
            user_input = input("->")
            if user_input == "exit" or user_input == "quit":
                break
            elif user_input == "save":
                with open(sys.argv[3], 'a') as output:
                    output.write(analogystring + answer[0] + " with similarity " + str(answer[1])[:6] + "\n")
                print("last analogy saved")
                continue
            user_input = user_input.split(":")
            analogy = [user_input[0], user_input[1], user_input[3]]
            analogystring = analogy[0] + ":" + analogy[1] + "::" + analogy[2] + ":"
            A = word2WE(analogy[0], glove)
            B = word2WE(analogy[1], glove)
            C = word2WE(analogy[2], glove)
            D = numpy.add(numpy.subtract(B,A),C)
            answer = findMostSimilar(D, glove, analogy)
            print(answer[0] + " with similarity " + str(answer[1])[:6] + "\n")
        except NotInGloVeException as e:
            print(e.msg)
        except:
            print("Problem with input, try again (or enter 'exit' or 'quit' to exit)")

    
    
if __name__== '__main__':
    main()
