import sys
import numpy
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr as correlation


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
            pairs.append((tupl[2], cosine(WE1, WE2)))
        except NotInGloVeException as e:
            print(e.msg)
    return pairs

def computeCorrelation(pairs):
    X = []
    Y = []
    for pair in pairs:
        X.append(pair[0])
        Y.append(pair[1])

    return correlation(X, Y)

def main():
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
            #if counter % 10000 == 0:
                #print("Read in " + str(counter) + " vectors")

    wordsim = []
    with open(sys.argv[2], 'r') as ws:
        lines = ws.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            split = line.split('\t')
            tupl = (split[1].lower(), split[2].lower(), float(split[3].replace('\n','')))
            wordsim.append(tupl)

    pairs = generateParallel(wordsim, glove)
    print(computeCorrelation(pairs))

    #analogy = ["strong", "stronger", "dark"]
    #A = word2WE(analogy[0], glove)
    #B = word2WE(analogy[1], glove)
    #C = word2WE(analogy[2], glove)
    #D = numpy.add(numpy.subtract(B,A),C)
    #print(findMostSimilar(D, glove, analogy))

    
    
if __name__== '__main__':
    main()
