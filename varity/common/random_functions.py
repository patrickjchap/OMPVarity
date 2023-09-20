import random
import cfg

# This function return True or False randomly
def lucky():
    return random.randrange(0, 2) == 0
# 1/4 probability of success
def veryLucky():
    return random.randrange(0, 5) == 4

def randomComputationType():
    choice = random.randrange(0, 3)
    if choice == 0:
        return "Shared"
    elif choice == 1:
        return "Reduction"
    return "Array"

# Splits list into n random chunks.
def randomListChunk(list_to_split, n):
    ret = [[] for i in range(n)]
    for v in list_to_split:
        if v == "comp":
            ret[2].append("comp")
            continue
        if "[i]" in v:
            # We don't include indices in the pointer name.
            ret[random.randrange(0,n)].append(v[:-3])
        ret[random.randrange(0,n)].append(v)

    #print(list_to_split, " ---> ", ret)
    return ret

# Chance of generating a math function
def generateMathExpression():
    return random.random() <= cfg.MATH_FUNC_PROBABILITY

