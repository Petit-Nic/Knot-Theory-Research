import DiagObjects
import itertools

def mixs(num):
    # Allows for sorting of str and int lists. Str come first.
    try:
        ele = int(num[1])
        return (1, ele, '')
    except ValueError:
        return (0, num, '')
    
def findInDict(dictionary, value):
    # Returns all indices in a dictionary corresponding to value.
    locs = []
    pairs = list(zip(dictionary.keys(), dictionary.values()))
    for item in pairs:
        if item[1] == value:
            locs.append(pairs.index(item))
    return locs


def AIP(gaussDiag, w=False):
    # Computes the AIP of the given diagram. Outputs the coefficients of the Laurent polynomial starting from the lowest
    # power, and the value of the lowest power. If the optional argument w is set to True it also first outputs the
    # weights of each crossing in the form (weight, sign)
    arrows = gaussDiag.arrows
    labels = DiagObjects.AffineLabeling(gaussDiag, 0)
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}  # used to apply the corresponding operation
    # depending on crossing sign
    weights = []

    # computing all the weights using the formula pre-tail label - pre-head label - sign
    for item in arrows:
        weight = ops[item.sign](labels[item.tail] - labels[item.head], -1)
        weights.append([weight, item.sign])
    weights.sort()

    # Finding the smallest exponent and the exponent range
    smallPower = weights[0][0]
    powerRange = weights[-1][0] - weights[0][0]
    if smallPower < 0 < smallPower + powerRange:
        polynomial = [0, ] * (powerRange + 1)
    else:
        polynomial = [0, ] * (powerRange + abs(smallPower) + 1)

    # Creates the polynomial, where polynomial[0] is the coefficient of the smallest power
    if smallPower < 0:
        for item in weights:
            polynomial[item[0]-smallPower] = ops[item[1]](polynomial[item[0]-smallPower], 1)
            polynomial[-smallPower] = ops[item[1]](polynomial[-smallPower], -1)
    else:
        for item in weights:
            polynomial[item[0]] = ops[item[1]](polynomial[item[0]], 1)
            polynomial[0] = ops[item[1]](polynomial[0], -1)
        smallPower = 0

    if w:
        return polynomial, smallPower, weights
    return polynomial, smallPower

def linkAIP(stringLink, startingLabels, w=False):
    # Given a stringLink and a set of startingLabels (with len(startingLabels) == stringlink.components) outputs
    # a vector whose entries are the exponents of t in the AIP with the relative coefficient
    arrows = stringLink.arrows
    label = DiagObjects.linkAffineLabeling(stringLink, startingLabels, integer=True)
    # Note that we take the integer labeling and convert it to a string at the end
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}
    weights = []

    # This part computes the weights and signs of each crossing. If we're dealing with a self-crossing we just
    # take the integer weight, otherwise we convert it to a string of the form 'weight + a - b', where a is the
    # starting tail label letter (so a+1, a, a-5 all have starting tail label letter 'a') and b is the
    # starting head label letter
    for item in arrows:
        tail, head, sign = item.tail, item.head, item.sign
        weight = ops[sign](label[tail[0]][tail[1]] - label[head[0]][head[1]], -1)
        if tail[0] != head[0]:
            strtlbl = startingLabels[tail[0]]
            endlbl = startingLabels[head[0]]
            if weight == 0:  # This avoids the ugly '0+a-b' weight, replacing it with 'a-b'
                weight = strtlbl.replace(strtlbl[1:], '') + '-' + endlbl.replace(endlbl[1:], '')
            else:
                weight = str(weight) + '+' + strtlbl.replace(strtlbl[1:], '') + '-' + endlbl.replace(endlbl[1:], '')
        weights.append([weight, item.sign])

    # Initialize exponents, coefficients, writhe
    exps = []
    coeffs = []
    writhe = 0

    # add each new exponent to the set. If the exponent was not present we add the vector [exponent, 1] to the set
    # of coefficients; otherwise we find the location of the exponents in coeffs and apply the ops corresponding to
    # the sign to change it by +-1
    for item in weights:
        if item[0] not in exps:
            exps.append(item[0])
            coeffs.append([item[0], ops[item[1]](0, 1)])
        else:
            coeffindex = exps.index(item[0])
            oldvalue = coeffs.pop(coeffindex)[1]
            coeffs.insert(coeffindex, [item[0], ops[item[1]](oldvalue, 1)])
        writhe = ops[item[1]](writhe, 1)

    # If no crossing had exponent 0 we add a coefficient corresponding to -writhe, otherwise we take the
    # 0 exponent coefficient and subtract the writhe from it
    if 0 not in exps:
        coeffs.append([0, -writhe])
    else:
        coeffindex = exps.index(0)
        oldvalue = coeffs.pop(coeffindex)[1]
        coeffs.append([0, oldvalue - writhe])

    output = reverseLinkOutput(coeffs)

    if w:
        return weights, sorted(output, key=mixs)
    return sorted(output, key=mixs)

def linkMultiAIP(stringLink, startingLabels, w=False):
    arrows = stringLink.arrows
    label = DiagObjects.linkAffineBilabeling(stringLink, startingLabels, integer=True)
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}
    weights = []

    for item in arrows:
        tail, head, sign = item.tail, item.head, item.sign
        weight = ops[sign](sum(label[tail[0]][tail[1]]) - sum(label[head[0]][head[1]]), -1)

        if tail[0] != head[0]:
            strtlbl = startingLabels[tail[0]]
            endlbl = startingLabels[head[0]]
            strtlblsum = '(' + strtlbl[0].replace(strtlbl[0][2:], '') + '+' \
                         + strtlbl[1].replace(strtlbl[1][2:], '') + ')'
            endlblsum = '(' + endlbl[0].replace(endlbl[0][2:], '') + '+' \
                        + endlbl[1].replace(endlbl[1][2:], '') + ')'
            if weight == 0:  # This avoids the ugly '0+a-b' weight, replacing it with 'a-b'
                weight = strtlblsum + '-' + endlblsum
            else:
                weight = str(weight) + '+' + strtlblsum + '-' + endlblsum
        weights.append([weight, item.sign, item.tail[0]])

    writhe = 0
    exps = [[] for i in range(stringLink.components)]
    polynomial = [[] for i in range(stringLink.components)]
    for item in weights:
        if item[0] != 0:
            writhe = ops[item[1]](writhe, 1)
            if item[0] not in exps[item[2]]:
                exps[item[2]].append(item[0])
                polynomial[item[2]].append([ops[item[1]](0, 1), item[0]])
            else:
                coeffindex = exps[item[2]].index(item[0])
                oldvalue = polynomial[item[2]].pop(coeffindex)[0]
                polynomial[item[2]].insert(coeffindex,[ops[item[1]](oldvalue, 1), item[0]])

    # Cleaning up the resulting polynomial by deleting any entries whose coefficient is zero, and any components
    # that do not contribute to the final polynomial
    for item in polynomial:
        for part in item:
            if part[0] == 0:
                item.remove(part)

    polynomial.append(-writhe)

    if w:
        return weights, polynomial
    return polynomial

def computeAIP(object, label=[]):
    # if the object is a GaussDiag ignores the labeling and computes the AIP. If the object is a Stringlink
    # and the label is a list of strings it computes the link AIP. If the object is a StringLink and the label
    # is a list of list of strings it computes the multivariable AIP
    if type(object) is DiagObjects.GaussDiag:
        return AIP(object)
    elif type(label[0]) is not list:
        if type(label[0]) is str:
            return linkAIP(object, label)
        else:
            letterlabel = DiagObjects.intToLetterLabel(label)
            aip = linkAIP(object, letterlabel)
            return evaluateLinkAIPLabel(aip, letterlabel, label)
    else:
        if type(label[0][0]) is str:
            return linkMultiAIP(object, label)
        else:
            letterlabel = DiagObjects.intToLetterLabel(label)
            aip = linkMultiAIP(object, letterlabel)
            return evaluateMultiAIPLabel(aip, letterlabel, label)


# Functions that play with AIPs and their outputs
def collapseMultiAIP(outputAIP):
    # Given a suitable multivariable AIP output it rewrites it as the output of the link AIP. Not user-proof.
    # Each bilabel has the form ['a1', 'a2'] and the corresponding label is 'a'.

    temp = [item for item in outputAIP]
    writhe = outputAIP[-1]
    temp.remove(writhe)
    flatten = [item for list in temp for item in list]
    fixedformat = []
    for item in flatten:
        fixedformat.append([item[1], item[0]])

    if type(outputAIP[-1]) is int:
        fixedformat.append([0, writhe])

    s = set()
    dict = {}
    for item in fixedformat:
        if type(item[0]) is str:
            substr = (item[0][-7:], item[0][-15:-8])
            for part in substr:
                if part not in s:
                    s.add(part)
                    dict[part] = part[1]

    for item in fixedformat:
        if type(item[0]) is str:
            for key, word in dict.items():
                item[0] = item[0].replace(key, word)

    return sorted(fixedformat, key=mixs)

def collapseLinkAIP(linkAIP):
    # Given a suitable linkAIP output it reduces it to the output of AIP(gaussDiag). Not user-proof.

    outputAIP = reverseLinkOutput(linkAIP)
    lowestpower = outputAIP[0][0]
    highestpower = outputAIP[-1][0]
    powerrange = highestpower - lowestpower
    coeffs = [0 for i in range(powerrange + 1)]

    for item in outputAIP:
        slot = item[0] - lowestpower
        coeffs[slot] = item[1]

    return (coeffs, lowestpower)

def evaluateLinkAIPLabel(flippedaip, startingLabel, labelValues):
    # Given a linkAIP output with startinglabel, and the values labelValues, outputs the specialization of the
    # linkAIP where startingLabels have been replaced by labelValues. So startingLabel = ['a', 'b'] and
    # labelValues = [1,0] will replace any 'a-b' string in aip with 1 and any b-a string with -1, and
    # add it appropriately to the integer part of the exponent.

    aip = reverseLinkOutput(flippedaip)
    output = []
    labelcomb = itertools.combinations(labelValues, 2)
    startingcomb = list(itertools.combinations(startingLabel,2))
    labelsum = [item[0]-item[1] for item in labelcomb]
    dictionary = {}

    for i in range(len(startingcomb)):
        dictionary[startingcomb[i][0] + '-' + startingcomb[i][1]] = labelsum[i]
        dictionary[startingcomb[i][1] + '-' + startingcomb[i][0]] = -labelsum[i]

    for item in aip:
        if type(item[0]) is str:
            if len(item[0]) > 3:
                output.append([int(item[0][:-4]) + int(dictionary[item[0][-3:]]), item[1]])
            else:
                output.append([int(dictionary[item[0]]), item[1]])
        else:
            output.append(item)

    exps = []
    simpleoutput = []

    for item in output:
        if item[0] in exps:
            index = exps.index(item[0])
            oldvalue = simpleoutput.pop(index)[1]
            simpleoutput.append([item[0], oldvalue + item[1]])
        else:
            simpleoutput.append(item)
            exps.append(item[0])

    return sorted(reverseLinkOutput(simpleoutput), key=mixs)

def evaluateMultiAIPLabel(aip, startingLabel, labelValues):
    # Take a multivariable AIP, the relative startingLabel (in the form e.g.  [['a1', 'a2'], ['b1', 'b2']])
    # and a list of labelValues and outputs the resulting specialization of the multivariable AIP to
    # those starting values. So startingLabel = [['a1', 'a2'], ['b1', 'b2']] and labelValues = [[0,1], [-1, 1]]
    # replaces '(a1+a2)-(b1+b2)' with (0+1)-(-1+1)=1, and '(b1+b2)-(a1+a2)' with -1 in every term of the
    # multivariable AIP (and combines coefficients of terms with the same exponent after the substitution).

    collapsedletters = ['(' + item[0] + '+' + item[1] + ')' for item in startingLabel]
    collapsedvalues = [item[0] + item[1] for item in labelValues]

    output = [[] for i in range(len(aip) - 1)]
    output.append(aip[-1])
    labelcomb = itertools.combinations(collapsedvalues, 2)
    startingcomb = list(itertools.combinations(collapsedletters,2))
    labelsum = [item[0]-item[1] for item in labelcomb]
    dictionary = {}

    for i in range(len(startingcomb)):
        dictionary[startingcomb[i][0] + '-' + startingcomb[i][1]] = labelsum[i]
        dictionary[startingcomb[i][1] + '-' + startingcomb[i][0]] = -labelsum[i]

    for i in range(len(aip)-1):
        for part in aip[i]:
            if type(part[1]) is str:
                if len(part[1]) > 15:
                    output[i].append([part[0], int(part[1][:-16]) + int(dictionary[part[1][-15:]])])
                else:
                    output[i].append([part[0], int(dictionary[part[1]])])
            else:
                output[i].append(part)

    exps = [[] for i in range(len(output)-1)]
    simpleoutput = [[] for i in range(len(output)-1)]
    simpleoutput.append(output[-1])

    for i in range(len(output)-1):
        for item in output[i]:
            if item[1] in exps[i]:
                index = exps[i].index(item[1])
                oldvalue = simpleoutput[i].pop(index)[0]
                simpleoutput[i].append([oldvalue + item[0], item[1]])
            else:
                simpleoutput[i].append(item)
                exps[i].append(item[1])

    sortedoutput = []
    for i in range(len(simpleoutput)-1):
        sortedoutput.append(sorted(simpleoutput[i]))
    sortedoutput.append(simpleoutput[-1])

    return sortedoutput

def allPossibleAIP(stringLink, startingLabel):
    # Given a stringLink and a starting label it computes all possible AIPs of that StringLink
    # with the given startingLabel. It does so by computing all the different Gauss Codes (i.e. all possible
    # locations of the starting point) and computing each AIP.

    # Prints out AIPdictionary (a dictionary with the Gauss Codes and relative outputs),
    # totalAIPOutput (all outputs), condensedAIPOutput (no duplicates), and duplicates (all codes which
    # yield the same AIP with their indices in the dictionary). Returns a dictionary object.

    totout = open('totalAIPOutput.txt', 'w+')
    shortout = open('condensedAIPOutput.txt', 'w+')
    dup = open('duplicates.txt', 'w+')
    dictionary = open('AIPdictionary.txt', 'w+')
    d = {}
    duplicates = []
    allcodes = DiagObjects.rotateCode(stringLink)
    for item in allcodes:
        aip = computeAIP(DiagObjects.StringLink(item), startingLabel)
        totout.write(str(aip) + '\n')
        if aip not in d.values():
            shortout.write(str(aip)+'\n')
        elif aip not in duplicates:
            duplicates.append(aip)
        d[item] = aip

    for item in duplicates:
        dup.write(str(item) + '\n')
        locs = findInDict(d, item)
        dup.write('Duplicates found at indices ' + str(locs) + '\n')
        dup.write('Gauss codes of Duplicates:' + '\n')
        for part in locs:
            dup.write('\t' + str(list(d.items())[part][0]) + '\n')

    totout.close()
    shortout.close()
    dup.close()
    for key, value in d.items():
        dictionary.write(str(key)+ ':' + str(value) + '\n')
    return d

#Utility functions
def reverseLinkOutput(linkAIP):

    output = []
    for item in linkAIP:
        output.append([item[1], item[0]])

    return output

#WIP below


def polyChengGao(gaussdiag):
    # Returns the Cheng-Gao writhe polynomial of a given Gauss Diagram. It is computed via the formula/normalization
    # W_K(t) = (P_K(t)+ Q_K)t where P_K(t) is the AIP and Q_K is the total write of the crossings with nonzero
    # index(C-G)/crossing weight(AIP)

    # Output is in the form (coeffs, lowestpower) and lists the coefficients of each power of t starting
    # with t^(lowestpower)

    polynomial, smallpower, weights = AIP(gaussdiag, w=True)
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}
    qk = 0
    for item in weights:
        if item[0] != 0:
            qk = ops[item[1]](qk, 1)

    cg = []
    if smallpower > 0:
        for i in range(smallpower):
            cg.append(0)
        for item in polynomial:
            cg.append(item)
    else:
        for i in range(len(polynomial)):
            cg.append(polynomial[i])
        oldweight = cg.pop(-smallpower)
        cg.insert(-smallpower, oldweight + qk)

    return cg, smallpower + 1


