import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import itertools

# New classes
class GaussDiag:
    def __init__(self, input):
        # Given a Gauss code generates the relative Gauss diagram, containing Arrow objects.
        length = len(input)
        self.code = input
        templist = [[0, 0, ''] for i in range(length // 6)]
        for i in range(length):
            if input[i] == 'O':
                templist[int(input[i + 1]) - 1][0] = (i + 1) // 3
                templist[int(input[i + 1]) - 1][2] = input[i + 2]
            else:
                if input[i] == 'U':
                    templist[int(input[i + 1]) - 1][1] = (i + 1) // 3
        templist = sorted(templist, key=lambda i: i[0])

        self.arrows = [Arrow(item) for item in templist]

    def __len__(self):
        # Returns the length (i.e. number of arrows/crossings) of the diagram
        return len(self.list())

    def list(self):
        # returns a list of arrows
        return diagToList(self)

    def __repr__(self):
        return str(self.list())

    def __str__(self):
        return str(self.code)

    def draw(self, affinelabeling=[], labelPoints=False, arrowsigns=True):
        # Draws the (round) Gauss diagram in pyplot. The endpoints of the arrows are the 2n-th roots of unity,
        # where n is the number of arrows of the diagram.

        # Finding the arrows and computing the equally spaced arrow endpoints via roots of unity
        arrows = self.arrows
        length = len(self) * 2
        roots = rootsOfUnity(length)

        # Setting the window and drawing the unit circle that forms the skeleton of the diagram
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_aspect('equal', adjustable='datalim')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        cir = plt.Circle((0, 0), 1, color='black', fill=False)
        ax.add_patch(cir)
        plt.xlim(-1.3, 1.3)
        plt.ylim(-1.3, 1.3)

        # Stuff to properly display legend
        s = []
        leg = []
        d = {'+':'k', '-':'r'}
        # Drawing each arrow red if negative, black if positive, and tracking the first occurence of a positive or
        # negative arrow
        for item in arrows:
            tail = roots[item.tail]
            head = roots[item.head]

            if item.sign not in s:
                s.append(item.sign)
                leg.append(plt.arrow(tail[0], tail[1], head[0] - tail[0], head[1] - tail[1], label=item.sign,
                                     length_includes_head=True, head_width=.1, head_length=.1,
                                     color=d[item.sign]))
            else:
                plt.arrow(tail[0], tail[1], head[0] - tail[0], head[1] - tail[1], label=item.sign,
                          length_includes_head=True, head_width=.1, head_length=.1,
                          color=d[item.sign])

            #Adds signs at the foot and head of arrows. Foot gets the opposite of the arrow sign, head gets the sign
            #Useful to compute Left/Right Over/Under indices, and the index of a crossing from it.
            if arrowsigns:
                plt.annotate(item.sign, # this is the text
                [1.1*head[0], 1.1*head[1]], # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center')
                minusec = ["+", "-"]
                minusec.remove(item.sign)
                plt.annotate(str(minusec[0]), # this is the text
                [1.1*tail[0], 1.1*tail[1]], # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center')

        # uncomment next bit to plot heads and tails of arrows
        # xcoords = [item[0] for item in roots]
        # ycoords = [item[1] for item in roots]
        # plt.plot(xcoords, ycoords, 'ok')



        # The following plots the labeling (if present) on the diagram, plotting the points of the labels in blue
        if affinelabeling:
            labelRoots = rootsOfUnity(2*length)[1::2]
            xcoords = [item[0] for item in labelRoots]
            ycoords = [item[1] for item in labelRoots]
            if labelPoints:
                plt.plot(xcoords, ycoords, 'ob')

            for item in labelRoots:
                plt.annotate(affinelabeling[labelRoots.index(item)], # this is the text
                [1.2*item[0], 1.2*item[1]], # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

        plt.legend(leg, list(s))
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close()

class Arrow:
    def __init__(self, list):
        self.tail = list[0]
        self.head = list[1]
        self.sign = list[2]

    def list(self):
        return [self.tail, self.head, self.sign]

    def __repr__(self):
        return str(self.list())

class StringLink:
    def __init__(self, input):
        # self.components is the number of components, self.complength[i] is the number of arrows/crossings
        # in component i, self.maxlength is the maximum number of crossings in a component (used for drawing
        # purposes, self.arrows is the list of arrows of the component.
        self.components = len(input)
        self.complength = []
        self.code = input
        for item in input:
            self.complength.append(len(item)//3)
        self.maxlength = max(self.complength)
        totallength = sum(self.complength)*3
        templist = [[0, 0, ''] for i in range(totallength//6)]

        d={'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'A':10, 'B':11, 'C':12, 'D':13, 'E':14,
           'F':15, 'G':16, 'H':17, 'I':18, 'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25}

        for item in input:
            length = len(item)//3
            while item != '':
                if item[0] == 'O':
                    templist[d[item[1]]-1][0] = [findSubstringIndex(item, input), length - (len(item)//3)]
                    templist[d[item[1]]-1][2] = item[2]
                    item = item.replace(str(item[0:3]), '')
                else:
                    templist[d[item[1]]-1][1] = [findSubstringIndex(item, input), length - (len(item)//3)]
                    item = item.replace(str(item[0:3]), '')

        self.arrows = [Arrow(item) for item in templist]

    def list(self):
        # returns a list of arrows
        return diagToList(self)

    def __len__(self):
        # Returns the length (i.e. number of arrows/crossings) of the diagram
        return len(self.list())

    def __repr__(self):
        return str(self.list())

    def __str__(self):
        return str(self.code)

    def draw(self, affineLabeling = [], labelPoints=False, arrowpoints=False, arrowsigns=False):
        # draws the StringLink
        comp = self.components
        totallength = sum(self.complength)

        # sets up the figure and adjusts the axes
        if totallength > 10:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.xlim(-0.5, self.maxlength)
        plt.ylim(-0.5, comp-0.5)

        # plots the skeleton of the string link
        for i in range(comp):
            ax.plot([j for j in range(-1, self.maxlength+1)], [i for j in range(-1, self.maxlength+1)],
                    color='k')

        # legend setup
        s = []
        leg = []
        # drawing every arrrow
        for item in self.arrows:
            head, tail, sign = item.head[::-1], item.tail[::-1], item.sign

            # arrows is black if crossing is positive, red if negative
            d = {'+':'k', '-':'r'}

            # plotting endpoints and arrows
            if arrowpoints:
                plt.plot([head[0], tail[0]], [head[1], tail[1]], 'ok')
            if head[1] == tail[1] or head[0] == tail[0]:
                a = patches.FancyArrowPatch(tail, head, arrowstyle='-|>', shrinkA=5, shrinkB=5,
                                            label=sign, mutation_scale=20, color=d[sign],
                                            connectionstyle='arc3, rad=-0.5')
                ax.add_patch(a)
            else:
                a = patches.FancyArrowPatch(tail, head, arrowstyle='-|>', shrinkA=5, shrinkB=5,
                                            label=sign, mutation_scale=20, color=d[sign],
                                            connectionstyle='arc3')
                ax.add_patch(a)
            if sign not in s:
                s.append(sign)
                leg.append(a)

            if arrowsigns:
                plt.annotate(sign, # this is the text
                [head[0], head[1]], # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,5), # distance from text to points (x,y)
                ha='center')
                minusec = ["+", "-"]
                minusec.remove(item.sign)
                plt.annotate(str(minusec[0]), # this is the text
                [tail[0], tail[1]], # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,5), # distance from text to points (x,y)
                ha='center')

        # If an affineLabeling is provided, writes the labels below the skeleton of the string link
        if affineLabeling:
            textsize = 'small'
            if type(affineLabeling[0][0]) is list:
                if type(affineLabeling[0][0][0]) is str:
                    textsize = 'x-small'
            for i in range(self.components):
                labelpos = []
                for j in range(self.complength[i] + 1):
                    labelpos.append((j-0.4, i))
                for item in labelpos:
                    # If labelPoints is true, adds blue dots to represent the spots the labels belong to
                    if labelPoints:
                        plt.plot(item[0], item[1], 'ob')
                    label = affineLabeling[i][labelpos.index(item)]
                    plt.annotate(label, # this is the text
                    item, # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                                 fontsize=textsize,
                    xytext=(0,-20), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

        plt.legend(leg, s)
        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close()


# Functions to define labelings and compute AIPs
def AffineLabeling(gaussDiag, startingLabel):
    # Returns the affine labeling of the given Gauss Diagram starting at the value startingLabel. The starting value
    # doesn't matter when computing the AIP, but can lead to different labelings.
    length = len(gaussDiag)
    label = [startingLabel]
    arrows = [item.list() for item in gaussDiag.arrows]
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}
    for i in range(2*length):
        [ar, ht] = findItem(arrows, i)
        # Leaving this legacy version of the labeling for reference
        if ht == 0:
            label.append(ops[arrows[ar][2]](label[i], -1))
        else:
            label.append(ops[arrows[ar][2]](label[i], 1))
    return label

def linkAffineLabeling(stringLink, startingLabels, integer=False):
    # Returns the affine labeling of stringLink given the initial startingLabels (such that
    # stringLink.components == len(startingLabels). If startingLabels is a string of the form ['a', 'b+1', 'c-1']
    # and integer=True we will work with [0, 1, -1] instead (useful for linkAIP).
    # Output is a list of list, ordered by stringLink components, with each sublist being the list of labels of
    # the component.

    # If startingLabels is a list of strings isolates the numerical weight they represent in tempstartlabel.
    if type(startingLabels[0]) == str:
        tempstartlabel = []
        for item in startingLabels:
            if len(item) == 1:
                tempstartlabel.append(0)
            else:
                tempstartlabel.append(int(item.replace(item[0], '')))
    else:
        tempstartlabel = startingLabels

    # turns label into a list of lists, so we can later use append
    label = []
    for i in range(len(tempstartlabel)):
        label.append([tempstartlabel[i]])

    # Lists out the arrows and defines the operations related to crossing signs
    arrows = [item.list() for item in stringLink.arrows]
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}

    # for every component i, for every arrow endpoint [i,j] in the component find the arrow to which it belongs
    # then take the last label and do ops with (-1) ** index[1]+1. If tail the index[1]+1=1, so we do ops
    # with -1, if heads it's 1+1=2, so we do ops with 1.
    for i in range(stringLink.components):
        j = 0
        while j < stringLink.complength[i]:
            index = findItem(arrows, [i, j])
            label[i].append(ops[arrows[index[0]][2]](label[i][-1], (-1)**(index[1]+1)))

            j += 1

    # If integer=False and startingLabels is a list of strings, take the numerical weights and add the
    # starting letter after them; e.g. a label of [0, 1, 2, 1, 0] with starting label 'c' turns into
    # ['c', '1+c', '2+c', '1+c', 'c']
    if type(startingLabels[0]) == str and not integer:
        for item in label:
            srtlbl = startingLabels[label.index(item)]
            for i in range(len(item)):
                if item[i] != 0:  # avoids having the ugly '0+a' weight
                    item[i] = str(item[i]) + "+" + srtlbl.replace(srtlbl[1:], '')
                else:
                    item[i] = srtlbl.replace(srtlbl[1:], '')

    return label

def linkAffineBilabeling(stringLink, startingLabels, integer=False):
    # Takes a string link and a starting Bilabel (a list of lists, with as many entries as the number of
    # components). Each entry has two slots, either both integers or both strings of the form letter+number
    # plus integer (e.g. [['a1', 'a2-1'], ['b1+3', 'b2']])

    # Outputs the list of bilabels of each component. If integer=True it ignores the starting letter+number labels

    # Converts if necessary the starting labels to integers, e.g. [['a1', 'a2-1'], ['b1+3', 'b2']] to
    # [[0,-1], [3, 0]]
    if type(startingLabels[0][0]) == str:
        templabel = []
        for item in startingLabels:
            bilabel = []
            for j in range(2):
                if len(item[j]) == 2:
                    bilabel.append(0)
                else:
                    bilabel.append(int(item[j].replace(item[j][:2], '')))
            templabel.append(bilabel)
    else:
        templabel = startingLabels

    # Initializes label as list of list of lists
    label = []
    for i in range(len(templabel)):
        label.append([templabel[i], ])

    arrows = [item.list() for item in stringLink.arrows]
    ops = {"+": (lambda x, y: x+y), "-": (lambda x, y: x-y)}

    # Computes the bilabeling for each component, for each arrow in the component. Slightly different than
    # the AffineLabeling code b/c python freaked out when using list of lists of lists and kept updating
    # the label list when appending the next labeling. Still works.
    for i in range(stringLink.components):
        j = 0
        while j < stringLink.complength[i]:
            index = findItem(arrows, [i, j])
            arrow = stringLink.arrows[index[0]]
            if arrow.head[0] == arrow.tail[0]:
                update = ops[arrow.sign](label[i][j][0], (-1) ** (index[1]+1))
                label[i].append([update, label[i][j][1]])
            else:
                update = ops[arrow.sign](label[i][j][1], (-1) ** (index[1]+1))
                label[i].append([label[i][j][0], update])

            j += 1

    # If startingLabels was a list of lists of strings and integer=False, converts each list atom into a string
    # and adds the appropriate letter+number from startingLabels (e.g. [0, 2] with startingLabels ['a1', 'a2-1']
    # turns into ['0+a1', '2+a2']
    if type(startingLabels[0][0]) == str and not integer:
        for item in label:
            for part in item:
                strtlbl = startingLabels[label.index(item)]
                for i in range(2):
                    if part[i] != 0:
                        part[i] = str(part[i]) + "+" + strtlbl[i].replace(strtlbl[i][2:], '')
                    else:
                        part[i] = strtlbl[i].replace(strtlbl[i][2:], '')

    return label

def AIP(gaussDiag, w=False):
    # Computes the AIP of the given diagram. Outputs the coefficients of the Laurent polynomial starting from the lowest
    # power, and the value of the lowest power. If the optional argument w is set to True it also first outputs the
    # weights of each crossing in the form (weight, sign)
    arrows = gaussDiag.arrows
    labels = AffineLabeling(gaussDiag, 0)
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
    label = linkAffineLabeling(stringLink, startingLabels, integer=True)
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
    label = linkAffineBilabeling(stringLink, startingLabels, integer=True)
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
    if type(object) is GaussDiag:
        return AIP(object)
    elif type(label[0]) is not list:
        if type(label[0]) is str:
            return linkAIP(object, label)
        else:
            letterlabel = intToLetterLabel(label)
            aip = linkAIP(object, letterlabel)
            return evaluateLinkAIPLabel(aip, letterlabel, label)
    else:
        if type(label[0][0]) is str:
            return linkMultiAIP(object, label)
        else:
            letterlabel = intToLetterLabel(label)
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
    allcodes = rotateCode(stringLink)
    for item in allcodes:
        aip = computeAIP(StringLink(item), startingLabel)
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


# Utility functions related to GD and SL
def draw(object, startingLabel = [], labelPoints=False, arrowpoints=False, arrowsigns=False):
    # Draws the Gauss Diagram/String Link. If startingLabel is provided, also includes the labeling.
    # If labelPoints is true, also draws blue dots to show where the labels are placed.

    if startingLabel:
        if type(object) is GaussDiag:
            affineLabeling = AffineLabeling(object, startingLabel[0])
        elif type(startingLabel[0]) is not list:
            affineLabeling = linkAffineLabeling(object, startingLabel)
        else:
            affineLabeling = linkAffineBilabeling(object, startingLabel)
        object.draw(affineLabeling, labelPoints, arrowpoints, arrowsigns)
    else:
        if type(object) is GaussDiag:
            object.draw([], labelPoints, arrowsigns)
        else:
            object.draw([], labelPoints, arrowpoints, arrowsigns)

def rotateCrossing(string):
    # Given a Gauss code takes the first crossing and moves it to the end of the code

    return string[3:] + string [:3]

def rotateComponent(string):
    # Given a Gauss code, returns all possible Gauss codes (as strings) due to changing the starting point.

    output = [string]
    currstring = string
    for i in range(len(string)//3 -1):
        currstring = rotateCrossing(currstring)
        output.append(currstring)
    return output

def rotateCode(stringLink):
    # Given a stringLink it returns all possible Gauss codes for the virtual link associated with the string link.
    # It rotates all components then computes the outer product of each rotateComponent result.

    code, components = stringLink.code, stringLink.components
    totalcode = []
    for i in range(components):
        totalcode.append(rotateComponent(code[i]))
    allcodes = list(itertools.product(*totalcode))
    return(allcodes)

def intToLetterLabel(label):
    # Take an integer labeling of a stringlink and returns a labeling of the form ['a', 'b', 'c', ...] for an
    # AffineLabeling and [['a1', 'a2'], ['b1', 'b2'], ...] for an AffineBilabeling

    if type(label[0]) is int:
        letterlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    else:
        letterlist = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2'], ['d1', 'd2'],['e1', 'e2'], ['f1', 'f2'],
                      ['g1', 'g2'], ['h1', 'h2'], ['i1', 'i2'], ['j1', 'j2']]
    return [letterlist[i] for i in range(len(label))]

def diagToList(gaussDiag):
    # Converts Arrow objects to list (for easier printing)
    return [item.list() for item in gaussDiag.arrows]

def diagToCode(gaussDiag):
    # Outputs the Gauss Code of the given Gauss Diagram
    list = diagToList(gaussDiag)
    code = ''
    for i in range(len(list) * 2):
        index = findItem(list, i)
        if index[1] == 0:
            code += 'O'
        else:
            code += 'U'
        code += str(index[0] + 1)
        code += list[index[0]][2]
    return code

def reverseLinkOutput(linkAIP):

    output = []
    for item in linkAIP:
        output.append([item[1], item[0]])

    return output


#Utility functions unrelated to GD and SL
def printList(list):
    # Prints all items of a list for easier console viewing

    for item in list:
        print(item)

def findInDict(dictionary, value):
    # Returns all indices in a dictionary corresponding to value.
    locs = []
    pairs = list(zip(dictionary.keys(), dictionary.values()))
    for item in pairs:
        if item[1] == value:
            locs.append(pairs.index(item))
    return locs

def keysInDict(dictionary, indices):
    # Returns all keys corresponding to a given list of indices in the dictionary

    keys = list(dictionary.keys())
    output = []
    for item in indices:
        output.append(keys[item])
    return output

def rootsOfUnity(power):
    # Finds the power-th roots of unity, starting from (1,0) and going around the circle counterclockwise
    roots = []
    pi = np.pi
    for i in range(power):
        roots.append([np.cos(2 * pi * i / power), np.sin(2 * pi * i / power)])
    return roots

def findItem(theList, item):
    # Finds index of item in list of lists
    return [(ind, theList[ind].index(item)) for ind in range(len(theList)) if item in theList[ind]][0]

def mixs(num):
    # Allows for sorting of str and int lists. Str come first.
    try:
        ele = int(num[1])
        return (1, ele, '')
    except ValueError:
        return (0, num, '')

def stringToSum(numstring):
    # Returns the sum of a string of integers numstring

    s = numstring.split('+')
    for item in s:
        if '-' in item:
            temp = item.split('-')
            tempsum = int(temp[0])
            temp.pop(0)
            while len(temp) > 0:
                tempsum = tempsum - int(temp.pop(0))
            s[s.index(item)] = tempsum
        else:
            s[s.index(item)] = int(item)
    return sum(s)

def findSubstringIndex(string, list):
    # find the index in a list of strings containing the desired substring
    for i in range(len(list)):
        if string in list[i]:
            return i


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

g = GaussDiag('O1-U2-O3+U1-O2-U3+')
g2 = GaussDiag('O1-O2-O3+U2-U1-O4-U5+U3+U6+O5+U4-U7+U8+O6+O8+O7+')
s = StringLink(['O1-U2-O3+U1-O2-U3+'])
s2 = StringLink(['O4+O2+U4+U1+U3+', 'O1+U2+O3-'])
s3 = StringLink(['O2+O3+O4-O1-', 'U4-U2+', 'U1-U3+'])
s4 = StringLink(['O1+U1+O2-U3-O4+U5-', 'O3-O5+U6-O7-O8+U7-O6-U8+', 'U4+U9+OA+UB-UA+O9+U2-OB-'])
s5 = StringLink(['O1+U2-O3-', 'O4+O2-U3-U4+U1+'])

draw(s2, startingLabel= ['a','b'], arrowpoints = True)
