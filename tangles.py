import matplotlib
import matplotlib.pyplot as plt


def findItem(theList, item):
        return [(ind, theList[ind].index(item)) for ind in range(len(theList)) if item in theList[ind]]


def positionIndex(labeling, level, position, pointer):
    index = 0
    for i in range(position):
        if labeling[level][i] == pointer:
            index += 1
    return index

def computeShift(index):
    if index % 2 == 0:
        shift = 1
    else: 
        shift = -1
    return shift

def levelIndex(labeling, level, position, pointer):
    index = 0
    for i in range(level):
        if labeling[i][position] == pointer:
            index += 1
    return index

class Tangle:
    def __init__(self, tangle_code):
        strands = []
        for slice in tangle_code:
            level = []
            for i in range(0, len(slice), 2):
                level.append(slice[i:i+2])
            strands.append(level)
        self.code = strands
        self.levels = len(self.code)


        orientedEndpNumberTop = {'Bl': 0, 'U+': 2, 'U-': 2, 'UO': 2, 'D+': 2, 'D-': 2, 'DO': 2, 
                                 'R+': 2, 'R-': 2, 'RO': 2, 'L+': 2, 'L-': 2, 'LO': 2, 'Lu': 2, 'Ru': 2, 'Ld': 0, 'Rd': 0}
        orientedEndpNumberBot = {'Bl' : 0, 'U+': 2, 'U-': 2, 'UO': 2, 'D+': 2, 'D-': 2, 'DO': 2,
                                 'R+': 2, 'R-': 2, 'RO': 2, 'L+': 2, 'L-': 2, 'LO': 2, 'Lu': 0, 'Ru': 0, 'Ld': 2, 'Rd': 2}
        endpNumberTop = {'Bl' : 0, 'Id': 1, 'X+': 2, 'X-': 2, 'XO': 2, 'Cu': 2, 'Ca': 0}
        endpNumberBot = {'Bl' : 0, 'Id': 1, 'X+': 2, 'X-': 2, 'XO': 2, 'Cu': 0, 'Ca': 2}
        i=0
        while self.code[0][i] == 'Bl':
            i += 1
        if self.code[0][i] in endpNumberTop:
            self.oriented = False
            self.topEndp = [[endpNumberTop.get(i, 1) for i in level] for level in strands]
            self.botEndp = [[endpNumberBot.get(i, 1) for i in level] for level in strands]
        else: 
            self.oriented = True
            self.topEndp = [[orientedEndpNumberTop.get(i, 1) for i in level] for level in strands]
            self.botEndp = [[orientedEndpNumberBot.get(i, 1) for i in level] for level in strands]
        self.positions = self.codeToPositions(strands)
        self.isValid() 
        self.width = max(len(item) for item in self.positions)
        

    def list(self):
        # returns a list of arrows
        return self.code
    
    def __repr__(self):
        return str(self.list())

    def __str__(self):
        return str(self.code)

        
    def isValid(self):
        if not self.oriented:
            totalBotEndp = [sum(item) for item in self.botEndp]
            totalTopEndp = [sum(item) for item in self.topEndp]
            for i in range(len(totalBotEndp)-1):
                if totalTopEndp[i+1] != totalBotEndp[i]:
                    print("Your unoriented tangle endpoints don't match, one level has too many strands! The code will try to run, but the results won't likely make sense.")
                    programPause = input("Press the <ENTER> key to continue...")
                    return False
            return True
        else:
            totalBotEndp = [sum(item) for item in self.botEndp]
            totalTopEndp = [sum(item) for item in self.topEndp]
            for i in range(len(totalBotEndp)-1):
                if totalTopEndp[i+1] != totalBotEndp[i]:
                    print("Your oriented tangle endpoints don't match, one level has too many strands! The code will try to run, but the results won't likely make sense.")
                    programPause = input("Press the <ENTER> key to continue...")
                    return False
            topHT = []
            botHT = []
            for item in self.code:
                topWord = ''
                botWord = ''
                for bit in item:
                    match bit:
                        case "Bl":
                            topWord += '_'
                            botWord += '_'
                        case 'Up':
                            topWord += 'T'
                            botWord += 'H'
                        case 'Do':
                            topWord += 'H'
                            botWord += 'T'
                        case 'U+' | 'U-' | 'UO':
                            topWord += 'TT'
                            botWord += 'HH'
                        case 'D+' | 'D-' | 'DO':
                            topWord += 'HH'
                            botWord += 'TT'
                        case 'R+' | 'R-' | 'RO':
                            topWord += 'HT'
                            botWord += 'HT'
                        case 'L+' | 'L-' | 'LO':
                            topWord += 'TH'
                            botWord += 'TH'
                        case 'Lu':
                            topWord += 'TH'
                            botWord += '__'
                        case 'Ru':
                            topWord += 'HT'
                            botWord += '__'
                        case 'Ld':
                            botWord += 'TH'
                            topWord += '__'
                        case 'Rd': 
                            botWord += 'HT'
                            topWord += '__'
                        case _:
                            if bit[0] == 'U':
                                for i in range(int(bit[1])):
                                    botWord += '_'
                                topWord += 'T'
                                botWord += 'H'
                                for i in range(int(bit[1])):
                                    topWord += '_'
                            elif bit[0] == 'D':
                                for i in range(int(bit[1])):
                                    topWord += '_'
                                topWord += 'H'
                                botWord += 'T'
                                for i in range(int(bit[1])):
                                    botWord += '_'
                            elif bit[1] == 'U':
                                for i in range(int(bit[0])):
                                    topWord += '_'
                                topWord += 'H'
                                botWord += 'T'
                                for i in range(int(bit[0])):
                                    botWord += '_'
                            else: 
                                for i in range(int(bit[0])):
                                    botWord += '_'
                                topWord += 'T'
                                botWord += 'H'
                                for i in range(int(bit[0])):
                                    topWord += '_'
                topHT.append(topWord)
                botHT.append(botWord)
            self.topHT = topHT.copy()
            self.botHT = botHT.copy()

            for i in range(len(botHT)-1):
                topHT[i] = topHT[i].replace('_', '')
                botHT[i] = botHT[i].replace('_', '')
                for j in range(len(botHT[i])):
                    if botHT[i][j] == topHT[i+1][j]:
                        print("Your oriented tangle endpoints don't match, heads and tails of arrows are not properly connected! The code will try to run, but the results won't likely make sense.")
                        programPause = input("Press the <ENTER> key to continue...")
                        return False
            return True

    
    def isKnot(self):
        if self.isValid():
            basicTangles = {item for item in self.code[-1]}
            basicTangles.add(item for item in self.code[0])
            badTangles = ['X+', 'X-', 'XO', 'Id', 'U+', 'U-', 'UO', 'D+', 'D-', 'DO', 'Up', 'Do']
            for item in badTangles:
                if item in basicTangles:
                    return False
            
            return True
        return False


    def codeToPositions(self, code):
        positions = []
        if self.oriented:
            for level in code:
                tempTop = []
                tempBot = []
                for word in level:
                    if word in ['Up', 'Do', 'Bl']:
                        tempTop.append(word)
                        tempBot.append(word)
                    elif word[0].isdigit() & (word[1] == 'U'):
                        tempBot.append(word)
                        for i in range(int(word[0])):
                            tempBot.append('Bl')
                            tempTop.append('Bl')
                        tempTop.append(word)
                    elif word[0].isdigit():
                        tempTop.append(word)
                        for i in range(int(word[0])):
                            tempBot.append('Bl')
                            tempTop.append('Bl')
                        tempBot.append(word)
                    elif word[1].isdigit() & (word[0] == 'U'):
                        tempTop.append(word)
                        for i in range(int(word[1])):
                            tempBot.append('Bl')
                            tempTop.append('Bl')
                        tempBot.append(word)
                    elif word[1].isdigit():
                        tempBot.append(word)
                        for i in range(int(word[1])):
                            tempBot.append('Bl')
                            tempTop.append('Bl')
                        tempTop.append(word)
                    elif word in ['U+', 'U-', 'UO', 'D+', 'D-', 'DO', 'R+', 'R-', 'RO', 'L+', 'L-', 'LO']:
                        tempTop.append(word)
                        tempTop.append(word)
                        tempBot.append(word)
                        tempBot.append(word)
                    elif word in ['Lu', 'Ru']:
                        tempTop.append(word)
                        tempTop.append(word)
                        tempBot.append('Bl')
                        tempBot.append('Bl')
                    elif word in ['Ld', 'Rd']:
                        tempBot.append(word)
                        tempBot.append(word)
                        tempTop.append('Bl')
                        tempTop.append('Bl')
                positions.append(tempTop)
                positions.append(tempBot)
        else:
            for level in code:
                tempTop = []
                tempBot = []
                for word in level:
                    if word in ['Id', 'Bl']:
                        tempTop.append(word)
                        tempBot.append(word)
                    if word[0] == 'X':
                        tempTop.append(word)
                        tempTop.append(word)
                        tempBot.append(word)
                        tempBot.append(word)
                    if word == 'Ca':
                        tempTop.append('Bl')
                        tempTop.append('Bl')
                        tempBot.append(word)
                        tempBot.append(word)
                    if word == 'Cu':
                        tempBot.append('Bl')
                        tempBot.append('Bl')
                        tempTop.append(word)
                        tempTop.append(word)
                positions.append(tempTop)
                positions.append(tempBot)
        return positions
                    
   
    def countingStrands(self):
        code = self.positions.copy()
        oriented = self.oriented
        if oriented:
            component = 0
            labeling = [['Bl' for i in range(self.width)]]
            for item in code:
                labeling += [item]
            labeling.append(['Bl' for i in range(self.width)])
            
            
            diagramLabelsTop = [[character for character in word] for word in self.topHT]
            diagramLabelsBot = [[character for character in word] for word in self.botHT]
            diagramLabelsTop.append(['_' for i in range(self.width)])
            diagramLabelsBot.insert(0, ['_' for i in range(self.width)])
            diagramLabels = diagramLabelsBot.copy()
            for i in range(len(diagramLabelsTop)):
                diagramLabels.insert(2*i+1, diagramLabelsTop[i])
                           
            done = False

            while not done:
                characterSet = set(character for part in diagramLabels for character in part)
                if ('H' not in characterSet):
                    done = True
                    for item in diagramLabels:
                        print(item)
                    continue
                else:
                    component += 1
                
                index = findItem(diagramLabels,'H')
                level = index[0][0]
                position = index[0][1]
                pointer = diagramLabels[level][position]
            

                while pointer == 'H':
                    match labeling[level][position]:
                        case 'Bl':
                            continue
                        case 'Rd':
                            if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level+1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position += 1
                            diagramLabels[level][position] = diagramLabels[level][position-1]
                            level += 1
                            pointer = diagramLabels[level][position]
                        case 'Ld' :
                            if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level+1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position -= 1
                            diagramLabels[level][position] = diagramLabels[level][position+1]
                            level += 1
                            pointer = diagramLabels[level][position]
                        case "Do":
                            if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level-1][position]
                            else: 
                                diagramLabels[level][position] = str(component)
                            level +=1
                            diagramLabels[level][position] = diagramLabels[level-1][position]
                            level += 1
                            pointer = diagramLabels[level][position]
                        case 'Up':
                            if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level+1][position]
                            else: 
                                diagramLabels[level][position] = str(component)
                            level -= 1
                            diagramLabels[level][position] = diagramLabels[level+1][position]
                            level -= 1
                            pointer = diagramLabels[level][position]
                        case 'Ru':
                            if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level-1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position += 1
                            diagramLabels[level][position] = diagramLabels[level][position-1]
                            level -= 1
                            pointer = diagramLabels[level][position]
                        case 'Lu':
                            if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level-1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position -= 1
                            diagramLabels[level][position] = diagramLabels[level][position+1]
                            level -= 1
                            pointer = diagramLabels[level][position]
                        case 'U+' | 'U-' | 'UO':
                            shift = computeShift(positionIndex(labeling, level, position, labeling[level][position]))
                            if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level+1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position += shift
                            level -= 1
                            diagramLabels[level][position] = diagramLabels[level+1][position-shift]
                            level -= 1
                            pointer = diagramLabels[level][position]                          
                        case 'D+' | 'D-' | 'DO':
                            shift = computeShift(positionIndex(labeling, level, position, labeling[level][position]))
                            if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                diagramLabels[level][position] = diagramLabels[level-1][position]
                            else:
                                diagramLabels[level][position] = str(component)
                            position += shift
                            level += 1
                            diagramLabels[level][position] = diagramLabels[level-1][position-shift]
                            level += 1
                            pointer = diagramLabels[level][position]
                        case 'R+' | 'R-' | 'RO':
                            shift = computeShift(levelIndex(labeling, level, position, labeling[level][position]))
                            if shift == 1:
                                if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                    diagramLabels[level][position] = diagramLabels[level-1][position]
                                else:
                                    diagramLabels[level][position] = str(component)
                            else:
                                if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                    diagramLabels[level][position] = diagramLabels[level+1][position]
                                else:
                                    diagramLabels[level][position] = str(component)
                            level += shift
                            position += 1
                            diagramLabels[level][position] = diagramLabels[level-shift][position-1]
                            level += shift
                            pointer = diagramLabels[level][position]

                        case 'L+' | 'L-' | 'LO':
                            shift = computeShift(levelIndex(labeling, level, position, labeling[level][position]))
                            if shift == 1:
                                if (labeling[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                    diagramLabels[level][position] = diagramLabels[level-1][position]
                                else:
                                    diagramLabels[level][position] = str(component)
                            else:
                                if (labeling[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                    diagramLabels[level][position] = diagramLabels[level+1][position]
                                else:
                                    diagramLabels[level][position] = str(component)
                            level += shift
                            position -= 1
                            diagramLabels[level][position] = diagramLabels[level-shift][position+1]
                            level += shift
                            pointer = diagramLabels[level][position]
                        case _:
                            if labeling[level][position][0].isdigit():
                                shift = int(labeling[level][position][0])
                                if labeling[level][position][1] == 'U':
                                    if (diagramLabels[level+1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                        diagramLabels[level][position] = diagramLabels[level+1][position]
                                    else:
                                        diagramLabels[level][position] = str(component)
                                    position += shift
                                    level -= 1
                                    diagramLabels[level][position] = diagramLabels[level+1][position-shift]
                                    level -= 1
                                    pointer = diagramLabels[level][position]
                                else:
                                    if (diagramLabels[level-1][position] != 'Bl') & (labeling[level-1][position].isdigit()):
                                        diagramLabels[level][position] = diagramLabels[level-1][position]
                                    else:
                                        diagramLabels[level][position] = str(component)
                                    position += shift
                                    level += 1
                                    diagramLabels[level][position] = diagramLabels[level-1][position-shift]
                                    level += 1
                                    pointer = diagramLabels[level][position]
                            else:
                                shift = int(labeling[level][position][1])
                                if labeling[level][position][0] == 'U':
                                    if (diagramLabels[level+1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                        diagramLabels[level][position] = diagramLabels[level+1][position]
                                    else:
                                        diagramLabels[level][position] = str(component)
                                    position -= shift
                                    level -= 1
                                    diagramLabels[level][position] = diagramLabels[level+1][position+shift]
                                    level -= 1
                                    pointer = diagramLabels[level][position]
                                else:
                                    if (diagramLabels[level-1][position] != 'Bl') & (labeling[level+1][position].isdigit()):
                                        diagramLabels[level][position] = diagramLabels[level-1][position]
                                    else:
                                        diagramLabels[level][position] = str(component)
                                    position -= shift
                                    level += 1
                                    diagramLabels[level][position] = diagramLabels[level-1][position+shift]
                                    level += 1
                                    pointer = diagramLabels[level][position]

                
            return component
        else:
            component = 0
            labeling = [['Bl' for i in range(self.width)]]
            for item in code:
                labeling += [item]
            labeling.append(['Bl' for i in range(self.width)])
            tangleBits = [item.copy() for item in labeling]
            done = False
            while not done:
                for codeid, word in enumerate(labeling):
                    index = 0
                    for (wordid, x) in (enumerate(word)):
                        if (not x.isdigit()) & (x != 'Bl'):
                            index = (codeid, wordid)
                            level = codeid
                            position = wordid
                            pointer = labeling[level][position]
                            component += 1
                            break
                    if index != 0:
                        break
                if index == 0:
                    for item in labeling:
                        print(item)
                    done = True
                    continue
                    
                while not pointer.isdigit():
                    match pointer:
                        case 'Bl':
                            pointer = '0'
                        case 'Ca':
                            shift = computeShift(positionIndex(tangleBits, level, position, pointer))
                            if labeling[level+1][position].isdigit():
                                labeling[level][position] = labeling[level+1][position]
                            else:
                                labeling[level][position] = str(component)    
                            position += shift
                            labeling[level][position] = labeling[level][position-shift]
                            level += 1
                            pointer = labeling[level][position]
                        case 'Cu':
                            shift = computeShift(positionIndex(tangleBits, level, position, pointer))
                            if labeling[level-1][position].isdigit():
                                labeling[level][position] = labeling[level-1][position]
                            else:
                                labeling[level][position] = str(component)   
                            position += shift
                            labeling[level][position] = labeling[level][position-shift]
                            level -= 1
                            pointer = labeling[level][position]
                        case 'Id':
                            if labeling[level-1][position].isdigit():
                                labeling[level][position] = labeling[level-1][position]
                                level += 1
                                labeling[level][position] = labeling[level-1][position]
                                level += 1
                                pointer = labeling[level][position]
                            elif labeling[level+1][position].isdigit():
                                labeling[level][position] = labeling[level+1][position]
                                level -= 1
                                labeling[level][position] = labeling[level+1][position]
                                level -= 1
                                pointer = labeling[level][position]
                            else:
                                labeling[level][position] = str(component)
                                if labeling[level-1][position] != 'Bl':
                                    level -= 1
                                    labeling[level][position] = labeling[level+1][position]
                                    level -= 1
                                    pointer = labeling[level][position]
                                else:
                                    level += 1
                                    labeling[level][position] = labeling[level-1][position]
                                    level += 1
                                    pointer = labeling[level][position]
                        case 'XO' | 'X+' | 'X-' :
                            shift = computeShift(positionIndex(tangleBits, level, position, pointer))
                            if labeling[level-1][position].isdigit():
                                labeling[level][position] = labeling[level-1][position]
                                position += shift
                                level += 1
                                labeling[level][position] = labeling[level-1][position-shift]
                                level += 1
                                pointer = labeling[level][position]
                            elif labeling[level+1][position].isdigit(): 
                                labeling[level][position] = labeling[level+1][position]
                                position += shift
                                level -= 1
                                labeling[level][position] = labeling[level+1][position-shift]
                                level -= 1
                                pointer = labeling[level][position]
                            elif labeling[level+1][position] == 'Bl':
                                labeling[level][position] = str(component)
                                position += shift
                                level -= 1
                                labeling[level][position] = labeling[level+1][position-shift]
                                level -= 1
                                pointer = labeling[level][position]
                            else: 
                                labeling[level][position] = str(component)
                                position += shift
                                level += 1
                                labeling[level][position] = labeling[level-1][position-shift]
                                level += 1
                                pointer = labeling[level][position]
            for word in labeling:
                for i, part in enumerate(word):
                    word[i] = part.replace('Bl', '0')    
            return max(int(word) for level in labeling for word in level)                      




                    
        



                

    def draw(self):
        totalStrands = list(zip([sum(item) for item in self.topEndp], [sum(item) for item in self.botEndp]))
        maxLength = max(max(item[0] for item in totalStrands), max(item[1] for item in totalStrands))
        fig, ax = plt.subplots(1, 1, figsize=(maxLength+1, self.levels+1))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.xlim(-0.5, maxLength)
        plt.ylim(-self.levels-1.5, -0.5)
        for i in range(self.levels+2):
            ax.plot([j for j in range(-1, maxLength+1)], [-i for j in range(-1, maxLength+1)],
                    color='k')
            
        levelIndex = 0
        for level in self.code:
            levelIndex += 1
            endpointIndex = 0 
            if not self.oriented:
                for bit in level:
                    match bit:
                        case 'Bl':
                            endpointIndex += 1
                        case 'Id':
                            ax.plot([endpointIndex, endpointIndex], [-levelIndex, -levelIndex-1], color= 'k')
                            endpointIndex += 1
                        case 'XO':
                            ax.plot([endpointIndex, endpointIndex+1], [-levelIndex, -levelIndex-1], color='k')
                            ax.plot([endpointIndex +1, endpointIndex], [-levelIndex, -levelIndex-1], color='k')
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            endpointIndex += 2
                        case 'X+':
                            ax.plot([endpointIndex+1, endpointIndex], [-levelIndex, -levelIndex-1], color='k')
                            ax.plot([endpointIndex +1, endpointIndex + 0.7], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.plot([endpointIndex +0.3, endpointIndex], [-levelIndex-0.3, -levelIndex], color='k')
                            endpointIndex += 2
                        case 'X-':
                            ax.plot([endpointIndex, endpointIndex+1], [-levelIndex, -levelIndex-1], color='k')
                            ax.plot([endpointIndex +1, endpointIndex + 0.7], [-levelIndex, -levelIndex-0.3], color='k')
                            ax.plot([endpointIndex +0.3, endpointIndex], [-levelIndex-0.7, -levelIndex-1], color='k')
                            endpointIndex += 2
                        case 'Ca':
                            cap = matplotlib.patches.Arc((endpointIndex+0.5, -levelIndex-1), 1, 1, theta1=0.0, theta2=180.0, color='k')
                            ax.add_patch(cap)
                            endpointIndex += 2
                        case 'Cu':
                            cup = matplotlib.patches.Arc((endpointIndex+0.5, -levelIndex), 1, 1, theta1=180.0, theta2=360.0, color='k')
                            ax.add_patch(cup)
                            endpointIndex += 2
                        case _:
                            if bit[0].isdigit():
                                n = int(bit[0])
                                if bit[1] == 'U':
                                    ax.plot([endpointIndex, endpointIndex+n], [-levelIndex-1, -levelIndex], color = 'k')
                                else:
                                    ax.plot([endpointIndex, endpointIndex+n], [-levelIndex, -levelIndex-1], color = 'k')
                                endpointIndex += n
                            else: 
                                n = int(bit[1])
                                if bit[0] == 'U':
                                    ax.plot([endpointIndex, endpointIndex+n], [-levelIndex, -levelIndex-1], color = 'k')
                                else:
                                    ax.plot([endpointIndex, endpointIndex+n], [-levelIndex-1, -levelIndex], color = 'k')
                                endpointIndex += n
            else: 
                for bit in level:
                    match bit:
                        case 'Bl':
                            endpointIndex += 1
                        case 'Up':
                            arrow = plt.arrow(endpointIndex, -levelIndex-1, 0, 1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.add_patch(arrow)
                            endpointIndex += 1
                        case 'Do':
                            arrow = plt.arrow(endpointIndex, -levelIndex, 0, -1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.add_patch(arrow)
                            endpointIndex += 1
                        case 'UO':
                            arrow1 = plt.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "DO":
                            arrow1 = plt.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "RO":
                            arrow1 = plt.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "LO":
                            arrow1 = plt.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'U+':
                            arrow1 = plt.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.3, -levelIndex-0.3, -0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex +1, endpointIndex + 0.7], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'D+':
                            arrow1 = plt.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.7, -levelIndex-0.7, 0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex+0.3], [-levelIndex, -levelIndex-0.3], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'R+':
                            arrow1 = plt.arrow(endpointIndex, -levelIndex, +1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.7, -levelIndex-0.3, 0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex+0.3], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'L+':
                            arrow1 = plt.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.3, -levelIndex-0.7, -0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex+0.7, endpointIndex+1], [-levelIndex-0.3, -levelIndex], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'U-':
                            arrow1 = plt.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.7, -levelIndex-0.3, 0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex + 0.3], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'D-':
                            arrow1 = plt.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.3, -levelIndex-0.7, -0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex+0.7, endpointIndex + 1], [-levelIndex-0.3, -levelIndex], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'R-':
                            arrow1 = plt.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.7, -levelIndex-0.7, 0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex + 0.3], [-levelIndex, -levelIndex-0.3], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'L-':
                            arrow1 = plt.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = plt.arrow(endpointIndex+0.3, -levelIndex-0.3, -0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex+0.7, endpointIndex + 1], [-levelIndex-0.7, -levelIndex-1], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'Lu':
                            path = matplotlib.path.Path(((endpointIndex+1, -levelIndex), (endpointIndex+0.5, -levelIndex-1), (endpointIndex, -levelIndex)), codes=[matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
                            cap = matplotlib.patches.FancyArrowPatch(path=path, color='k', arrowstyle='-|>', shrinkA=3, shrinkB=3, mutation_scale=20, connectionstyle='arc3')
                            ax.add_patch(cap)
                            endpointIndex += 2
                        case 'Ru':
                            path = matplotlib.path.Path(((endpointIndex, -levelIndex), (endpointIndex+0.5, -levelIndex-1), (endpointIndex+1, -levelIndex)), codes=[matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
                            cap = matplotlib.patches.FancyArrowPatch(path=path, color='k', arrowstyle='-|>', shrinkA=3, shrinkB=3, mutation_scale=20, connectionstyle='arc3')
                            ax.add_patch(cap)
                            endpointIndex += 2
                        case 'Ld':
                            path = matplotlib.path.Path(((endpointIndex+1, -levelIndex-1), (endpointIndex+0.5, -levelIndex), (endpointIndex, -levelIndex-1)), codes=[matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
                            cap = matplotlib.patches.FancyArrowPatch(path=path, color='k', arrowstyle='-|>', shrinkA=3, shrinkB=3, mutation_scale=20, connectionstyle='arc3')
                            ax.add_patch(cap)
                            endpointIndex += 2
                        case 'Rd':
                            path = matplotlib.path.Path(((endpointIndex, -levelIndex-1), (endpointIndex+0.5, -levelIndex), (endpointIndex+1, -levelIndex-1)), codes=[matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
                            cap = matplotlib.patches.FancyArrowPatch(path=path, color='k', arrowstyle='-|>', shrinkA=3, shrinkB=3, mutation_scale=20, connectionstyle='arc3')
                            ax.add_patch(cap)
                            endpointIndex += 2
                        case _:
                            if bit[0].isdigit():
                                n = int(bit[0])
                                if bit[1] == 'U':
                                    arrow = plt.arrow(endpointIndex, -levelIndex-1, n, 1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                                    ax.add_patch(arrow)
                                else:
                                    arrow = plt.arrow(endpointIndex, -levelIndex, n, -1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                                    ax.add_patch(arrow)
                                endpointIndex += n+1
                            else: 
                                n = int(bit[1])
                                if bit[0] == 'U':
                                    arrow = plt.arrow(endpointIndex+n, -levelIndex-1, -n, 1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                                    ax.add_patch(arrow)
                                else:
                                    arrow = plt.arrow(endpointIndex+n, -levelIndex, -n, -1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                                    ax.add_patch(arrow)
                                endpointIndex += n+1


        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close()

 
 
##Testing area
       
#code = ['LdRd', 'DoU-Do', 'ROLu', 'Lu']
#tangle = Tangle(code)
#tangle.draw()
#print(tangle.countingStrands())

#code2 = ['LdRd', 'DoU+Do', 'RuLu']
#tangle2 = Tangle(code2)
#tangle2.draw()
#print(tangle2.countingStrands())

#code3 = ['BlBlBlLd', 'D3U1', 'DoLdRdUp', 'D-UOR+','DoR+U+Do','RuRuLu']
#tangle3 = Tangle(code3)
#tangle3.draw()
#print(tangle3.countingStrands())

#code4 = ['CaId', 'XOId', 'CuId']
#tangle4 = Tangle(code4)
#tangle4.draw()
#print(tangle4.countingStrands())

#code5 = ['UpDO','L-Do', 'DoLu']
#tangle5 = Tangle(code5)
#tangle5.draw()
#print(tangle5.countingStrands())



