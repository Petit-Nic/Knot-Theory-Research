import matplotlib.collections
import matplotlib.pyplot
import DiagObjects
import matplotlib

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


        orientedEndpNumberTop = {'Bl': 0, 'Up': 1, 'Do': 1, 'U+': 2, 'U-': 2, 'U0': 2, 'D+': 2, 'D-': 2, 'D0': 2,
                                 'R+': 2, 'R-': 2, 'R0': 2, 'L+': 2, 'L-': 2, 'L0': 2, 'Lu': 2, 'Ru': 2, 'Ld': 0, 'Rd': 0}
        orientedEndpNumberBot = {'Bl' : 0, 'Up': 1, 'Do': 1, 'U+': 2, 'U-': 2, 'U0': 2, 'D+': 2, 'D-': 2, 'D0': 2, 
                                 'R+': 2, 'R-': 2, 'R0': 2, 'L+': 2, 'L-': 2, 'L0': 2, 'Lu': 0, 'Ru': 0, 'Ld': 2, 'Rd': 2}
        endpNumberTop = {'Bl' : 0, 'Id': 1, 'X+': 2, 'X-': 2, 'X0': 2, 'Cu': 2, 'Ca': 0}
        endpNumberBot = {'Bl' : 0, 'Id': 1, 'X+': 2, 'X-': 2, 'X0': 2, 'Cu': 0, 'Ca': 2}
        if self.code[0][0] in endpNumberTop:
            self.oriented = False
            self.topEndp = [[endpNumberTop[i] for i in level] for level in strands]
            self.botEndp = [[endpNumberBot[i] for i in level] for level in strands]
        else: 
            self.oriented = True
            self.topEndp = [[orientedEndpNumberTop[i] for i in level] for level in strands]
            self.botEndp = [[orientedEndpNumberBot[i] for i in level] for level in strands]
        self.isValid()

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
                        case 'Up':
                            topWord += 'T'
                            botWord += 'H'
                        case 'Do':
                            topWord += 'H'
                            botWord += 'T'
                        case 'U+' | 'U-' | 'U0':
                            topWord += 'TT'
                            botWord += 'HH'
                        case 'D+' | 'D-' | 'D0':
                            topWord += 'HH'
                            botWord += 'TT'
                        case 'R+' | 'R-' | 'R0':
                            topWord += 'HT'
                            botWord += 'HT'
                        case 'L+' | 'L-' | 'L0':
                            topWord += 'TH'
                            botWord += 'TH'
                        case 'Lu':
                            topWord += 'TH'
                        case 'Ru':
                            topWord += 'HT'
                        case 'Ld':
                            botWord += 'TH'
                        case 'Rd': 
                            botWord += 'HT'
                topHT.append(topWord)
                botHT.append(botWord)
            self.topHT = topHT
            self.botHT = botHT
            for i in range(len(botHT)-1):
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
            badTangles = ['X+', 'X-', 'X0', 'Id', 'U+', 'U-', 'U0', 'D+', 'D-', 'D0', 'Up', 'Do']
            for item in badTangles:
                if item in basicTangles:
                    return False
            return True
        return False

    def codeToPositions(self):
         code = self.code
         positions = []
         for level in code:
             temp = []
             for word in level:
                if word not in ['Up', 'Do', 'Id', 'Bl']:
                     temp.append(word)
                     temp.append(word)
                else:
                    temp.append(word)
             positions.append(temp)
         return positions
                    
   
    def countingStrands(self):
        code = self.codeToPositions()

        component = 1
        diagramLabelsTop = [[character for character in word] for word in self.topHT]
        diagramLabelsBot = [[character for character in word] for word in self.botHT]
        done = False
    
        while not done:
            index = findItem(diagramLabelsBot,'H')
            if index != []:
                level = index[0][0]
                position = index[0][1]
                pointer = diagramLabelsBot[level][position]

            while pointer == 'H':
                match code[level][position]:
                    case 'Rd':
                        if (diagramLabelsBot[level][position] == 'H') | (diagramLabelsBot[level][position] == 'T'):
                            diagramLabelsBot[level][position] = str(component)
                        else:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                        position += 1
                        diagramLabelsBot[level][position] = diagramLabelsBot[level][position-1]
                        level += 1
                        pointer = diagramLabelsTop[level][position]
                    case 'Ld' :
                        if diagramLabelsBot[level][position] in ['H', 'T']:
                            diagramLabelsBot[level][position] = str(component)
                        else:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                        position -= 1
                        diagramLabelsBot[level][position]=diagramLabelsBot[level][position+1]
                        level += 1
                        pointer = diagramLabelsTop[level][position]
                    case 'Bl':
                        position += 1 
                    case "Do":
                        diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                        diagramLabelsBot[level][position] = diagramLabelsTop[level][position]
                        level += 1
                        pointer = diagramLabelsTop[level][position]
                    case 'Up':
                        diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                        diagramLabelsTop[level][position] = diagramLabelsBot[level][position]
                        level -= 1
                        pointer = diagramLabelsBot[level][position]
                    case 'Ru':
                        diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                        position += 1
                        diagramLabelsTop[level][position] = diagramLabelsTop[level][position-1]
                        level -= 1
                        pointer = diagramLabelsBot[level][position]
                    case 'Lu':
                        diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                        position -= 1
                        diagramLabelsTop[level][position] = diagramLabelsTop[level][position+1]
                        level -= 1
                        pointer = diagramLabelsBot[level][position]
                    case 'U+' | 'U-' | 'U0':
                        index = 0
                        for i in range(position):
                            if code[level][i] == code[level][position]:
                                index += 1
                        if index % 2 == 0:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                            position += 1
                            diagramLabelsTop[level][position] = diagramLabelsBot[level][position-1]
                            level -= 1
                            pointer = diagramLabelsBot[level][position]
                        else:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                            position -= 1
                            diagramLabelsTop[level][position] = diagramLabelsBot[level][position+1]
                            level -= 1
                            pointer = diagramLabelsBot[level][position]
                    case 'D+' | 'D-' | 'D0':
                        index = 0
                        for i in range(position):
                            if code[level][i] == code[level][position]:
                                index += 1
                        if index % 2 == 0:
                            diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                            position += 1
                            diagramLabelsBot[level][position] = diagramLabelsTop[level][position-1]
                            level += 1
                            pointer = diagramLabelsTop[level][position]
                        else:
                            diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                            position -= 1
                            diagramLabelsBot[level][position] = diagramLabelsTop[level][position+1]
                            level += 1
                            pointer = diagramLabelsTop[level][position]
                    case 'R+' | 'R-' | 'R0':
                        if (diagramLabelsBot[level-1][position] not in ['T', 'H']) & (diagramLabelsTop[level][position] in ['T', 'H']):
                            diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                            position += 1
                            diagramLabelsBot[level][position] = diagramLabelsTop[level][position-1]
                            level += 1
                            pointer = diagramLabelsTop[level][position]
                        else:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                            position += 1
                            diagramLabelsTop[level][position] = diagramLabelsBot[level][position-1]
                            level -= 1
                            pointer = diagramLabelsBot[level][position]
                    case 'L+' | 'L-' | 'L0':
                        if (diagramLabelsBot[level-1][position] not in ['T', 'H']) & (diagramLabelsTop[level][position] in ['T', 'H']):
                            diagramLabelsTop[level][position] = diagramLabelsBot[level-1][position]
                            position -= 1
                            diagramLabelsBot[level][position] = diagramLabelsTop[level][position+1]
                            level -= 1
                            pointer = diagramLabelsTop[level][position]
                        else:
                            diagramLabelsBot[level][position] = diagramLabelsTop[level+1][position]
                            position -= 1
                            diagramLabelsTop[level][position] = diagramLabelsBot[level][position+1]
                            level -= 1
                            pointer = diagramLabelsBot[level][position]

            topCharacterSet = set(character for part in diagramLabelsTop for character in part)
            if 'H' in topCharacterSet :
                component += 1
                continue
            done=True
        return component
        



                

    def draw(self):
        totalStrands = list(zip([sum(item) for item in self.topEndp], [sum(item) for item in self.botEndp]))
        maxLength = max(max(item[0] for item in totalStrands), max(item[1] for item in totalStrands))
        fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(maxLength+1, self.levels+1))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        matplotlib.pyplot.xlim(-0.5, maxLength)
        matplotlib.pyplot.ylim(-self.levels-1.5, -0.5)
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
                        case 'X0':
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
            else: 
                for bit in level:
                    match bit:
                        case 'Bl':
                            endpointIndex += 1
                        case 'Up':
                            arrow = matplotlib.pyplot.arrow(endpointIndex, -levelIndex-1, 0, 1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.add_patch(arrow)
                            endpointIndex += 1
                        case 'Do':
                            arrow = matplotlib.pyplot.arrow(endpointIndex, -levelIndex, 0, -1, color= 'k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.add_patch(arrow)
                            endpointIndex += 1
                        case 'U0':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "D0":
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "R0":
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case "L0":
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            circle = matplotlib.patches.Circle((endpointIndex+0.5, -levelIndex-0.5), 0.2, color='k', fill=False)
                            ax.add_patch(circle)
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'U+':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.3, -levelIndex-0.3, -0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex +1, endpointIndex + 0.7], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'D+':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.7, -levelIndex-0.7, 0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex+0.3], [-levelIndex, -levelIndex-0.3], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'R+':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex, +1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.7, -levelIndex-0.3, 0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex+0.3], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'L+':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.3, -levelIndex-0.7, -0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex+0.7, endpointIndex+1], [-levelIndex-0.3, -levelIndex], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'U-':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex-1, -1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.7, -levelIndex-0.3, 0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex + 0.3], [-levelIndex-1, -levelIndex-0.7], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'D-':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex, 1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.3, -levelIndex-0.7, -0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex+0.7, endpointIndex + 1], [-levelIndex-0.3, -levelIndex], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'R-':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex, -levelIndex-1, 1, 1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.7, -levelIndex-0.7, 0.3, -0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            ax.plot([endpointIndex, endpointIndex + 0.3], [-levelIndex, -levelIndex-0.3], color='k')
                            ax.add_patch(arrow2)
                            ax.add_patch(arrow1)
                            endpointIndex += 2
                        case 'L-':
                            arrow1 = matplotlib.pyplot.arrow(endpointIndex+1, -levelIndex, -1, -1, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
                            arrow2 = matplotlib.pyplot.arrow(endpointIndex+0.3, -levelIndex-0.3, -0.3, 0.3, color='k', length_includes_head=True, head_width=0.2, head_length=0.2)
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


        matplotlib.pyplot.show(block=False)
        matplotlib.pyplot.waitforbuttonpress(0)
        matplotlib.pyplot.close()

 
 
def findItem(theList, item):
        return [(ind, theList[ind].index(item)) for ind in range(len(theList)) if item in theList[ind]]


       

code = ['LdRd', 'DoU-Do', 'R0Lu', 'Lu']
code2 = ['LdRd', 'DoU+Do', 'RuLu']
tangle = Tangle(code)
tangle2 = Tangle(code2)



