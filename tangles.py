import matplotlib.collections
import matplotlib.pyplot
import DiagObjects
import matplotlib

class Tangle:
    def __init__(self, input):
        self.code = input
        strands = []
        for slice in code:
            level = []
            for i in range(0, len(slice), 2):
                level.append(slice[i:i+2])
            strands.append(level)
        self.strands = strands

        strandNumberTop = {'Id': 1, 'X+': 2, 'X-': 2, 'Xo': 2, 'Cu': 2, 'Ca': 0}
        strandNumberBot = {'Id': 1, 'X+': 2, 'X-': 2, 'Xo': 2, 'Cu': 0, 'Ca': 2}
        self.topStrands = [[strandNumberTop[i] for i in level] for level in strands]
        self.botStrands = [[strandNumberBot[i] for i in level] for level in strands]
        
        

    def draw(self):
        totalStrands = list(zip([sum(item) for item in self.topStrands], [sum(item) for item in self.botStrands]))
        maxLength = max(max(item[0] for item in totalStrands), max(item[1] for item in totalStrands))
        levels = len(self.strands)
        fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(maxLength+1, levels+1))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        matplotlib.pyplot.xlim(-0.5, maxLength)
        matplotlib.pyplot.ylim(-levels-1.5, -0.5)
        for i in range(levels+2):
            ax.plot([j for j in range(-1, maxLength+1)], [-i for j in range(-1, maxLength+1)],
                    color='k')
            
        levelIndex = 0
        for level in self.strands:
            levelIndex += 1
            endpointIndex = 0 
            for bit in level:
                match bit:
                    case 'Id':
                        ax.plot([endpointIndex, endpointIndex], [-levelIndex, -levelIndex-1], color= 'k')
                        endpointIndex += 1
                    case 'Xo':
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


        matplotlib.pyplot.show(block=False)
        matplotlib.pyplot.waitforbuttonpress(0)
        matplotlib.pyplot.close()

       

code = ['IdX+Xo', 'X-IdIdId', 'X+X-Id', 'IdXoCu']
tangle = Tangle(code)

tangle.draw()