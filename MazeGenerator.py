import numpy as np
import os

class Maze:

    def __init__(self, hight = 10, width = 10):

        self.width = width
        self.hight = hight

        self.start = np.array([0,0], dtype=int)
        self.end = np.array([self.hight-1, self.width-1], dtype= int)
        self.walls = np.zeros((self.hight,self.width),dtype= bool)

    def SetStart(self, row, column):
        self.start = np.array([row,column])
        self.walls[row,column] = False
        if np.all(self.start==self.end):
            self.end = np.array([-1,-1])

    def SetEnd(self, row, column):
        self.end = np.array([row,column])
        self.walls[row,column] = False
        if np.all(self.start==self.end):
            self.start = np.array([-1,-1])

    def PutWall(self, row, column):
        if not np.all(self.start == np.array([row,column])) and not np.all(self.end == np.array([row,column])):
            self.walls[row,column] = True

    def RemoveWall(self, row, column):
        self.walls[row,column] = False

    def Save(self, name):
        name = 'Mazes/'+ name
        if not os.path.isdir('Mazes/'):
            os.mkdir('Mazes/')
        if not os.path.isdir(name):
            os.mkdir(name)
        np.savetxt(name + '/start.txt', self.start, fmt='%i')
        np.savetxt(name + '/end.txt', self.end, fmt='%i')
        np.savetxt(name + '/walls.txt', self.walls, fmt='%i')

    def Load(self, name):
        name = 'Mazes/' + name
        self.start = np.loadtxt(name + '/start.txt', dtype= int)
        self.end = np.loadtxt(name + '/end.txt', dtype=int)
        self.walls = np.loadtxt(name + '/walls.txt', dtype=bool)


name = input('Maze name: ')
name =  name
if os.path.isdir('Mazes/'+ name):
    print('Loading existing maze:', name)
    test = Maze()
    test.Load(name)

else:
    print('Creating new maze:', name)
    test = Maze(10, 15)

