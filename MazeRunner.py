import ANN
import MazeGenerator as MG
import numpy as np

class MazeRunner:

    winning_action_reward = 1.0
    neutral_action_reward = -0.01
    prohibited_action_reward = -1.0

    def __init__(self, maze_name):

        self.maze = MG.Maze()
        self.maze.Load(maze_name)
        self.current_position = self.maze.start
        self.goal = self.maze.end

    def SetPosition(self, row, column):

        if row < 0 or column < 0 or self.maze.hight <= row or self.maze.width <= column or self.maze.walls[row,column]:
            return False

        self.current_position=np.array([row,column])
        return True

    def MakeMoveWithReward(self, action):
        row = self.current_position[0]
        column = self.current_position[1]

        if action==0:
            row+=1
        elif action==1:
            row-=1
        elif action==2:
            column+=1
        else:
            column-=1

        if not self.SetPosition(row,column):
            return self.prohibited_action_reward, True
        elif np.all(self.current_position==self.goal):
            self.current_position = np.array([row, column])
            return self.winning_action_reward, True
        else:
            self.current_position = np.array([row,column])
            return self.neutral_action_reward, False

    def GetState(self):
        return self.current_position

    def ResetGame(self):
        self.current_position = self.maze.start

store_path = "MazeNewTraining"
MazeANN = ANN.TrainingNetwork(2, 4, 0.05, [50, 50, 50],discount=0.95)

def PrintLearnedPath(name):
    runner = MazeRunner(name)
    game_ended = False
    saver = ANN.tf.train.Saver()
    path = "Trainings/" + store_path
    with ANN.tf.Session() as sess:
        saver.restore(sess, path)
        count = 0
        while not game_ended:
            if runner.maze.hight*runner.maze.width< count:
                break
            count +=1
            current_state = runner.GetState()
            chosen_action = MazeANN.GetOptimalAction(current_state)
            reward, game_ended = runner.MakeMoveWithReward(chosen_action)
            print(current_state, end="->")
    print(runner.current_position)

if __name__ == '__main__':
    runner = MazeRunner("test_maze")
    MazeANN.Training_Episodic_Single_Matches_Reverse(runner,store_path,250, MazeANN.SARSA_Episodic_Single_Game,is_1v1=False,reverse=False)

    PrintLearnedPath("test_maze")