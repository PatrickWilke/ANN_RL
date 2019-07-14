import numpy as np
import ANN

class FourInARowBoard:

    def __init__(self):
        self.board = np.zeros((2, 6, 7), dtype=bool)
        self.hights = np.zeros((7), dtype=int)
        self.next_move = 0
        self.number_of_moves = 0

    def SiteIsOccupied(self, row, column):
        return self.board[0][row][column] or self.board[1][row][column]

    def ColumnIsFull(self, column):
        return self.hights[column] > 5

    def WinningMove(self, column, row):

        if self.hights[column] > 3:
            if np.all(self.board[self.next_move,self.hights[column]-4:self.hights[column],column]):
                return True

        for possible_directions in [self.board[self.next_move,row],np.diagonal(self.board[self.next_move],offset=column-row),
                                    np.diagonal(self.board[self.next_move,:, ::-1],offset=6-column-row)]:
            count = 0
            for element in possible_directions:
                if element:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        return False

    def MakeMove(self,column):
        self.number_of_moves += 1
        self.board[self.next_move, self.hights[column], column] = True
        self.hights[column] += 1
        if self.WinningMove(column, self.hights[column]-1):
            return True
        else:
            self.next_move = 1 - self.next_move
            return False

    def __str__(self):
        state = "-----------------------------\n"
        for i in reversed(range(0,6)):
            state += "|"
            for j in range(0, 7):
                if self.board[0, i, j]:
                    state += " X |"
                elif self.board[1, i, j]:
                    state += " O |"
                else:
                    state += "   |"
            state += "\n-----------------------------\n"
        return state

    def ResetGame(self):
        self.board = np.zeros((2, 6, 7), dtype=bool)
        self.hights = np.zeros((7), dtype=int)
        self.next_move = 0
        self.number_of_moves = 0


    def GameEnded(self):
        return self.number_of_moves == 42

    def GetNextMove(self):
        return self.next_move

    def GetSate(self):
        return self.board.astype(float).flatten()

    def GetColumnHight(self, column):
        return self.hights[column]


class LearningFourInARow(FourInARowBoard):

    winning_action_reward = 1.0
    loosing_action_reward = -0.5
    neutral_action_reward = 0.0
    prohibited_action_reward = -1.0


    def MakeMoveWithReward(self, action):

        if self.ColumnIsFull(action):
            return self.prohibited_action_reward, True
        elif self.MakeMove(action):
            return self.winning_action_reward, True
        else:
            if self.GameEnded():
                return self.neutral_action_reward, True
            return self.neutral_action_reward, False


store_path = "FourInARowNewTraining"
FourInARowANN = ANN.TrainingNetwork(84, 7, 0.05, [50, 50, 50])

if __name__ == '__main__':
    board = LearningFourInARow()
    FourInARowANN.Training_Episodic_Single_Matches_Reverse(board,store_path,10000, FourInARowANN.Q_Learning_Episodic_Single_Game)
    #FourInARowANN.Training_Episodic_Decorrelated_Batches(board,store_path,50, 50, 5, FourInARowANN.Q_Learning_Episodic)
