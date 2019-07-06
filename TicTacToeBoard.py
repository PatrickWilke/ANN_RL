import numpy as np
import ANN

class TicTacToeBoard:

    def SiteIsOccupied(self, row, column):
        return self.positions[0, row, column] or self.positions[1, row, column]


    def WinningMove(self,row, column):
        if self.positions[self.next_move, 0, column] and self.positions[self.next_move, 1, column] and self.positions[self.next_move, 2, column]:
            return True
        if self.positions[self.next_move, row, 0] and self.positions[self.next_move, row, 1] and self.positions[self.next_move, row, 2]:
            return True
        if row == column and self.positions[self.next_move, 0, 0] and self.positions[self.next_move, 1, 1] and self.positions[self.next_move, 2, 2]:
            return True
        if row == 2-column and self.positions[self.next_move, 0, 2] and self.positions[self.next_move, 1, 1] and self.positions[self.next_move, 2, 0]:
            return True
        return False

    def ResetGame(self):
        self.number_of_moves = 0
        self.positions = np.zeros((2, 3, 3), dtype=bool)
        self.next_move = 0

    def __init__(self):
        self.positions = np.zeros((2, 3, 3), dtype=bool)
        self.next_move = 0
        self.number_of_moves = 0

    def GameEnded(self):
        return self.number_of_moves == 9

    def GetNextMove(self):
        return self.next_move

    def MakeMove(self,row, column):
        self.number_of_moves += 1
        self.positions[self.next_move, row, column] = True
        if self.WinningMove(row, column):
            return True
        else:
            self.next_move = 1 - self.next_move
            return False

    def GetSate(self):
        return self.positions.astype(float).flatten()

    def __str__(self):
        state = "-------------\n"
        for i in range(0,3):
            state += "|"
            for j in range(0, 3):
                if self.positions[0, i, j]:
                    state += " X |"
                elif self.positions[1, i, j]:
                    state += " O |"
                else:
                    state += "   |"
            state += "\n-------------\n"
        return state

class LearningBoard(TicTacToeBoard):

    winning_action_reward = 1.0
    loosing_action_reward = -0.5
    neutral_action_reward = 0.0
    prohibited_action_reward = -1.0


    def MakeMoveWithReward(self, action):
        row = action // 3
        column = action % 3

        if self.SiteIsOccupied(row,column):
            return self.prohibited_action_reward, True
        elif self.MakeMove(row,column):
            return self.winning_action_reward, True
        else:
            if self.GameEnded():
                return self.neutral_action_reward, True
            return self.neutral_action_reward, False


store_path = "TicTacToeNewTraining2"
TicTacToeANN = ANN.TrainingNetwork(18, 9, 0.05, [50, 50, 50])

if __name__ == '__main__':
    board = LearningBoard()
    TicTacToeANN.Training_1v1_Episodic(board,store_path,50000)