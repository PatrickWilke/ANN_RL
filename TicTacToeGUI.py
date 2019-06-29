from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import (
    ListProperty, StringProperty
)
import TicTacToeBoard as TTTB
import ANN


load_path = "./TicTacToeEasy"
AI_for_X = True
AI_for_O = False


class TicTacToeGame(Widget):

    fields = ListProperty(["", "", "", "", "", "", "", "", ""])
    display_score_X = StringProperty("")
    display_score_O = StringProperty("")
    score = [0, 0]
    board = TTTB.TicTacToeBoard()
    game_ended = False
    show_score = False
    first_move = True




    def SetField(self, row, column):
        if self.board.GetNextMove() == 0:
            self.fields[3*row + column] = "X"
        else:
            self.fields[3 * row + column] = "O"

    def ShowScoreAfterEnd(self):
        self.board.ResetGame()
        self.fields = ["", "", "", "", "", "", "", "", ""]
        self.game_ended = False
        self.show_score = True
        self.display_score_X = "X:" + str(self.score[0])
        self.display_score_O = "O:" + str(self.score[1])

    def RestartGameAfterScore(self):
        self.show_score = False
        self.display_score_X = ""
        self.display_score_O = ""
        if AI_for_X and not AI_for_O:
            self.first_move = True
            self.AIMakeMove()

    def NormalGameMove(self, row, column):

        if not self.board.SiteIsOccupied(row, column):
            self.SetField(row, column)
            if self.board.MakeMove(row, column):
                self.score[self.board.GetNextMove()] += 1
                self.game_ended = True

            elif self.board.GameEnded():
                self.game_ended = True

    def AIMakeMove(self):
        state = self.board.GetSate()
        if self.first_move:
            action = TTTB.TicTacToeANN.WeightedAction(state)
            self.first_move = False
        else:
            action = TTTB.TicTacToeANN.GetOptimalAction(state)

        if self.board.SiteIsOccupied(action // 3, action % 3):
            print("AI made prohibited move. Game counts as Draw.")
            self.game_ended = True
        else:
            self.NormalGameMove(action//3,action%3)


    def on_touch_down(self, touch):
        if self.game_ended:
            self.ShowScoreAfterEnd()
        elif self.show_score:
            self.RestartGameAfterScore()
        else:

            if not (AI_for_O and AI_for_X):
                row = int(2 - (3 * touch.y) // self.top)
                column = int((3 * touch.x) // self.width)
                self.NormalGameMove(row, column)
            if AI_for_X and not self.game_ended:
                self.AIMakeMove()
            if AI_for_O and not self.game_ended:
                self.AIMakeMove()
    pass


class TicTacToeApp(App):
    def build(self):
        game = TicTacToeGame()
        if AI_for_X and not AI_for_O:
            game.AIMakeMove()
        return game


if __name__ == '__main__':
    if AI_for_X or AI_for_O:
        with ANN.tf.Session() as sess:
            saver = ANN.tf.train.Saver()
            if ANN.os.path.isfile(load_path + ".index"):
                saver.restore(sess, load_path)
            else:
                print("Path for AI does not exist! Game with completely untrained AI ...")
                ANN.init.run()
            TicTacToeApp().run()
    else:
        TicTacToeApp().run()

