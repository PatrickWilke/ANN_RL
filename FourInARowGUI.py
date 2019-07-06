from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse
from kivy.core.window import Window
import FourInARow as FIAR
import ANN


load_path = "./FourInARowVeryEasy"
AI_for_red = True
AI_for_yellow = False

class BoardGrafics(Widget):

    board = FIAR.FourInARowBoard()
    score = [0, 0]
    game_ended = False
    show_score = False
    first_move = True

    def SetToEmptyBoard(self):
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1)
            d = Window.size[1]/12
            for i in range(0, 6):
                for j in range(0, 7):
                    Ellipse(pos=(Window.size[0]/7 * j+ Window.size[0]/14 - d/2, Window.size[1]/6 * i + Window.size[1]/12 - d/2), size=(d, d))


    def SetColumn(self, column):
        with self.canvas:
            Color(1, self.board.GetNextMove(), 0)
            d = Window.size[1]/12
            Ellipse(pos=(Window.size[0]/7 * column + Window.size[0]/14 - d/2, self.board.GetColumnHight(column) * Window.size[1]/6 + Window.size[1]/12 - d/2), size=(d, d))


    def ShowScoreAfterEnd(self):
        self.canvas.clear()
        with self.canvas:
            d = Window.size[1] / 12
            Color(1, 1, 0)
            Ellipse(pos=(Window.size[0]/3, Window.size[1]/2 - d), size=(d, d))

            Color(1, 0, 0)
            Ellipse(pos=(Window.size[0]/3, Window.size[1] / 2 + d), size=(d, d))
            l1 = Label(text=': ' + str(self.score[0]), font_size='50sp', pos=(Window.size[0] / 3 + 1.5*d, Window.size[1] / 2 +d ))
            l2 = Label(text=': ' + str(self.score[1]), font_size='50sp',
                  pos=(Window.size[0] / 3 + 1.5*d, Window.size[1] / 2 -d))
        self.game_ended = False
        self.show_score = True


    def RestartGameAfterScore(self):
        self.show_score = False
        self.board.ResetGame()
        self.SetToEmptyBoard()
        if AI_for_red and not AI_for_yellow:
            self.first_move = True
            self.AIMakeMove()

    def NormalGameMove(self, column):
        if not self.board.ColumnIsFull(column):
            self.SetColumn(column)
            if self.board.MakeMove(column):
                self.score[self.board.GetNextMove()] += 1
                self.game_ended = True
            elif self.board.GameEnded():
                self.game_ended = True

    def AIMakeMove(self):
        state = self.board.GetSate()
        if self.first_move:
            action = FIAR.FourInARowANN.WeightedAction(state)
            self.first_move = False
        else:
            action = FIAR.FourInARowANN.GetOptimalAction(state)

        if self.board.ColumnIsFull(action):
            print("AI made prohibited move. Game counts as Draw.")
            self.game_ended = True
        else:
            self.NormalGameMove(action)

    def on_touch_down(self, touch):
        if self.game_ended:
            self.ShowScoreAfterEnd()
        elif self.show_score:
            self.RestartGameAfterScore()
        else:
            if not (AI_for_yellow and AI_for_red):
                column = int((7 * touch.x) // self.width)
                self.NormalGameMove(column)
            if AI_for_red and not self.game_ended:
                self.AIMakeMove()
            if AI_for_yellow and not self.game_ended:
                self.AIMakeMove()



class FourInARowApp(App):

    def build(self):
        new_graphics = BoardGrafics()
        new_graphics.SetToEmptyBoard()
        if AI_for_red and not AI_for_yellow:
            new_graphics.AIMakeMove()
        return new_graphics



if __name__ == '__main__':
    if AI_for_red or AI_for_yellow:
        with ANN.tf.Session() as sess:
            saver = ANN.tf.train.Saver()
            if ANN.os.path.isfile(load_path + ".index"):
                saver.restore(sess, load_path)
            else:
                print("Path for AI does not exist! Game with completely untrained AI ...")
                ANN.init.run()
            FourInARowApp().run()
    else:
        FourInARowApp().run()

