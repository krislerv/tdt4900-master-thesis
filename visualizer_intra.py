from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle

from random import random

class MyLabel(Label):

    def on_size(self, *args):
        color = float(self.text)
        self.canvas.before.clear()
        with self.canvas.before:
            Color(color, color, color, 0.5)
            Rectangle(pos=self.pos, size=self.size)


class Visualizer(GridLayout):

    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)

        show_timestamp = False

        artist_name_map = open("reddit_map.txt", "r", encoding="utf-8")
        remap = open("reddit_remap.txt", "r", encoding="utf-8")
        attn_weights = open("attn_weights_intra.txt", "r")

        artists = {}
        artists_remap = {}


        for line in artist_name_map:
            line = line.split(" ")
            artists[line[0]] = line[1]
        #print(artists)

        for line in remap:
            line = line.replace(" ", "")
            line = line.split(",")
            for kv in line:
                kv = kv.split(":")
                artists_remap[kv[1]] = artists[kv[0]]

        #print(artists_remap)

        session_length = int(attn_weights.readline()) + 1

        top_predictions = attn_weights.readline().split(",")
        top_predictions.pop() # remove trailing comma

        attn_weights.readline()

        attn_weight_list = []
        session_contents = []

        for i in range(15):
            attn_weight = attn_weights.readline().split(",")
            attn_weight.pop()
            attn_weight_list.append(attn_weight)
            session_content = attn_weights.readline().split(",")
            session_content.pop()
            session_contents.append(session_content)
            attn_weights.readline()

        # normalize attention weights
        """
        min = 1
        max = 0
        for i in attn_weight_list:
            for j in i:
                if float(j) < min:
                    min = float(j)
                if float(j) > max:
                    max = float(j)
        for i in range(len(attn_weight_list)):
            for j in range(len(attn_weight_list[i])):
                attn_weight_list[i][j] = (float(attn_weight_list[i][j]) - min) / (max - min)
        """

        self.add_widget(Label(size_hint=(1, 12)))


        # print sess reps
        for i in range(15):
            if show_timestamp:
                self.add_widget(Label(text="timestamp"))
            else:
                string = ""
                for j in range(19, -1, -1):
                    sss = session_contents[i][j]
                    if sss == "0":
                        string += "\n"
                    else:
                        string += artists_remap[session_contents[i][j]] + "\n"
                self.add_widget(Label(text=string))

        # print prediction + attn weights

        for i in range(19):
            self.add_widget(Label(text=artists_remap[top_predictions[i]]))
            for j in range(15):
                label = MyLabel(
                    text="{0:.6}".format(attn_weight_list[j][i]),
                    pos=(20, 20),
                    size_hint=(0.5, 0.5))
                self.add_widget(label)

        self.cols = 16

class MyApp(App):

    def build(self):
        return Visualizer()


if __name__ == '__main__':
    MyApp().run()