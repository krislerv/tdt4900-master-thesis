from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle

from random import random

class MyLabel(Label):

    def __init__(self, attn_text, color, **kwargs):
        super(MyLabel, self).__init__(**kwargs)
        self.text = attn_text
        self.c = color


    def on_size(self, *args):
        color = float(self.c)
        self.canvas.before.clear()
        with self.canvas.before:
            Color(color, color, color, 0.5)
            Rectangle(pos=self.pos, size=self.size)


class Visualizer(GridLayout):

    def __init__(self, **kwargs):
        super(Visualizer, self).__init__(**kwargs)

        artist_name_map = open("reddit_map.txt", "r", encoding="utf-8")
        remap = open("reddit_remap.txt", "r", encoding="utf-8")
        attn_weights = open("attn_weights_intra.txt", "r")

        artists = {}
        artists_remap = {}


        for line in artist_name_map:
            line = line.split(" ")
            artists[line[0].strip()] = " ".join(line[1:]).strip()
        #print(artists)

        for line in remap:
            line = line.split(" ")
            artists_remap[line[1].strip()] = artists[line[0]]

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

        #attn_weight_list = list(reversed(attn_weight_list))

        normalized_weights = []

        # normalize attention weights
        min = 1
        max = 0
        for i in attn_weight_list:
            normalized_weights.append([])
            for j in i:
                if float(j) < min:
                    min = float(j)
                if float(j) > max:
                    max = float(j)
        for i in range(len(attn_weight_list)):
            for j in range(len(attn_weight_list[i])):
                normalized_weights[i].append((float(attn_weight_list[i][j]) - min) / (max - min))


        self.add_widget(Label(size_hint=(1, 12), font_size='10sp'))


        # print sess reps
        for i in range(15):
            string = ""
            for j in range(19, -1, -1):
                sss = session_contents[i][j]
                if sss == "0":
                    string += "\n"
                else:
                    l = artists_remap[session_contents[i][j]]
                    if len(l) > 9:
                        l = l[:6] + "..."
                    string += l + "\n"
            self.add_widget(Label(text=string, size_hint=(0.5, 0.5), font_size='10sp'))

        # print prediction + attn weights

        for i in range(19):
            l = artists_remap[top_predictions[i]]
            if len(l) > 20:
                l = l[:17] + "..."
            self.add_widget(Label(text=l, font_size='10sp'))
            for j in range(15):
                label = MyLabel(
                    attn_text="{0:.6}".format(attn_weight_list[j][i]),
                    color=str(normalized_weights[j][i]),
                    pos=(20, 20),
                    font_size='10sp',
                    size_hint=(0.5, 0.5))
                self.add_widget(label)

        self.cols = 16

class MyApp(App):

    def build(self):
        return Visualizer()


if __name__ == '__main__':
    MyApp().run()