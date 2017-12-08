from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle

from random import random

import math

class MyLabel(Label):

    def on_size(self, *args):
        color = float(self.text)
        self.canvas.before.clear()
        with self.canvas.before:
            Color(color, color, color, 0.5)
            Rectangle(pos=self.pos, size=self.size)


class LoginScreen(GridLayout):

    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)

        artist_name_map = open("reddit_map.txt", "r", encoding="utf-8")
        remap = open("reddit_remap.txt", "r", encoding="utf-8")
        attn_weights = open("attn_weights_inter.txt", "r")

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

        attn_weight_values = attn_weights.readline().split(",")

        attn_weight_values = list(reversed(attn_weight_values))

        attn_weights.readline()

        input_timestamp = attn_weights.readline()

        attn_weights.readline()

        session_timestamps = []
        session_contents = []

        for i in range(15):
            session_timestamps.append(attn_weights.readline())
            session_contents.append(attn_weights.readline().split(","))

        # normalize attention weights
        """
        min = 1
        max = 0
        for i in attn_weight_values[:15]:
            if float(i) < min:
                min = float(i)
            if float(i) > max:
                max = float(i)
        for i in range(15):
                attn_weight_values[i] = (float(attn_weight_values[i]) - min) / (max - min)
        """

        show_timestamp = True

        # print sess reps
        for i in range(15):
            if show_timestamp:
                label = str(math.floor((float(input_timestamp) - float(session_timestamps[i]))/3600))
                self.add_widget(Label(text=label, size_hint=(0.3, 0.2)))
            else:
                string = ""
                for j in range(19, -1, -1):
                    sss = session_contents[i][j]
                    if sss == "0":
                        string += "\n"
                    else:
                        string += artists_remap[session_contents[i][j]] + "\n"
                self.add_widget(Label(text=string))

        #for i in range(15):
            label = MyLabel(
                    text="{:.3f}".format(float(attn_weight_values[i+1])),
                    pos=(20, 20),
                    size_hint=(0.5, 0.2))
            self.add_widget(label)

        self.cols = 2

class MyApp(App):

    def build(self):
        return LoginScreen()


if __name__ == '__main__':
    MyApp().run()