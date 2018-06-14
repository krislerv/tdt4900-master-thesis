from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle

from random import random

import datetime

import math

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

        # to show sessions, both False
        show_timestamp = False
        show_week_time = False

        artist_name_map = open("lastfm_map.txt", "r", encoding="utf-8")
        remap = open("lastfm_remap.txt", "r", encoding="utf-8")
        attn_weights = open("attn_weights_inter.txt", "r")

        artists = {}
        artists_remap = {}


        for line in artist_name_map:
            line = line.split(" ")
            artists[line[0].strip()] = " ".join(line[1:]).strip()

        for line in remap:
            line = line.split(" ")
            artists_remap[line[1].strip()] = artists[line[0]]

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
        normalized_weights = []
        min = 1
        max = 0
        for i in attn_weight_values[1:]:
            if float(i) < min:
                min = float(i)
            if float(i) > max:
                max = float(i)
        for i in range(15):
            normalized_weights.append((float(attn_weight_values[i+1]) - min) / (max - min))
        
        print(session_contents)

        if not show_timestamp and not show_week_time:
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
                self.add_widget(Label(text=string, font_size='10sp'))
            for i in range(15):
                label = MyLabel(
                        attn_text="{:.3f}".format(float(attn_weight_values[i+1])),
                        color=str(normalized_weights[i]),
                        pos=(20, 20),
                        size_hint=(0.2, 0.2))
                self.add_widget(label)

            self.cols = 15

        else:
            # print sess reps
            if show_timestamp:
                self.add_widget(Label(text="Δt", size_hint=(0.3, 0.4)))
                self.add_widget(Label(text="attn", size_hint=(0.3, 0.4)))

            for i in range(15):
                if show_timestamp:
                    label = str(math.floor((float(input_timestamp) - float(session_timestamps[i]))/3600))
                    self.add_widget(Label(text=label, size_hint=(0.3, 0.2)))
                elif show_week_time:
                    label = datetime.datetime.utcfromtimestamp(float(session_timestamps[i])).strftime("%A")[:3] + " " + str(datetime.datetime.utcfromtimestamp(float(session_timestamps[i])).hour) + ":00"
                    self.add_widget(Label(text=label, size_hint=(0.3, 0.2)))

                label = MyLabel(
                        attn_text="{:.3f}".format(float(attn_weight_values[i+1])),
                        color=str(normalized_weights[i]),
                        pos=(20, 20),
                        size_hint=(0.2 if show_week_time else 0.5, 0.2))
                self.add_widget(label)

            self.cols = 2

class MyApp(App):

    def build(self):
        return Visualizer()


if __name__ == '__main__':
    MyApp().run()