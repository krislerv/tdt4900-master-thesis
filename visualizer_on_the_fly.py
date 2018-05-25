from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window

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

        self.dataset = "lastfm-3-months"
        self.run_name = "on_the_fly_attn_weights-2018-05-24-13-47-15-hierarchical-lastfm-3-months"

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


        artist_name_map = open(dataset + "_map.txt", "r", encoding="utf-8")
        remap = open(dataset + "_remap.txt", "r", encoding="utf-8")

        artists = {}
        self.artists_remap = {}


        for line in artist_name_map:
            line = line.split(" ")
            artists[line[0].strip()] = " ".join(line[1:]).strip()

        for line in remap:
            line = line.split(" ")
            self.artists_remap[line[1].strip()] = artists[line[0]]


        self.attn_weights = open("attn_weights/on_the_fly_attn_weights-" + run_name + ".txt", "r").readlines().split("\n\n\n\n")

        print(self.attn_weights)
        

    def visualize(self):
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

        # print sess reps
        self.add_widget(Label(text="event", size_hint=(0.3, 0.4)))
        self.add_widget(Label(text="attn", size_hint=(0.3, 0.4)))

        for i in range(15):
            if show_timestamp:
                label = str(math.floor((float(input_timestamp) - float(session_timestamps[i]))/3600))
                self.add_widget(Label(text=label, size_hint=(0.3, 0.2)))

            label = MyLabel(
                    attn_text="{:.3f}".format(float(attn_weight_values[i+1])),
                    color=str(normalized_weights[i]),
                    pos=(20, 20),
                    size_hint=(0.2 if show_week_time else 0.5, 0.2))
            self.add_widget(label)

        self.cols = 2


    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':
            print("w)")
        elif keycode[1] == 's':
            print(s)
        return True


class MyApp(App):

    def build(self):
        return Visualizer()


if __name__ == '__main__':
    MyApp().run()