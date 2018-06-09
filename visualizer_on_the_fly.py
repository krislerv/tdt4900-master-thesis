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

        self.dataset = "reddit"
        self.run_name = "on_the_fly_attn_weights-2018-06-05-10-14-26-hierarchical-subreddit"

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


        artist_name_map = open(self.dataset + "_map.txt", "r", encoding="utf-8")
        remap = open(self.dataset + "_remap.txt", "r", encoding="utf-8")

        artists = {}
        self.artists_remap = {}


        for line in artist_name_map:
            line = line.split(" ")
            artists[line[0].strip()] = " ".join(line[1:]).strip()

        for line in remap:
            line = line.split(" ")
            self.artists_remap[line[1].strip()] = artists[line[0]]


        self.attn_weights = open("attn_weights/" + self.run_name + ".txt", "r")

        self.all_attn_weights = []

        counter = 0
        session_data = ""
        attn_weight_data = ""

        line = self.attn_weights.readline()

        while line:
            if counter == 0:
                session_data = line.split(",")[:-1]
            elif counter == 1:
                attn_weight_data = line.split(",")[:-1]
                self.all_attn_weights.append((session_data, attn_weight_data))

            counter += 1
            if counter == 5:
                counter = 0
            line = self.attn_weights.readline()

        self.current_index = 0
        

    def visualize(self, search_label, show_sub_count):
        self.clear_widgets()
        session_contents = self.all_attn_weights[self.current_index][0]
        attn_weight_values = self.all_attn_weights[self.current_index][1]

        # normalize attention weights
        normalized_weights = []
        min = 1
        max = 0
        for i in attn_weight_values:
            if float(i) < min:
                min = float(i)
            if float(i) > max:
                max = float(i)
        for i in range(20):
            normalized_weights.append((float(attn_weight_values[i]) - min) / (max - min))

        # print sess reps
        self.add_widget(Label(text="Event", size_hint=(0.5, 0.4)))
        self.add_widget(Label(text="Attn", size_hint=(0.3, 0.4)))

        subcount = {"funny": "19,672,000", "AskReddit": "19,340,000", "pics": "18,733,000", "videos": "17,839,000", "WTF": "5,144,000", "politics": "3,852,000", "JUSTNOMIL": "266,000", "techsupport": "262,000"}

        found_search_label = False

        if show_sub_count:
            self.cols = 3
        else:
            self.cols = 2

        labels = []

        for i in range(20):
            if session_contents[i] == '0':
                label = ''
            else:
                label = self.artists_remap[session_contents[i]]
                labels.append(label)
                if label == search_label:
                    found_search_label = True

        if not found_search_label:
            return False


        for i in range(20):
            if session_contents[i] == '0':
                label = ''
            else:
                label = self.artists_remap[session_contents[i]]
            self.add_widget(Label(text=label, size_hint=(0.5, 0.2)))

            label = MyLabel(
                    attn_text="{:.3f}".format(float(attn_weight_values[i])),
                    color=str(normalized_weights[i]),
                    pos=(20, 20),
                    size_hint=(0.3, 0.2))
            self.add_widget(label)

        if show_sub_count:
            attn_weight_values, normalized_weights, labels =  (list(reversed(list(t))) for t in zip(*sorted(zip(attn_weight_values, normalized_weights, labels))))  


            self.clear_widgets()
            self.add_widget(Label(text="Event", size_hint=(0.5, 0.4)))
            self.add_widget(Label(text="Attn", size_hint=(0.3, 0.4)))
            self.add_widget(Label(text="Sub Count", size_hint=(0.4, 0.4)))
            for i in range(16):
                if session_contents[i] == '0':
                    label = ''
                else:
                    label = labels[i]
                self.add_widget(Label(text=label, size_hint=(0.5, 0.2)))

                label = MyLabel(
                    attn_text="{:.3f}".format(float(attn_weight_values[i])),
                    color=str(normalized_weights[i]),
                    pos=(20, 20),
                    size_hint=(0.3, 0.2))
                self.add_widget(label)

                if show_sub_count:
                    if session_contents[i] == '0':
                        self.add_widget(Label(text=str(0), size_hint=(0.4, 0.2)))
                    else:
                        label = subcount[labels[i]]
                        self.add_widget(Label(text=str(label), size_hint=(0.4, 0.2)))



        return found_search_label or search_label == ""

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        search_label = "AskReddit"
        if keycode[1] == 'w':
            while True:
                self.current_index -= 1
                found_search_label = self.visualize(search_label, False)
                if found_search_label:
                    break
        elif keycode[1] == 's':
            while True:
                self.current_index += 1
                found_search_label = self.visualize(search_label, False)
                if found_search_label:
                    break

        elif keycode[1] == 'd':
            found_search_label = self.visualize(search_label, True)
        return True


class MyApp(App):

    def build(self):
        return Visualizer()


if __name__ == '__main__':
    MyApp().run()