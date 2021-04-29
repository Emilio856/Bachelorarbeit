"""
Code was given and was modified to add two more funtions: visualize labels
and label augmented images
"""

from math import sqrt

import matplotlib.pyplot as plt
from abstract_labler import Abstract_Labler
from visualize_labels import Label_Tester
from augment_labels import Label_Augmented_Imgs


class Pitch_Length_Labler(Abstract_Labler):
    #path_to_images = '\\\\os.lsdf.kit.edu\\itiv-projects\\Stents4Tomorrow' \
    #                 '\\Data\\2020-12-03\\Debayered_images_cropped'
    path_to_images = "K:\\Data\\2020-12-03\\Debayered_images_cropped"
    json_path = '..\\data'
    json_file_name = "pitch_length_labels.json"
    image_size = (450, 450)  # width, height
    activated_keys = [
        'dblclick', 'w', 'a', 't', 'r', 'e', 'q'
    ]
    folders_to_check = [
        'iteration_0_30grad_maximal_licht',
        'iteration_0_50grad_1_draht_gerissen',
        'iteration_0_60grad',
        'iteration_0_70grad', 'iteration_30grad_3',
        'iteration_40grad_1_draht_gerissen',
        'iteration_40grad_5_draht_gerissen',
        'iteration_50grad_3_draht_gerissen',
        'iteration_60grad_3_draht_gerissen',
        'iteration_60grad_ohne_licht',
        'iteration_70grad_1',
        'iteration_0_30grad_1_draht_gerissen'
    ]

    def __int__(self):
        super(Pitch_Length_Labler, self).__int__()

    def add_values_to_dict(self, session, image):
        dist = sqrt(
            (self.controlling_dict['x_2'] - self.controlling_dict['x_1']) **
            2 +
            (self.controlling_dict['y_2'] - self.controlling_dict['y_1']) ** 2
        )
        self.values[session][image] = (
            self.controlling_dict['x_1'],
            self.controlling_dict['y_1'],
            self.controlling_dict['x_2'],
            self.controlling_dict['y_2'],
            dist
        )

    def plot_additional_content_before_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_second_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )
        plt.plot(
            [self.controlling_dict['x_1'], self.controlling_dict['x_2']],
            [self.controlling_dict['y_1'], self.controlling_dict['y_2']],
            marker='None', linestyle='solid', color='r'
        )

    def print_additional_instructions(self):
        pass

class Tester(Label_Tester):
    #path_to_images = '\\\\os.lsdf.kit.edu\\itiv-projects\\Stents4Tomorrow' \
    #                 '\\Data\\2020-12-03\\Debayered_images_cropped'
    path_to_images = "K:\\Data\\2020-12-03\\Debayered_images_cropped"
    json_path = '..\\data'
    json_file_name = 'pitch_length_labels.json'
    image_size = (450, 450)   # width, height
    activated_keys = [
        'dblclick', 'w', 'a', 't', 'r', 'e', 'q'
    ]
    folders_to_check = [
        'iteration_0_30grad_maximal_licht',
        'iteration_0_50grad_1_draht_gerissen',
        'iteration_0_60grad',
        'iteration_0_70grad', 'iteration_30grad_3',
        'iteration_40grad_1_draht_gerissen',
        'iteration_40grad_5_draht_gerissen',
        'iteration_50grad_3_draht_gerissen',
        'iteration_60grad_3_draht_gerissen',
        'iteration_60grad_ohne_licht',
        'iteration_70grad_1',
        'iteration_0_30grad_1_draht_gerissen'
    ]

    def __int__(self):
        super(Tester, self).__int__()

    def add_values_to_dict(self, session, image):
        dist = sqrt(
            (self.controlling_dict['x_2'] - self.controlling_dict['x_1']) **
            2 +
            (self.controlling_dict['y_2'] - self.controlling_dict['y_1']) ** 2
        )
        self.values[session][image] = (
            self.controlling_dict['x_1'],
            self.controlling_dict['y_1'],
            self.controlling_dict['x_2'],
            self.controlling_dict['y_2'],
            dist
        )

    def plot_additional_content_before_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_second_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )
        plt.plot(
            [self.controlling_dict['x_1'], self.controlling_dict['x_2']],
            [self.controlling_dict['y_1'], self.controlling_dict['y_2']],
            marker='None', linestyle='solid', color='r'
        )

    def print_additional_instructions(self):
        pass

class Label_Augmented_Images(Label_Augmented_Imgs):
    #path_to_images = '\\\\os.lsdf.kit.edu\\itiv-projects\\Stents4Tomorrow' \
    #                 '\\Data\\2020-12-03\\Debayered_images_cropped'
    path_to_images = "K:\\Data\\2020-12-03\\debayered_images_cropped_augmented_180degree"
    # path_to_augmented_imgs = "K:\\Data\\2020-12-03\\debayered_images_cropped_augmented_180degree"
    json_path = '..\\data'
    # json_file_name = "combined_labels.json"
    json_file_name = "labels_augmented_imgs.json"
    json_labeled_imgs = "C:\\Users\\emili\\Desktop\\Python\\data\\combined_labels.json"
    image_size = (450, 450)   # width, height
    activated_keys = ['dblclick', 'w', 'a', 't', 'r', 'e', 'q']

    folders_to_check = [
        "iteration_0_30grad",
        "iteration_0_40grad",
        "iteration_0_50grad",
        "iteration_0_60grad",
        "iteration_0_70grad",
        "iteration_30grad",
        "iteration_30grad_2",
        "iteration_30grad_3",
        "iteration_30grad_4",
        "iteration_40grad",
        "iteration_40grad_2",
        "iteration_40grad_3",
        "iteration_40grad_4",
        "iteration_50grad",
        "iteration_50grad_1",
        "iteration_50grad_2",
        "iteration_60grad",
        "iteration_60grad_2",
        "iteration_60grad_4",
        "iteration_70grad",
        "iteration_70grad_1",
        "iteration_70grad_2",
        "iteration_70grad_3"
    ]

    def __int__(self):
        super(Tester, self).__int__()

    def add_values_to_dict(self, session, image, x1, y1, x2, y2):
        dist = sqrt(
            (x2 - x1) **
            2 +
            (y2 - y1) ** 2
        )
        self.values[session][image] = (
            x1,
            y1,
            x2,
            y2,
            dist
        )

    def plot_additional_content_before_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_first_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )

    def plot_additional_content_second_click(self, figure):
        plt.scatter(
            int(self.image_size[0] / 2),
            int(self.image_size[1] / 2),
            color='c'
        )
        plt.plot(
            [self.controlling_dict['x_1'], self.controlling_dict['x_2']],
            [self.controlling_dict['y_1'], self.controlling_dict['y_2']],
            marker='None', linestyle='solid', color='r'
        )

    def print_additional_instructions(self):
        pass


if __name__ == '__main__':
    mode = input(
        "'L': Bilder labeln\n'V': Labels visualisieren und ggf. löschen\n'A': Augmentierte Bilder labeln\n"
    )
    
    if mode == "L":
        labler = Pitch_Length_Labler()
        labler.label()
    elif mode == "V":
        labler = Tester()
        labler.check_labels()
    elif mode == "A":
        labler = Label_Augmented_Images()
        labler.label_augmented()
