from math import sqrt

import matplotlib.pyplot as plt
from LabelsTest import Label_Tester


class Tester(Label_Tester):
    #path_to_images = '\\\\os.lsdf.kit.edu\\itiv-projects\\Stents4Tomorrow' \
    #                 '\\Data\\2020-12-03\\Debayered_images_cropped'
    path_to_images = "K:\\Data\\2020-12-03\\Debayered_images_cropped"
    json_path = '..\\data'
    json_file_name = 'pitch_length_labels.json'
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


if __name__ == '__main__':
    labler = Tester()
    labler.check_labels()