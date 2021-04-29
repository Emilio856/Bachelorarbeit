"""
Helper class for labeling augmented images
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

class Label_Augmented_Imgs:

    def __init__(self):
        self.values = dict()
        self.controlling_dict = {
            "x_1": None, "y_1": None, "x_2": None, "y_2": None,
            "skip": False, "abort": False, "number": None,
            "press_valid_key": False
        }
    
    def label_augmented(self):
        _path_json = os.path.join(self.json_path, self.json_file_name)
        if os.path.isfile(_path_json):
            print("Found existing labels file - extending this one.")
            with open(_path_json, "r") as f:
                self.values = json.load(f)
            
        print("Opening combined labels...")
        with open(self.json_labeled_imgs) as f:
            combined_labels = json.load(f)
        print("Loaded combined labels")

        print(
            "w to save\na to skip image\nr to skip session\ne to close"
        )
        self.print_additional_instructions()

        sessions_tmp = sorted(os.listdir(self.path_to_images))

        sessions = []
        for session in sessions_tmp:
            if self.folders_to_check != [] and session not in self.folders_to_check:
                continue
            sessions.append(session)
        
        for i, session in enumerate(sessions):
            if self.folders_to_check != [] and session not in self.folders_to_check:
                continue
            if session not in self.values:
                self.values[session] = dict()
            try:
                images = sorted([
                    img for img in os.listdir(
                        os.path.join(self.path_to_images, session)
                    )
                    if ".png" in img
                ])

            except NotADirectoryError as nade:
                print(nade)
                print(str(nade))
                print(f"Path to images is dir: {os.path.isdir(self.path_to_images)}")
                print("Image is file: {}".format(os.path.isdir(os.path.join(
                    self.path_to_images, sessions))))
                continue

            for j, image in enumerate(images):
                if image in self.values[session]:
                    continue
                if self.controlling_dict["skip"]:
                    self.controlling_dict["skip"] = False
                    break
                if self.controlling_dict["abort"]:
                    print("Exiting...")
                    sys.exit()
                self.controlling_dict["press_valid_key"] = False
                while not self.controlling_dict["press_valid_key"]:
                    img = Image.open(
                        os.path.join(
                            self.path_to_images, session, image
                        )
                    )
                    img.load()

                    # check img size
                    width, height = img.size   # 0 = width, 1 = height
                    if width != self.image_size[0] or height != \
                            self.image_size[1]:
                        if width != self.image_size[1] or height != \
                                self.image_size[0]:
                            print("Image size does not align with specified "
                                  "one - using computed one..\nSpecified by "
                                  "user: {} but found {}".format(
                                self.image_size, (width, height)))
                            self.image_size = (width, height)
                        else:
                            # witch them
                            tmp1 = self.image_size[1]
                            tmp0 = self.image_size[0]
                            self.image_size = (tmp1, tmp0)
                    
                    fig_name = "{} {} -- Session {} von {} und Aufnahme {} von {}".format(
                        session, image, i, len(sessions), j, len(images)
                    )

                    def onkey(event):
                        if event.key not in self.activated_keys:
                            print("Unknown key! {}".format(event.key))
                            return
                        self.controlling_dict["press_valid_key"] = True
                        if event.key == "w":
                            # save
                            self.add_values_to_dict(
                                session,
                                image,
                                labeled_x1,
                                labeled_y1,
                                labeled_x2,
                                labeled_y2
                            )
                            plt.close()
                            self.save_dict()
                        elif event.key == "a":
                            # skip image
                            self._del_entry(session, image)
                            plt.close()
                            self.save_dict()
                        elif event.key == "r":
                            # skip session
                            self._del_entry(session, image)
                            self.controlling_dict["skip"] = True
                            plt.close()
                            self.save_dict()
                        elif event.key == "e":
                            # exit
                            plt.close()
                            # self._del_entry(session, image)
                            self.save_dict()
                            print("Written everything to {}". format(
                                os.path.join(
                                    self.json_path,
                                    self.json_file_name
                                )
                            ))
                            sys.exit()
                        elif event.key == "0":
                            self.controlling_dict["number"] = 0
                            plt.close()
                            self.add_values_to_dict(
                                session,
                                image,
                                labeled_x1,
                                labeled_y1,
                                labeled_x2,
                                labeled_y2
                            )
                            self.save_dict()
                        elif event.key == "1":
                            self.controlling_dict["number"] = 1
                            plt.close()
                            self.add_values_to_dict(
                                session,
                                image,
                                labeled_x1,
                                labeled_y1,
                                labeled_x2,
                                labeled_y2
                            )
                            self.save_dict()
                    
                    
                    # Assign values from json file
                    labeled_x1 = combined_labels[session][image][0]
                    labeled_y1 = combined_labels[session][image][1]
                    labeled_x2 = combined_labels[session][image][2]
                    labeled_y2 = combined_labels[session][image][3]

                    fig = plt.figure(fig_name)
                    normal_img = img.rotate(180)
                    rotated_img = img
                    axes = []

                    axes.append(fig.add_subplot(1,2,1))
                    plt.imshow(normal_img)
                    plt.scatter(
                        labeled_x1,
                        labeled_y1,
                        color="r"
                    )
                    plt.scatter(
                        labeled_x2,
                        labeled_y2,
                        color="r"
                    )
                    plt.grid(True)
                    self.plot_additional_content_before_first_click(fig)

                    axes.append(fig.add_subplot(1,2,2))
                    plt.imshow(rotated_img)
                    plt.grid(True)
                    self.plot_additional_content_before_first_click(fig)

                    self.plot_additional_content_before_first_click(fig)
                    fig.canvas.mpl_connect("key_press_event", onkey)

                    fig.canvas.mpl_connect("key_event", onkey)
                    # plt.grid(True)
                    wm = plt.get_current_fig_manager()
                    wm.window.state("zoomed")
                    plt.show()

    
    def _del_entry(self, session, image):
        try:
            del self.values[session][image]
        except:
            pass

    def add_values_to_dict(self, session, image, x1, y1, x2, y2):
        self.values[session][image] = (
            x1, y1, x2, y2
        )
    def save_dict(self):
        if not os.path.isdir(self.json_path):
            os.mkdir(self.json_path)
        path = os.path.join(self.json_path, self.json_file_name)
        with open(path, 'w') as f:
            json.dump(self.values, f)

    def plot_additional_content_before_first_click(self, figure):
        raise NotImplementedError()

    def plot_additional_content_first_click(self, figure):
        raise NotImplementedError()

    def plot_additional_content_second_click(self, figure):
        raise NotImplementedError()

    def print_additional_instructions(self):
        raise NotImplementedError()

    @property
    def path_to_images(self):
        raise NotImplementedError()

    @property
    def json_path(self):
        raise NotImplementedError()

    @property
    def json_file_name(self):
        raise NotImplementedError()

    @property
    def image_size(self):
        raise NotImplementedError()

    @property
    def activated_keys(self):
        raise NotImplementedError()

    @property
    def folders_to_check(self):
        raise NotImplementedError()