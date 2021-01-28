import json
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

class Label_Tester:

    def __init__(self):
        self.values = dict()
        self.controlling_dict = {
            "delete": False, "valid_key": False, "skip": False,
            "abort": False, "clicked": False, "number": None,
            "x_1": None, "y_1":None, "x_2": None, "y_2": None
        }

    def check_labels(self):
        del_session_index = list()
        del_img_index = list()
        _path_json = os.path.join(self.json_path, self.json_file_name)
        if os.path.isfile(_path_json):
            with open(_path_json, "r") as f:
                self.values = json.load(f)
        print(
            "w to save\na to skip image\nr to skip session\ne to close\nq to "
            "delete label\nt to mark image as useless"
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
                print("Path to images is dir: {}".format(os.path.isdir(
                    self.path_to_images
                )))
                print("Image is file: {}".format(os.path.isdir(os.path.join(
                    self.path_to_images
                ))))
                continue
            
            for j, image in enumerate(images):
                if image in self.values[session] and self.values[session][image] is None:
                    continue
                if self.controlling_dict["skip"]:
                    self.controlling_dict["skip"] = False
                    break
                if self.controlling_dict["abort"]:
                    print("Exiting...")
                    sys.exit()
                self.controlling_dict["valid_key"] = False
                while not self.controlling_dict["valid_key"]:
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
                            print('Image size does not align with specified '
                                  'one - using computed one..\nSpecified by '
                                  'user:{} but found:{}'.format(
                                self.image_size, (width, height)))
                            self.image_size = (width, height)
                        else:
                            # witch them
                            tmp1 = self.image_size[1]
                            tmp0 = self.image_size[0]
                            self.image_size = (tmp1, tmp0)

                    fig_name = '{} {} -- Session {} von {} und Aufnahme {} von {}'.format(
                        session, image, i, len(sessions), j, len(images)
                    )

                    def onkey(event):
                        if event.key not in self.activated_keys:
                            print(f"Unknown key! {event.key}")
                            return
                        self.controlling_dict["valid_key"] = True
                        if event.key == "w":
                            # save
                            plt.close()
                            self.save_dict()
                        elif event.key == "t":
                            # do not use image
                            self.values[session][image] = None
                            plt.close()
                            self.save_dict()
                        elif event.key == "a":
                            # skip image
                            # self._del_entry(session, image)
                            plt.close()
                            self.save_dict()
                        elif event.key == "r":
                            # skip session
                            # self._del_entry(session, image)
                            self.controlling_dict["skip"] = True
                        elif event.key == "e":
                            # exit
                            plt.close()
                            # self._del_entry(session, image)
                            self.save_dict()
                            # self.print_deleted_imgs(del_session_index, del_img_index)
                            print("Written everything to {}".format(
                                os.path.join(
                                    self.json_path,
                                    self.json_file_name
                                )
                            ))
                            sys.exit()
                        elif event.key == "q":
                            # delete label
                            plt.close()
                            del_session_index.append(session)
                            del_img_index.append(image)
                            self._del_entry(session, image)
                            self.controlling_dict["clicked"] = False
                        elif event.key == '0':
                            self.controlling_dict['number'] = 0
                            plt.close()
                            self.add_values_to_dict(session, image)
                            self.save_dict()
                        elif event.key == '1':
                            self.controlling_dict['number'] = 1
                            plt.close()
                            self.add_values_to_dict(session, image)
                            self.save_dict()


                    fig = plt.figure(fig_name)
                    plt.imshow(img)

                    # if len(self.values[session][image]) == 5:
                    if self.values[session][image] is not None:
                        self.controlling_dict["x_1"] = self.values[session][image][0]
                        self.controlling_dict["y_1"] = self.values[session][image][1]
                        self.controlling_dict["x_2"] = self.values[session][image][2]
                        self.controlling_dict["y_2"] = self.values[session][image][3]
                        
                        plt.scatter(
                            self.controlling_dict["x_1"],
                            self.controlling_dict["y_1"],
                            color="r"
                        )
                        plt.scatter(
                            self.controlling_dict["x_2"],
                            self.controlling_dict["y_2"], 
                            color="r"
                        )

                    self.plot_additional_content_before_first_click(fig)
                    fig.canvas.mpl_connect("key_press_event", onkey)

                    fig.canvas.mpl_connect("key_event", onkey)
                    plt.grid(True)
                    wm = plt.get_current_fig_manager()
                    wm.window.state("zoomed")
                    plt.show()


    def _del_entry(self, session, image):
        try:
            del self.values[session][image]
        except:
            pass

    def add_values_to_dict(self, session, image):
        self.values[session][image] = (
            self.controlling_dict['x_1'],
            self.controlling_dict['y_1'],
            self.controlling_dict['x_2'],
            self.controlling_dict['y_2']
        )

    def save_dict(self):
        if not os.path.isdir(self.json_path):
            os.mkdir(self.json_path)
        path = os.path.join(self.json_path, self.json_file_name)
        with open(path, 'w') as f:
            json.dump(self.values, f)

    """def print_deleted_imgs(self, indices, imgs):
        for i in len(imgs):
            print(f"Deleted label for image: {imgs[i]} in session {indices[i]}.")"""

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