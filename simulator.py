import numpy as np
from matplotlib import pyplot as plt
from utils import loadAndResizeImages2, preprocess_size
from skimage.transform import rotate

class Sim(object):
    def __init__(self, dir_with_src_images='data/generated/', base_image_filename='median_image.png'
                 , object_image_list=['circle-red.png', 'robo-green.png'], img_shape=(64, 64, 3), max_iter=20):
        # load images
        self.base_image = loadAndResizeImages2(dir_with_src_images, [base_image_filename])[0]
        self.objects = loadAndResizeImages2(dir_with_src_images, object_image_list, load_alpha=True)
        self.h, self.w, self.ch = img_shape

        self.max_iterations = max_iter +1
        self.iter = 0

        self.size_factor = 3.0

        self.images_history = np.zeros((self.max_iterations, self.h, self.w, self.ch), dtype=np.float32)
        self.actions_history = np.zeros((self.max_iterations, 1), dtype=np.float32)

        self.prepare_simulator()

        self.L = self.wn // 2  # wn // 2 # Desplazamiento
        self.L_reward = 12.0#self.wn//2 # Distancia a reward

        self.reward = False

    def prepare_simulator(self):
        # resize to desired size
        orig_h, orig_w, _ = self.base_image.shape
        self.ratio_h = orig_h / self.h
        self.ratio_w = orig_w / self.w

        self.base_image = preprocess_size(self.base_image, (self.h, self.w))
        # imwrite("tmp/{}.png".format("imagen_base"), base_image)
        self.resized_objects = []
        self.objects_pos = []
        self.rotated_objs = [None] * len(self.objects)
        for o in self.objects:
            ho, wo, cho = o.shape
            if ho == wo:
                hn = int((ho / self.ratio_w) * self.size_factor)
                self.wn = int((wo / self.ratio_w) * self.size_factor)
            else:
                hn = int((ho / self.ratio_h) * self.size_factor)
                self.wn = int((wo / self.ratio_w) * self.size_factor)
            resized_o = preprocess_size(o, (hn, self.wn))
            # imwrite("tmp/{}.png".format("resized_object"), resized_o)
            self.resized_objects.append(resized_o)

            x = np.random.randint(low=0, high=self.w - self.wn)  # +wo
            y = np.random.randint(low=(60 / self.ratio_h), high=self.h - hn - (30 / self.ratio_h))
            self.objects_pos.append((x, y))

    def apply_action(self, action):
        robot_object_distance = self.h + self.w
        for ix, o in enumerate(self.resized_objects):
            if ix == 1:  # only do this for the robot and not the object
                x, y = self.objects_pos[ix]
                x_t, y_t = self.objects_pos[ix - 1]

                o_rot = self.random_rotation(o, action - 90)  # +90 since robot is sideways
                ho, wo, cho = o_rot.shape

                # this is new position of the robot
                self.actions_history[self.iter] = action # / 180  # np.sin(a), np.cos(a) # action / 180

                action = np.radians(action)  # need to convert to radians before using sin and cos
                x = x + self.L * np.sin(action)
                y = y + self.L * np.cos(action)

                x = int(np.round(x))
                y = int(np.round(y))

                xg = x - (wo // 2)
                yg = y - (ho // 2)

                if xg + wo > self.w:
                    xg = self.w - wo
                if yg + ho > self.h:
                    yg = self.h - ho
                if xg < 0:
                    xg = 0
                if yg < 0:
                    yg = 0

                x = xg + (wo // 2)
                y = yg + (ho // 2)

                self.objects_pos[ix] = (x, y)
                self.rotated_objs[ix] = o_rot

                robot_object_distance = np.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
                # print("robot dist / L: {} / {}".format(robot_object_distance, L))

        if robot_object_distance < self.L_reward:#self.L:
            self.reward = True
        self.save_image()

    @staticmethod
    def random_rotation(image_array: np.ndarray, angle=None):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        if angle is None:
            angle = np.random.uniform(-180, 180)
        return rotate(image_array, angle)

    def restart_scenario(self):
        self.reward = False
        for ix, o in enumerate(self.resized_objects):
            a = np.random.uniform(-180, 180)
            o_rot = self.random_rotation(o, angle=a - 90)  # +90 since robot is sideways

            ho, wo, cho = o_rot.shape
            x = np.random.randint(low=0, high=self.w - wo)  # +wo
            # print((100 / ratio_h))
            # 30 is the magic number to limit the random placement of objects inside image
            # y = np.random.randint(low=(60 / self.ratio_h), high=self.h - ho - (30 / self.ratio_h))
            y = np.random.randint(low=15, high=self.h - ho - (30 / self.ratio_h))

            xg = x - (wo // 2)
            yg = y - (ho // 2)

            if xg + wo > self.w:
                xg = self.w - wo
            if yg + ho > self.h:
                yg = self.h - ho
            if xg < 0:
                xg = 0
            if yg < 0:
                yg = 0

            x = xg + (wo // 2)
            y = yg + (ho // 2)

            self.objects_pos[ix] = (x, y)
            self.rotated_objs[ix] = o_rot
        self.save_image()

    def save_image(self):
        np.copyto(self.images_history[self.iter], self.base_image)
        for ix, o in enumerate(self.resized_objects):
            x, y = self.objects_pos[ix]
            o_rot = self.rotated_objs[ix]
            ho, wo, cho = o_rot.shape
            mask = o_rot[:, :, 3]  # / 255.0
            # print(mask.max(), mask.min())
            # print(x, y, ho, wo, cho)
            xg = x - (wo // 2)
            yg = y - (ho // 2)

            self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 0] = self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 0] * (
                    1 - mask) + mask * o_rot[:, :, 0]  # *255.0
            self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 1] = self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 1] * (
                    1 - mask) + mask * o_rot[:, :, 1]  # *255.0
            self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 2] = self.images_history[self.iter][yg:yg + ho, xg:xg + wo, 2] * (
                    1 - mask) + mask * o_rot[:, :, 2]  # *255.0
        self.iter += 1

    def show_image(self, iteration):
        fig = plt.figure()  # 20,4 if 10 imgs
        # display original
        ax = plt.subplot(1, 1, 1)
        plt.yticks([])
        plt.imshow(self.images_history[iteration])  # .reshape(img_dim, img_dim)
        plt.gray()
        ax.get_xaxis().set_visible(True)

    # def plot_traces(self):

