import os
# os.environ["TF_USE_CUDNN"]="0"   # set CUDNN to false, since it is non-deterministic: https://github.com/keras-team/keras/issues/2479
# os.environ['PYTHONHASHSEED'] = '0'

# ensure reproducibility
# '''
from numpy.random import seed

seed(3)
import tensorflow

tensorflow.random.set_seed(5)
# '''

# includes all keras layers for building the model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K

# include demo dataset mnist
from keras.datasets import mnist
import numpy as np

# from scipy.misc import imresize, imsave #, imread, imsave
# from scipy.ndimage import imread
from skimage.transform import resize

from utils import trim, read_files, psnr, log10, preprocess_size, loadAndResizeImages2
from models import build_conv_only_ae, make_forward_model
from world_models_vae_arch import build_vae_world_model

from data_generators import data_generator, data_generator_mnist, random_data_generator, brownian_data_generator, brownian_data_generator_corregido
from utils import load_parameters, list_images_recursively

from train_ae import prepare_optimizer_object

from numpy.random import seed

seed(6)
tensorflow.random.set_seed(6)

import math
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_generators import cdist, max_cdist

def resize_images(images, dims=(8, 8, 1)):
    # print("imgm ax: ", images.max())
    resized = np.zeros((len(images), dims[0], dims[1], dims[2]), dtype=float)
    for i in range(len(images)):
        # print(images[i].shape)
        if dims[2] == 1:
            # tmp = imresize(images[i], size=(dims[0], dims[1]), interp='nearest') / 255.0
            tmp = resize(images[i], output_shape=(dims[0], dims[1]))  # / 255.0
        else:
            # tmp = imresize(images[i][:, :, 0], size=(dims[0], dims[1]), interp='bilinear') / 255.0
            tmp = resize(images[i][:, :, 0], output_shape=(dims[0], dims[1]))  # / 255.0
        #    imsave('tmp/test_{}.png'.format(i), tmp)
        if dims[2] == 1:
            resized[i][:, :, 0] = (tmp[:, :, 0]).astype(float)
        else:
            resized[i][:, :, 0] = (tmp[:, :]).astype(float)
            resized[i][:, :, 1] = (tmp[:, :]).astype(float)
            resized[i][:, :, 2] = (tmp[:, :]).astype(float)
    # exit()
    # print("imgm ax: ", resized.max(), resized.min())
    return resized


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def check_latent_size(latent_size):
    if not type(latent_size) == int:
        _, lat_h, lat_w, lat_ch = latent_size
    else:
        if int(np.sqrt(latent_size)) ** 2 == latent_size:
            lat_h, lat_w = int(np.sqrt(latent_size)), int(np.sqrt(latent_size))
            lat_ch = 1
        else:
            lat_ch = 1
            if latent_size % 3 == 0:
                lat_ch = 3
            tmp = list(divisorGenerator(latent_size // lat_ch))
            lat_h = int(tmp[len(tmp) // 2])
            lat_w = int(latent_size // (lat_h * lat_ch))
    return lat_h, lat_w, lat_ch

def load_param_general(parameters):
    do_train = eval(parameters["general"]["do_train"])
    do_test = eval(parameters["general"]["do_test"])
    interactive = eval(parameters["general"]["interactive"])
    include_forward_model = eval(parameters["general"]["include_forward_model"])
    train_only_forward = eval(parameters["general"]["train_only_forward"])
    return do_train, do_test, interactive, include_forward_model, train_only_forward

def load_param_dataset(parameters):
    img_shape = eval(parameters["dataset"]["img_shape"])
    return img_shape

def load_param_synthetic(parameters):
    size_factor = float(parameters["synthetic"]["size_factor"])
    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])
    return size_factor, obj_attention, back_attention

def load_param_hyperparam(parameters):
    num_epochs = int(parameters["hyperparam"]["num_epochs"])
    batch_size = int(parameters["hyperparam"]["batch_size"])
    latent_size = int(parameters["hyperparam"]["latent_size"])
    conv_layers = int(parameters["hyperparam"]["conv_layers"])
    num_filters = int(parameters["hyperparam"]["num_filters"])
    kernel_size = int(parameters["hyperparam"]["kernel_size"])
    kernel_mult = int(parameters["hyperparam"]["kernel_mult"])
    residual_forward = eval(parameters["hyperparam"]["residual_forward"])
    train_without_ae = eval(parameters["hyperparam"]["train_without_ae"])
    loss = (parameters["hyperparam"]["loss"])
    opt = (parameters["hyperparam"]["opt"])
    model_label = (parameters["hyperparam"]["model_label"])
    return num_epochs, batch_size, latent_size, conv_layers, num_filters, kernel_size, kernel_mult, residual_forward, train_without_ae, loss, opt, model_label

def create_fm(latent_size, residual_forward):
    forward_model = make_forward_model([latent_size], latent_size, learn_only_difference=residual_forward)
    forward_model.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])
    forward_model.summary()
    return forward_model


def evaluate_decoded_images():
    if os.environ.get('DISPLAY', '') == '':
        print('No display found. Using non-interactive Agg backend')
        mpl.use('Agg')

    # TODO: evaluate the reconstructed images

    threshold = .2
    h, w, ch = img_shape
    class_masks_R = np.zeros((batch_size, h, w, 3))
    class_masks_B = np.zeros((batch_size, h, w, 3))

    print("resized objects: ", resized_objects[0].shape)
    avg_robot1 = resized_objects[0][:, :, :3].mean(axis=(0, 1))
    avg_robot2 = resized_objects[1][:, :, :3].mean(axis=(0, 1))

    print(avg_robot1, avg_robot2)

    robot1 = np.ones((h, w, ch)) * avg_robot1
    robot2 = np.ones((h, w, ch)) * avg_robot2

    def mse(A, B):
        return ((A - B) ** 2).mean(axis=None)  # None == scalar value

    def rmse(A, B):
        return np.sqrt(mse(A, B))

    IoUs = np.zeros(batch_size)
    minRBs = np.zeros(batch_size)
    mseRs = np.zeros(batch_size)
    mseBs = np.zeros(batch_size)

    for i in range(1, x_test.shape[0]):
        original = x_test[i]
        mask = x_mask[i]  # .astype(int)

        reconstructed = decoded_imgs_t_plus_1[i - 1]
        # print(original.max(), original.min(), reconstructed.max(), reconstructed.min())
        # pass the positive mask pixels:
        # N_R = mask[(mask / obj_attention) >= .5].shape[0]
        # N_B = mask[(mask / obj_attention) < .5].shape[0]
        tmp = cdist(background, original, keepdims=True)  # color distance between images
        # tmp = (background - original) # np.abs
        juan_mask = (tmp > threshold * max_cdist).astype(float)  # ( tmp > 0 ).astype(float)
        # juan_mask = (np.abs(mask - obj_attention) < .001).astype(float) # mask for the robot
        zero_mask = (~(juan_mask).astype(bool)).astype(
            float)  # ((mask - obj_attention) < .5).astype(float) # mask for the background
        # print("robot pixels mask", mask[(mask / obj_attention) > .5].shape)
        # print("back pixels mask", mask[(mask / obj_attention) < .5].shape)
        # print(juan_mask.shape, juan_mask.max(), juan_mask.min())

        # '''
        cond_R = juan_mask[:, :, 0].astype(bool) == True
        cond_B = zero_mask[:, :, 0].astype(bool) == True
        class_masks_R[i][:, :, 1][cond_R] = \
            (cdist(juan_mask * original, juan_mask * reconstructed) < threshold * max_cdist)[cond_R]  # RR
        class_masks_R[i][:, :, 0][cond_R] = \
            (cdist(juan_mask * background, juan_mask * reconstructed) < threshold * max_cdist)[cond_R]  # RB
        class_masks_R[i][:, :, 0][cond_R] = \
            (~class_masks_R[i][:, :, 1].astype(bool) & class_masks_R[i][:, :, 0].astype(bool))[cond_R]
        class_masks_R[i][:, :, 2][cond_R] = \
            (~class_masks_R[i][:, :, 1].astype(bool) & ~class_masks_R[i][:, :, 0].astype(bool))[cond_R]  # RX

        class_masks_B[i][:, :, 1][cond_B] = \
            (cdist(zero_mask * original, zero_mask * reconstructed) < threshold * max_cdist)[cond_B]  # BB
        class_masks_B[i][:, :, 0][cond_B] = (np.minimum(cdist(zero_mask * robot1, zero_mask * reconstructed),
                                                        cdist(zero_mask * robot2,
                                                              zero_mask * reconstructed)) < threshold * max_cdist)[
            cond_B]  # B(R1,R2)
        class_masks_B[i][:, :, 0][cond_B] = \
            (~class_masks_B[i][:, :, 1].astype(bool) & class_masks_B[i][:, :, 0].astype(bool))[cond_B]

        class_masks_B[i][:, :, 2][cond_B] = \
            (~class_masks_B[i][:, :, 1].astype(bool) & ~class_masks_B[i][:, :, 0].astype(bool))[cond_B]  # BX

        N_RR = np.sum(class_masks_R[i][:, :, 1][cond_R])  # [class_masks_R[i][:, :, 0] > .5].shape[0]
        N_RB = np.sum(class_masks_R[i][:, :, 0][cond_R])
        N_RX = np.sum(class_masks_R[i][:, :, 2][cond_R])

        N_BR = np.sum(class_masks_B[i][:, :, 0][cond_B])  # [class_masks_B[i][:, :, 1] > .5].shape[0]

        if N_RR + N_RB + N_RX < 0.1:
            IoU = 0
            __R = 0
            __B = 0
        else:
            IoU = N_RR / (N_RR + N_BR + N_RB + N_RX)
            __R = N_RR / (N_RR + N_RB + N_RX)
            __B = 1.0 / ((N_BR / (N_RR + N_RB + N_RX)) + 1.0)

        minRB = min(__R, __B)
        # print("N_R: ", N_R, "N_RR: ", N_RR, " N_RB: ", N_RB, " N_RX: ", N_RX, " N_BR: ", N_BR)

        IoUs[i] = IoU
        minRBs[i] = minRB

        mseRs[i] = mse(juan_mask * original, juan_mask * reconstructed)
        mseBs[i] = mse(zero_mask * original, zero_mask * reconstructed)

        # print(class_masks_B[i].shape, class_masks_B[i].max(), class_masks_B[i].min())
        label = str(np.random.randint(0, 1000))

        '''
        imsave("tmp/{}.png".format("{}_original".format(label)), x_test[i])
        imsave("tmp/{}.png".format("{}_decoded".format(label)), decoded_imgs[i])
        imsave("tmp/{}.png".format("{}_Cmask_B".format(label)), class_masks_B[i])
        imsave("tmp/{}.png".format("{}_Cmask_R".format(label)), class_masks_R[i])
        imsave("tmp/{}.png".format("{}_Cmask_combo".format(label)), (class_masks_R[i].astype(int) + class_masks_B[i].astype(int)).astype(int))
        '''
        # if i > 5:
        #    exit()
    # print("avg. IoU: ", IoUs.mean(), " - ", IoUs.std(), "avg min(R,B): ", minRBs.mean(), " - ", minRBs.std())
    np.savetxt('snapshots/forward_{}_{}.eval'.format("diff" if residual_forward else "nodiff", experiment_label),
               [mseRs.mean(), mseRs.std(),
                mseBs.mean(), mseBs.std(),
                IoUs.mean(), IoUs.std(),
                minRBs.mean(), minRBs.std()],
               delimiter=",", fmt='%1.5f', newline=' ')

def plot_results():
    n = batch_size
    fig = plt.figure(figsize=(int(n * 2.5), int(n * 0.5)))  # 20,4 if 10 imgs
    for i in range(n):
        # display original
        ax = plt.subplot(6, n, i + 1)
        plt.yticks([])
        plt.imshow(x_test[i])  # .reshape(img_dim, img_dim)
        plt.gray()
        ax.get_xaxis().set_visible(True)
        if len(x_action[i]) > 1:
            ax.set_title("action:{:1.2f}".format(x_action[i][0]*180), rotation=0, size='large')
        else:
            ax.set_title("ac:{:1.2f}".format(x_action[i][0]), rotation=0, size='large')
        ax.set_xticklabels([])
        if i == 0:
            ax.set_ylabel("original t", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
        # display encoded - vmin and vmax are needed for scaling (otherwise single pixels are drawn as black)
        ax = plt.subplot(6, n, i + 1 + n)
        plt.yticks([])
        plt.imshow(
            encoded_imgs[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs[i].reshape(lat_h, lat_w),
            vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(True)
        # ax.set_xlabel("min:{:1.2f} max:{:1.2f}".format(encoded_imgs[i].min(), encoded_imgs[i].max()), rotation=0,
        #               size='x-large')
        # ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel("encoded t", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(6, n, i + 1 + 2 * n)
        plt.yticks([])
        plt.imshow(decoded_imgs[i], vmin=decoded_imgs[i].min(),
                   vmax=decoded_imgs[i].max())  # .reshape(img_dim, img_dim)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel("decoded t", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
        # display masks
        ax = plt.subplot(6, n, i + 1 + 3 * n)
        plt.yticks([])
        plt.imshow(encoded_imgs_t_plus_1[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs_t_plus_1[
            i].reshape(lat_h, lat_w),
                   vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(True)
        # ax.set_xlabel(
        #     "min:{:1.2f} max:{:1.2f}".format(encoded_imgs_t_plus_1[i].min(), encoded_imgs_t_plus_1[i].max()),
        #     rotation=0, size='x-large')
        # ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel("predicted t+1", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
        # display dreamed latent space
        ax = plt.subplot(6, n, i + 1 + 4 * n)
        plt.yticks([])
        plt.imshow(decoded_imgs_t_plus_1[i], vmin=decoded_imgs.min(), vmax=decoded_imgs.max(),
                   interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel("decoded t+1", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
        # display dreamed images
        try:
            ax = plt.subplot(6, n, i + 1 + 5 * n)
            plt.yticks([])
            plt.imshow(x_test[i + 1], vmin=x_test[i + 1].min(), vmax=x_test[i + 1].max())
            plt.gray()
        except:
            # plt.imshow(np.ones((2,2))*255, vmin=0, vmax=255)
            plt.gray()
        ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel("original t+1", rotation=90, size='large')
            ax.set_yticklabels([])
        else:
            ax.get_yaxis().set_visible(False)
    fig.savefig('snapshots/forward_{}_{}_bs100.pdf'.format("diff" if residual_forward else "nodiff", experiment_label),
                bbox_inches='tight')

    # n = 10
    # fig = plt.figure(figsize=(int(n * 2.5), int(n * 0.5)))  # 20,4 if 10 imgs
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(6, n, i + 1);
    #     plt.yticks([])
    #     plt.imshow(x_test[i])  # .reshape(img_dim, img_dim)
    #     plt.gray()
    #     ax.get_xaxis().set_visible(True)
    #     if len(x_action[i]) > 1:
    #         ax.set_title("action:{:1.2f}".format(x_action[i][0]*180), rotation=0, size='large')
    #     else:
    #         ax.set_title("ac:{:1.2f}".format(x_action[i][0]), rotation=0, size='large')
    #
    #     ax.set_xticklabels([])
    #
    #     if i == 0:
    #         ax.set_ylabel("original t", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)
    #
    #     # display encoded - vmin and vmax are needed for scaling (otherwise single pixels are drawn as black)
    #     ax = plt.subplot(6, n, i + 1 + n);
    #     plt.yticks([])
    #     plt.imshow(encoded_imgs[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs[i].reshape(lat_h, lat_w),
    #                vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
    #     plt.gray()
    #     ax.get_xaxis().set_visible(True)
    #     # ax.set_xlabel("min:{:1.2f} max:{:1.2f}".format(encoded_imgs[i].min(), encoded_imgs[i].max()), rotation=0,
    #     #               size='x-large')
    #     # ax.get_yaxis().set_visible(False)
    #
    #     if i == 0:
    #         ax.set_ylabel("latent t", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(6, n, i + 1 + 2 * n);
    #     plt.yticks([])
    #     plt.imshow(decoded_imgs[i], vmin=decoded_imgs[i].min(), vmax=decoded_imgs[i].max())  # .reshape(img_dim, img_dim)
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    #
    #     if i == 0:
    #         ax.set_ylabel("decoded t", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)
    #
    #     # display masks
    #     ax = plt.subplot(6, n, i + 1 + 3 * n);
    #     plt.yticks([])
    #     plt.imshow(
    #         encoded_imgs_t_plus_1[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs_t_plus_1[i].reshape(
    #             lat_h, lat_w),
    #         vmin=encoded_imgs.min(), vmax=encoded_imgs.max(), interpolation='nearest')
    #     plt.gray()
    #     ax.get_xaxis().set_visible(True)
    #     # ax.set_xlabel("min:{:1.2f} max:{:1.2f}".format(encoded_imgs_t_plus_1[i].min(), encoded_imgs_t_plus_1[i].max()),
    #     #               rotation=0, size='x-large')
    #     # ax.get_yaxis().set_visible(False)
    #
    #     if i == 0:
    #         ax.set_ylabel("predicted t+1", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)
    #
    #     # display dreamed latent space
    #     ax = plt.subplot(6, n, i + 1 + 4 * n);
    #     plt.yticks([])
    #     plt.imshow(decoded_imgs_t_plus_1[i], vmin=decoded_imgs.min(), vmax=decoded_imgs.max(), interpolation='nearest')
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    #
    #     if i == 0:
    #         ax.set_ylabel("decoded t+1", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)
    #
    #     # display dreamed images
    #     try:
    #         ax = plt.subplot(6, n, i + 1 + 5 * n);
    #         plt.yticks([])
    #         plt.imshow(x_test[i + 1], vmin=x_test[i + 1].min(), vmax=x_test[i + 1].max())
    #         plt.gray()
    #     except:
    #         # plt.imshow(np.ones((2,2))*255, vmin=0, vmax=255)
    #         plt.gray()
    #
    #     ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    #
    #     if i == 0:
    #         ax.set_ylabel("original t+1", rotation=90, size='large')
    #         ax.set_yticklabels([])
    #     else:
    #         ax.get_yaxis().set_visible(False)

    # Carga modelos:
    # experiment_label = 'simple_vaewm.osize-3.0.oatt-0.8.e50.bs32.lat64.c4.opt-adamw.loss-wmse'
    # autoencoder2, encoder2, decoder_mu_log_var2, decoder2, latent_shape2 = build_vae_world_model(
    #     img_shape=img_shape, latent_size=latent_size,
    #     opt=opt, loss=loss,  # batch_size=batch_size,
    #     conv_layers=conv_layers, initial_filters=num_filters)
    # autoencoder2.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)
    # forward_model2 = make_forward_model([latent_size], latent_size, learn_only_difference=residual_forward)
    # forward_model2.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])
    # forward_model2.summary()
    # forward_model2.load_weights(
    #     'trained_models/forward_model_{}_{}.h5'.format("diff" if residual_forward else "nodiff", experiment_label))


if __name__ == "__main__":

    # Image generator config
    dir_with_src_images = Path('data/generated/')  # _simple
    base_image = 'median_image_new.png'
    object_images = ['circle-red.png', 'robo-green.png']  # circle in the first place, as robo can be drawn over it
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    do_train, do_test, interactive, include_forward_model, train_only_forward = load_param_general(parameters)
    img_shape = load_param_dataset(parameters)
    size_factor, obj_attention, back_attention = load_param_synthetic(parameters)
    num_epochs, batch_size, latent_size, conv_layers, num_filters, kernel_size, kernel_mult, residual_forward, train_without_ae, loss, opt, model_label = load_param_hyperparam(
        parameters)

    if train_without_ae:
        model_label = 'simple_nn'

    experiment_label = "{}.osize-{}.oatt-{}.e{}.bs{}.lat{}.c{}.opt-{}.loss-{}".format(
        model_label, size_factor, obj_attention, num_epochs, batch_size,
        "{:02d}".format(latent_size) if type(latent_size) == int else 'x'.join(map(str, latent_size[1:])),
        conv_layers, opt, loss)

    # vae from world models
    if 'vaewm' in model_label:
        autoencoder, encoder, decoder_mu_log_var, decoder, latent_shape = build_vae_world_model(
            img_shape=img_shape, latent_size=latent_size,
            opt=opt, loss=loss,  # batch_size=batch_size,
            conv_layers=conv_layers, initial_filters=num_filters)  # , kernel_size=kernel_size, kernel_mult=kernel_mult)
    # ae
    elif 'ae_conv' in model_label:
        autoencoder, encoder, decoder, latent_size = build_conv_only_ae(
            img_shape=img_shape, latent_size=latent_size,
            opt=opt, loss=loss,
            conv_layers=conv_layers, initial_filters=num_filters)  # , kernel_size=kernel_size, kernel_mult=kernel_mult)

    if not train_without_ae:
        print("Preloading weights from previous model: {}".format(experiment_label))
        autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)

    # Comprobar dimension espacio latente
    lat_h, lat_w, lat_ch = check_latent_size(latent_size)

    # Creo forward model
    forward_model = create_fm(latent_size, residual_forward)

    do_train = True
    if do_train:
        # batch_size = 100
        fitting_generator = brownian_data_generator_corregido(dir_with_src_images, base_image, object_images, img_shape=img_shape,
                                                    batch_size=batch_size + 1)
        valid_generator = brownian_data_generator_corregido(dir_with_src_images, base_image, object_images, img_shape=img_shape,
                                                  batch_size=batch_size + 1)
        batches_per_epoch = 2*100
        num_iterations = num_epochs * batches_per_epoch
        iterations = 0
        history = []
        val_history = []
        with tqdm(total=num_iterations) as pbar:
            pbar.set_description("loss: %f" % -1.0)
            for ([batch_inputs, batch_masks, batch_actions], batch_outputs), val in zip(fitting_generator,
                                                                                        valid_generator):
                if iterations >= num_iterations:
                    pbar.close()
                    print("over the iteration limit, stopping")
                    break
                if not train_without_ae:
                    latent = encoder.predict(batch_inputs)
                else:
                    latent = resize_images(batch_inputs)
                if len(latent.shape) > 2:
                    bs, h, w, ch = latent.shape
                    latent = np.reshape(latent, (bs, h * w * ch))
                # latent = (latent - np.min(latent))/np.ptp(latent)
                latent_t = latent[0:batch_size]
                if not residual_forward:
                    latent_t_plus_1 = latent[1:1 + batch_size]
                else:
                    latent_t_plus_1 = ((latent[1:1 + batch_size] - latent_t) + 1.0) / 2.0  # already normalized
                batch_actions = batch_actions[1:1 + batch_size]
                batch_actions = np.repeat(batch_actions, latent_size, axis=1)  # balance the actions with latent_size
                result = forward_model.train_on_batch([latent_t, batch_actions], latent_t_plus_1)
                ([batch_inputs, batch_masks, batch_actions], batch_outputs) = val
                if not train_without_ae:
                    latent = encoder.predict(batch_inputs)
                else:
                    latent = resize_images(batch_inputs)
                if len(latent.shape) > 2:
                    bs, h, w, ch = latent.shape
                    latent = np.reshape(latent, (bs, h * w * ch))
                # latent = (latent - np.min(latent))/np.ptp(latent)
                latent_t = latent[0:batch_size]
                if not residual_forward:
                    latent_t_plus_1 = latent[1:1 + batch_size]
                else:
                    latent_t_plus_1 = ((latent[1:1 + batch_size] - latent_t) + 1.0) / 2.0
                batch_actions = batch_actions[1:1 + batch_size]
                batch_actions = np.repeat(batch_actions, latent_size, axis=1)
                val_result = forward_model.test_on_batch([latent_t, batch_actions], latent_t_plus_1)
                history.append(result[0])
                val_history.append(val_result[0])
                iterations += 1
                pbar.set_description(
                    "loss: %f, val_loss: %f, lr: %f" % (result[0], val_result[0], K.eval(forward_model.optimizer.lr)))
                pbar.update(1)
        fig = plt.figure()
        plt.plot(history, color='red')
        plt.plot(val_history, color='blue')
        # plt.show()
        fig.savefig(
            'snapshots/forward_loss_{}_{}_arqRich6.png'.format("diff" if residual_forward else "nodiff", experiment_label),
            bbox_inches='tight')
        forward_model.save(
            'trained_models/forward_model_{}_{}_arqRich6.h5'.format("diff" if residual_forward else "nodiff", experiment_label))

    if do_test:
        seed(9)
        tensorflow.random.set_seed(9)
        batch_size = 64
        forward_model.load_weights(
            'trained_models/forward_model_{}_{}_arqRich6.h5'.format("diff" if residual_forward else "nodiff", experiment_label))
        resized_objects = []
        # test the trained forward model:
        valid_generator = brownian_data_generator_corregido(dir_with_src_images, base_image, object_images,
                                                  img_shape=img_shape, batch_size=batch_size + 1,
                                                  resized_objects=resized_objects)
        h, w, ch = img_shape
        x_test = np.zeros((batch_size + 1, h, w, ch))
        x_mask = np.zeros((batch_size + 1, h, w, ch))
        x_action = np.zeros((batch_size + 1, 1))
        [x_test, x_mask, x_action], out = valid_generator.__next__()
        x_test = x_test[0:batch_size]
        x_action = x_action[1:1 + batch_size]
        x_action = np.repeat(x_action, latent_size, axis=1)
        print(x_action[1])
        back_generator = brownian_data_generator_corregido(dir_with_src_images, base_image, object_images, img_shape=img_shape,
                                                 batch_size=batch_size + 1)
        [batch_inputs, batch_masks, batch_actions], batch_outputs = next(back_generator)
        background = np.median(batch_outputs, axis=0, keepdims=False)
        # decoded_imgs = autoencoder.predict(x_test)
        print("input images shape: ", x_test.shape)
        # encoded_imgs = encoder.predict(x_test)
        if not train_without_ae:
            encoded_imgs = encoder.predict(x_test)
        else:
            encoded_imgs = resize_images(x_test)
        if len(encoded_imgs.shape) > 2:
            bs, h, w, ch = encoded_imgs.shape
            tmp = np.reshape(encoded_imgs, (bs, h * w * ch))
            encoded_imgs_t_plus_1 = forward_model.predict([tmp, x_action])
            if residual_forward:
                print("predicting residual forward...")
                encoded_imgs_t_plus_1 = tmp + ((encoded_imgs_t_plus_1 * 2) - 1)
        else:
            encoded_imgs_t_plus_1 = forward_model.predict([encoded_imgs, x_action])
            if residual_forward:
                print("predicting residual forward...")
                encoded_imgs_t_plus_1 = encoded_imgs + ((encoded_imgs_t_plus_1 * 2) - 1)

        if type(encoded_imgs_t_plus_1) == list:
            encoded_imgs_t_plus_1 = encoded_imgs_t_plus_1[-1]
        print("encoded images shape: ", encoded_imgs.shape)  # , encoded_imgs[1].shape, encoded_imgs[2].shape)
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        # normalize before displaying
        # encoded_imgs = (encoded_imgs - encoded_imgs.min()) / np.ptp(encoded_imgs) * 255.0
        # print(encoded_imgs)
        # decoded_imgs = decoder.predict(encoded_imgs)
        if not train_without_ae:
            decoded_imgs = decoder.predict(encoded_imgs)
        else:
            decoded_imgs = x_test
            # decoded_imgs_t_plus_1 = np.zeros(decoded_imgs.shape)
            if len(encoded_imgs.shape) > 2:
                bs, h, w, ch = encoded_imgs.shape
                tmp = np.reshape(encoded_imgs_t_plus_1, (bs, h, w, ch))
                print(encoded_imgs.shape, encoded_imgs_t_plus_1.shape)
                decoded_imgs_t_plus_1 = resize_images(tmp, dims=(64, 64, 3))
            else:
                print(encoded_imgs.shape, encoded_imgs_t_plus_1.shape)
                decoded_imgs_t_plus_1 = resize_images(encoded_imgs_t_plus_1, dims=(64, 64, 3))
                # decoded_imgs_t_plus_1[0:-1] = x_test[1:]
            # decoded_imgs_t_plus_1[-1] = x_test[0]
            # print(len(x_test), len(decoded_imgs), len(decoded_imgs_t_plus_1))
            # exit()
        if not train_without_ae:
            if len(encoded_imgs.shape) > 2:
                bs, h, w, ch = encoded_imgs.shape
                tmp = np.reshape(encoded_imgs_t_plus_1, (bs, h, w, ch))
                decoded_imgs_t_plus_1 = decoder.predict(tmp)
            else:
                decoded_imgs_t_plus_1 = decoder.predict(encoded_imgs_t_plus_1)
        print(decoded_imgs.shape)
        print(decoded_imgs_t_plus_1.shape)
        print("h, w, ch: {},{},{}".format(lat_h, lat_w, lat_ch))
        print("encoded MAX / MIN: ", encoded_imgs.max(), " / ", encoded_imgs.min())
        print("input images shape: ", x_test.shape)
        print("decoded_imgs images shape: ", decoded_imgs.shape)
        # latent_dreams = encoder.predict(decoded_imgs)
        # if include_forward_model:
        # latent_dreams = forward_model.predict([latent_dreams, x_action])
        # dreams = decoder.predict(latent_dreams) # dream the images
        print("decoded images shape: ", decoded_imgs.shape)


        evaluate_decoded_images()

        plot_results()

        print("FIN")
