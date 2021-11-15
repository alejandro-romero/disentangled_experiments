from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow
import numpy as np
import math

from skimage.transform import resize

from data_generators import brownian_data_generator, vf_data_generator, random_data_generator
from train_ae import prepare_optimizer_object
from models import make_forward_model
from world_models_vae_arch import build_vae_world_model

from utils import load_parameters

from keras_neural_network import KerasNN


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


if __name__ == "__main__":
    # Image generator config
    dir_with_src_images = Path('data/generated/')  # _simple
    base_image = 'median_image.png'  # 'median_image.png'
    object_images = ['circle-red.png', 'robo-green.png']  # circle in the first place, as robo can be drawn over it
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    do_train, do_test, interactive, include_forward_model, train_only_forward = load_param_general(parameters)
    img_shape = load_param_dataset(parameters)
    size_factor, obj_attention, back_attention = load_param_synthetic(parameters)
    num_epochs, batch_size, latent_size, conv_layers, num_filters, kernel_size, kernel_mult, residual_forward, train_without_ae, loss, opt, model_label = load_param_hyperparam(
        parameters)

    load_vf = False

    # Load VAE and FM models:
    experiment_label = 'simple_vaewm.osize-3.0.oatt-0.8.e50.bs32.lat64.c4.opt-adamw.loss-wmse'
    autoencoder, encoder, decoder_mu_log_var, decoder, latent_shape = build_vae_world_model(
        img_shape=img_shape, latent_size=latent_size,
        opt=opt, loss=loss,  # batch_size=batch_size,
        conv_layers=conv_layers, initial_filters=num_filters)
    # autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)
    autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)

    forward_model = make_forward_model([latent_size], latent_size, learn_only_difference=residual_forward)
    forward_model.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])
    # forward_model.load_weights(
    #     'trained_models/forward_model_{}_{}_corregido.h5'.format("diff" if residual_forward else "nodiff", experiment_label))
    forward_model.load_weights(
        'trained_models/forward_model_{}_{}_arqRich5.h5'.format("diff" if residual_forward else "nodiff",
                                                       experiment_label))

    # Comprobar dimension espacio latente
    lat_h, lat_w, lat_ch = check_latent_size(latent_size)

    # Create VF
    vf = KerasNN()

    # Train VF
    # #   -  Image -> VAE -> Latent space -> VF -> Evaluation
    # batch_size = 2*1000
    # fitting_generator = vf_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape,
    #                                             batch_size=batch_size)
    # valid_generator = vf_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape,
    #                                           batch_size=batch_size)
    #
    # batches_per_epoch = 100
    # num_iterations = num_epochs * batches_per_epoch
    # iterations = 0
    # history = []
    # val_history = []
    #
    # for (batch_inputs, batch_rewards), val_data in zip(fitting_generator, valid_generator):
    #
    #     batch_latent = encoder.predict(batch_inputs)
    #     # OJO: debo mezclar los datos de las trazas para entrenar la VF
    #
    #
    #     # Paso entradas a espacio latente
    #     latent = encoder.predict(batch_inputs)
    #     if len(latent.shape) > 2:
    #         bs, h, w, ch = latent.shape
    #         latent = np.reshape(latent, (bs, h * w * ch))
    #     # latent = (latent - np.min(latent))/np.ptp(latent)
    #
    #     latent_val = encoder.predict(val_data[0])
    #     if len(latent_val.shape) > 2:
    #         bs, h, w, ch = latent_val.shape
    #         latent_val = np.reshape(latent_val, (bs, h * w * ch))
    #
    #     num_epochs = 2*500#100
    #     if load_vf:
    #         vf.load_model('trained_models/vf_model_{}.h5'.format(experiment_label))
    #     else:
    #         history = vf.train(input_data=batch_latent, output_data=batch_rewards, batch_size=100, epochs=num_epochs,
    #                            validation_data=(latent_val, val_data[1]))
    #
    #         fig = plt.figure()
    #         plt.plot(history.history['loss'], color='red', label='loss')
    #         plt.plot(history.history['val_loss'], color='blue', label='val_loss')
    #         plt.legend()
    #         # plt.show()
    #         fig.savefig(
    #             'snapshots/vf_loss_{}_{}.png'.format("diff" if residual_forward else "nodiff", experiment_label),
    #             bbox_inches='tight')
    #         vf.save_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno_arqRich5.h5'.format(experiment_label))
    #
    #     # Test trained VF
    #     # Para un estado actual, debo comprobar la valoracion que me da de diferentes acciones
    #     if do_test:
    #         '''# VF input (latent space t+1)
    #         encoded_imgs = encoder.predict(x_test) # Para cada imagen Genero latent t
    #         candidate_actions =  # Genero acciones candidatas
    #         for image in encoded_imgs:
    #             for action in candidate_actions:  # Para cada accion candidata
    #                 # Meto latent t y accion candidata en FM para obtener latent t+1
    #                 # Para latent t+1 obtendo valoracion y la asocio a la accion candidata
    #             # Veo accion candidata mas valorada y compruebo si es correcto
    #         '''
    #         # Genero acciones candidatas: 5 acciones
    #         candidate_actions = np.zeros((batch_size, 1), dtype=np.float32)
    #         candidate_actions[1] = 0.5
    #         candidate_actions[2] = 1.
    #         candidate_actions[3] = -0.5
    #         candidate_actions[4] = -1.
    #         candidate_actions = np.repeat(candidate_actions, latent_size, axis=1)
    #
    #         # Creo el mismo punto de partida para las 5 acciones candidatas
    #         # batch_latent_test = batch_latent[:]
    #         batch_latent_test = np.zeros((batch_size, latent_size), dtype=np.float32)
    #         for i in range(5):
    #             batch_latent_test[i] = batch_latent[0]
    #
    #         # Veo la posicion en t1 despues de aplicar cada una de las acciones candidatas al punto inicial
    #         encoded_imgs_t1 = forward_model.predict([batch_latent_test[0:5], candidate_actions[0:5]])
    #         if residual_forward:
    #             print("predicting residual forward...")
    #             encoded_imgs_t1 = batch_latent_test[0:5] + ((encoded_imgs_t1 * 2) - 1)
    #
    #         # Valoro con la vf cada uno de los posibles puntos en t1
    #         vf_valuations = vf.predict(encoded_imgs_t1)
    #
    #         # Decodifico las imagenes para poder representarlas y valorar los resultados
    #         decoded_imgs_t1 = decoder.predict(encoded_imgs_t1)
    #
    #         # Muestro resultados
    #         n = 5
    #         fig = plt.figure(figsize=(int(n * 2.5), int(n * 0.5)))  # 20,4 if 10 imgs
    #         fig.suptitle('VF valuations for 5 different candidate actions', fontsize=16)
    #         max_val_pos = vf_valuations.argmax()
    #         for i in range(n):
    #             # display original
    #             ax = plt.subplot(4, n, i + 1)
    #             plt.yticks([])
    #             plt.imshow(batch_inputs[0])  # .reshape(img_dim, img_dim)
    #             plt.gray()
    #             ax.get_xaxis().set_visible(True)
    #             if len(candidate_actions[i]) > 1:
    #                 if max_val_pos == i:
    #                    max_color = 'red'
    #                 else:
    #                     max_color = 'black'
    #                 ax.set_title("action:{:1.2f}, val:{:1.4f}".format(candidate_actions[i][0] * 180,
    #                                                                       vf_valuations[i].tolist()[0]), rotation=0,
    #                                  size='large', color=max_color)
    #             ax.set_xticklabels([])
    #             if i == 0:
    #                 ax.set_ylabel("original t", rotation=90, size='large')
    #                 ax.set_yticklabels([])
    #             else:
    #                 ax.get_yaxis().set_visible(False)
    #             # display encoded - vmin and vmax are needed for scaling (otherwise single pixels are drawn as black)
    #             ax = plt.subplot(4, n, i + 1 + n)
    #             plt.yticks([])
    #             plt.imshow(
    #                 batch_latent_test[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else batch_latent_test[i].reshape(
    #                     lat_h, lat_w),
    #                 vmin=batch_latent_test.min(), vmax=batch_latent_test.max(), interpolation='nearest')
    #             plt.gray()
    #             ax.get_xaxis().set_visible(True)
    #             # ax.set_xlabel("min:{:1.2f} max:{:1.2f}".format(encoded_imgs[i].min(), encoded_imgs[i].max()), rotation=0,
    #             #               size='x-large')
    #             # ax.get_yaxis().set_visible(False)
    #             if i == 0:
    #                 ax.set_ylabel("latent t", rotation=90, size='large')
    #                 ax.set_yticklabels([])
    #             else:
    #                 ax.get_yaxis().set_visible(False)
    #             # display latent prediction
    #             ax = plt.subplot(4, n, i + 1 + 2 * n)
    #             plt.yticks([])
    #             plt.imshow(encoded_imgs_t1[i].reshape(lat_h, lat_w, lat_ch) if lat_ch == 3 else encoded_imgs_t1[
    #                 i].reshape(lat_h, lat_w),
    #                        vmin=encoded_imgs_t1.min(), vmax=encoded_imgs_t1.max(), interpolation='nearest')
    #             plt.gray()
    #             ax.get_xaxis().set_visible(True)
    #             if i == 0:
    #                 ax.set_ylabel("predicted t+1", rotation=90, size='large')
    #                 ax.set_yticklabels([])
    #             else:
    #                 ax.get_yaxis().set_visible(False)
    #             # display decoded latent space
    #             ax = plt.subplot(4, n, i + 1 + 3 * n)
    #             plt.yticks([])
    #             plt.imshow(decoded_imgs_t1[i], vmin=decoded_imgs_t1.min(), vmax=decoded_imgs_t1.max(),
    #                        interpolation='nearest')
    #             plt.gray()
    #             ax.get_xaxis().set_visible(False)
    #             # ax.get_yaxis().set_visible(False)
    #             if i == 0:
    #                 ax.set_ylabel("decoded t+1", rotation=90, size='large')
    #                 ax.set_yticklabels([])
    #             else:
    #                 ax.get_yaxis().set_visible(False)

    print("FIN")
    # LOAD VF
    # vf2 = KerasNN()
    # vf.load_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno.h5'.format(experiment_label))
    vf.load_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno_arqRich5.h5'.format(experiment_label))


    candidates = 10
    n = 100
    from simulator import Sim

    sim = Sim(max_iter=n)
    # sim.restart_scenario()
    # sim.show_image(0)
    # sim.apply_action(90)
    # sim.show_image(1)
    batch_inputs = np.zeros((1, sim.h, sim.w, sim.ch), dtype=np.float32)
    # Test VF behaviour
    # Reinicio escenario
    sim.restart_scenario()
    sim.restart_scenario() # Para generar secuencias nuevas
    # Muestro figura
    sim.show_image(0)
    sim.show_image(1)
    plt.savefig('trazas_experimento/fig_{}.png'.format(0), dpi=100)
    # For n iteraciones
    steps = 0
    intentos = 0
    exitos = 0
    for i in range(1, n):
        # Pos. actual
        actual = sim.images_history[sim.iter - 1]
        np.copyto(batch_inputs, actual)
        lat_actual = encoder.predict(batch_inputs)
        # Genero acciones candidatas
        candidate_actions = np.zeros((candidates, 1), dtype=np.float32)
        for j in range(candidates):
            action = np.random.uniform(-180, 180)
            candidate_actions[j] = action / 180.0
        candidate_actions = np.repeat(candidate_actions, latent_size, axis=1)
        # Con FM y acciones candidatas calculo posic futuras
        batch_latent = np.zeros((candidates, latent_size), dtype=np.float32)
        for j in range(candidates):
            batch_latent[j] = lat_actual

        # Veo la posicion en t1 despues de aplicar cada una de las acciones candidatas al punto inicial
        encoded_imgs_t1 = forward_model.predict([batch_latent, candidate_actions])
        if residual_forward:
            print("predicting residual forward...")
            encoded_imgs_t1 = batch_latent + ((encoded_imgs_t1 * 2) - 1)

        # Valoro con la vf cada uno de los posibles puntos en t1
        vf_valuations = vf.predict(encoded_imgs_t1)
        show_decoded = True
        if show_decoded:
            # Veo estados predichos para ver si el FM condiciona que la VF valore mal
            decoded_imgs_t1 = decoder.predict(encoded_imgs_t1)
            #         n = 5
            fig = plt.figure()  # 20,4 if 10 imgs
            fig.suptitle('Decoded images', fontsize=16)
            max_val_pos = vf_valuations.argmax()
            for i in range(candidates):
                ax = plt.subplot(2, candidates / 2, i + 1)
                plt.yticks([])
                plt.imshow(decoded_imgs_t1[i], vmin=decoded_imgs_t1.min(), vmax=decoded_imgs_t1.max(),
                           interpolation='nearest')
                plt.gray()
                ax.get_xaxis().set_visible(True)
                if max_val_pos == i:
                    max_color = 'red'
                else:
                    max_color = 'black'
                ax.set_title("action:{:1.2f}, val:{:1.4f}".format(candidate_actions[i][0] * 180,
                                                                  vf_valuations[i].tolist()[0]), rotation=0,
                             size='large', color=max_color)
                ax.set_xticklabels([])
            fig.set_size_inches((18, 11), forward=False)
            plt.savefig('trazas_experimento/fig_{}_candidates.png'.format(sim.iter - 1), dpi=200)
        # Evalua pos. futuras con VF y elijo la mejor accion
        best_action = candidate_actions[vf_valuations.argmax()][0] * 180.0
        # Aplico accion en robot real
        # Actualizo pos. actual
        sim.apply_action(best_action)
        # Muestro figura
        sim.show_image(sim.iter - 1)
        plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
        # si llego al goal reinicio el escenario
        steps += 1
        if sim.reward or steps == 10:
            if sim.reward:
                exitos += 1
            intentos += 1
            steps = 0
            sim.restart_scenario()
            sim.show_image(sim.iter - 1)
            plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
            plt.close('all')
    plt.close('all')
    print("FIN")
