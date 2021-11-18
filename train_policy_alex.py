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

def change_range(old_value, old_max, old_min, new_max, new_min):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value

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
        'trained_models/forward_model_{}_{}.h5'.format("diff" if residual_forward else "nodiff",
                                                       experiment_label))

    # Comprobar dimension espacio latente
    lat_h, lat_w, lat_ch = check_latent_size(latent_size)

    # Create VF
    vf = KerasNN()


    # LOAD VF
    # vf.load_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno.h5'.format(experiment_label))
    vf.load_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno.h5'.format(experiment_label))


    candidates = 10
    n = 7500#5000
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
    # sim.restart_scenario() # Para generar secuencias nuevas
    # Muestro figura
    # sim.show_image(0)
    # sim.show_image(1)
    # plt.savefig('trazas_experimento/fig_{}.png'.format(0), dpi=100)
    # For n iteraciones
    steps = 0
    intentos = 0
    exitos = 0
    latent_t_array = []
    action_t_array = []
    trace_latent_t = []
    trace_action_t = []
    pos_rob_array = []
    pos_obj_array = []
    trace_pos_rob = []
    trace_pos_obj = []
    for i in range(1, n):
        if i%100==0:
            print("Iterations: ", i)
        # Pos. actual
        actual = sim.images_history[sim.iter - 1]
        np.copyto(batch_inputs, actual)
        lat_actual = encoder.predict(batch_inputs)
        trace_latent_t.append(lat_actual)
        trace_pos_obj.append(sim.objects_pos[0])
        trace_pos_rob.append(sim.objects_pos[1])
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
            # print("predicting residual forward...")
            encoded_imgs_t1 = batch_latent + ((encoded_imgs_t1 * 2) - 1)

        # Valoro con la vf cada uno de los posibles puntos en t1
        vf_valuations = vf.predict(encoded_imgs_t1)
        show_decoded = False
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
        trace_action_t.append(best_action)
        # Aplico accion en robot real
        # Actualizo pos. actual
        sim.apply_action(best_action)
        # Muestro figura
        # sim.show_image(sim.iter - 1)
        # plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
        # si llego al goal reinicio el escenario
        steps += 1
        if sim.reward or steps == 10:
            if sim.reward:
                exitos += 1
                latent_t_array.append(trace_latent_t)
                action_t_array.append(trace_action_t)
                pos_obj_array.append(trace_pos_obj)
                pos_rob_array.append(trace_pos_rob)
            trace_latent_t = []
            trace_action_t = []
            trace_pos_obj = []
            trace_pos_rob = []
            intentos += 1
            steps = 0
            sim.restart_scenario()
            # sim.show_image(sim.iter - 1)
            # plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
            plt.close('all')
    plt.close('all')
    print(f"Intentos: {intentos} / Exitos: {exitos}. % exito: {exitos*100/intentos:.2f}%")

    total = 0
    for i in range(len(latent_t_array)):
        total += len(latent_t_array[i])
    print("Total datos:", total)

    # Convierto matrices en un array de una dimension
    in_data = []
    for i in range(len(latent_t_array)):
        for j in range(len(latent_t_array[i])):
            in_data.append(latent_t_array[i][j][0])
    in_data = np.array(in_data)
    in_data1, in_data2, in_data3, in_data_valid = np.split(in_data, 4)
    in_data_train = np.concatenate((in_data1, in_data2, in_data3))

    out_data = []
    for i in range(len(action_t_array)):
        for j in range(len(action_t_array[i])):
            out_data.append(action_t_array[i][j])
    out_data = np.array(out_data)/180
    out_data1, out_data2, out_data3, out_data_valid = np.split(out_data, 4)
    out_data_train = np.concatenate((out_data1, out_data2, out_data3))

    # Convierto valores salida a rango 0-1
    for i in range(len(out_data_train)):
        out_data_train[i] = change_range(out_data_train[i], 1, -1, 1, 0)
    for i in range(len(out_data_valid)):
        out_data_valid[i] = change_range(out_data_valid[i], 1, -1, 1, 0)

    policy = KerasNN()

    # Train Policy
    batch_size = 2*1000
    batches_per_epoch = 100
    num_iterations = num_epochs * batches_per_epoch
    iterations = 0
    history = []
    val_history = []

    num_epochs = 2*500#100
    load_policy = False
    if load_policy:
        policy.load_model('trained_models/policy_model_{}.h5'.format(experiment_label))
    else:
        history = policy.train(input_data=in_data_train, output_data=out_data_train, batch_size=100, epochs=num_epochs,
                           validation_data=(in_data_valid, out_data_valid))

        fig = plt.figure()
        plt.plot(history.history['loss'], color='red', label='loss')
        plt.plot(history.history['val_loss'], color='blue', label='val_loss')
        plt.legend()
        # plt.show()
        fig.savefig(
            'snapshots/policy_loss_{}_{}.png'.format("diff" if residual_forward else "nodiff", experiment_label),
            bbox_inches='tight')
        policy.save_model('trained_models/policy_model_{}.h5'.format(experiment_label))

    #Evaluo policy
    n = 5000
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
    # sim.restart_scenario() # Para generar secuencias nuevas
    # Muestro figura
    sim.show_image(0)
    # sim.show_image(1)
    # plt.savefig('trazas_experimento/fig_{}.png'.format(0), dpi=100)
    # For n iteraciones
    steps = 0
    intentos = 0
    exitos = 0
    for i in range(1, n):
        if i % 100 == 0:
            print("Iterations: ", i)
        # Pos. actual
        actual = sim.images_history[sim.iter - 1]
        np.copyto(batch_inputs, actual)
        lat_actual = encoder.predict(batch_inputs)
        # Evalua pos. futuras con VF y elijo la mejor accion
        best_action = vf.predict(lat_actual)[0][0]
        best_action = change_range(best_action, 1, 0, 1, -1)*180
        # Aplico accion en robot real
        # Actualizo pos. actual
        sim.apply_action(best_action)
        # Muestro figura
        sim.show_image(sim.iter - 1)
        # plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
        # si llego al goal reinicio el escenario
        steps += 1
        if sim.reward or steps == 10:
            if sim.reward:
                exitos += 1
            intentos += 1
            steps = 0
            sim.restart_scenario()
            sim.show_image(sim.iter - 1)
            # plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
            plt.close('all')
    plt.close('all')
    print(f"Intentos: {intentos} / Exitos: {exitos}. % exito: {exitos * 100 / intentos:.2f}%")
    print("FIN")
