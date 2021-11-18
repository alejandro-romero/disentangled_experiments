from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import math

from skimage.transform import resize

from train_ae import prepare_optimizer_object
from models import make_forward_model
from world_models_vae_arch import build_vae_world_model

from utils import load_parameters

from keras_neural_network import KerasNN

import time
import random as rnd
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial, EvaluateGenomeList_Parallel
from MultiNEAT import GetGenomeList, ZipFitness

from scipy.optimize import curve_fit

from concurrent.futures import ProcessPoolExecutor, as_completed


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


def evaluate(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    acum_utility = []

    for i in range(len(in_data_valid)):
        # Perception
        sens_t = in_data_valid[i]

        entrada = sens_t
        # self.plot_utility_actions()

        # utility_t = 1.0-sens_t[1]#self.change_range(utility_t, old_max=1.0, old_min=0.4, new_max=1.0, new_min=0.0)

        utility_t = vf.predict(sens_t.reshape(1,64))

        # Evaluation
        net.Flush()
        net.Input(np.append(entrada, 1.0))# net.Input(np.append(sens_t, 1.0))
        for _ in range(2):
            net.Activate()
        action = net.Output()
        # Cambio de escala la accion: [0,1] -> [-180, 180]
        best_action = change_range(action[0], 1, 0, 1, -1)*180
        act_fm = np.zeros((1, 1), dtype=np.float32)
        act_fm[0] = best_action/180
        act_fm = np.repeat(act_fm, latent_size, axis=1)
        sens_t1 = forward_model.predict([sens_t.reshape(1, 64), act_fm])
        if residual_forward:
            # print("predicting residual forward...")
            sens_t1 = sens_t + ((sens_t1 * 2) - 1)


        utility_t1 = vf.predict(sens_t1.reshape(1,64))
        # acum_utility.append(float(utility))
        dif = float(utility_t1) - float(utility_t)
        if dif > 0.015:# if dif > 0.055:
            dif = 2 * dif
        # elif dif < 0:
        #     dif = 3*dif
        # acum_utility.append(float(utility_t1)-float(utility_t))
        acum_utility.append(dif)

    fitness = sum(acum_utility)/len(acum_utility)
    return fitness

def getbest(i, inputs, outputs, params, initial_positions, max_generations, batch_time, evaluate):
    g = NEAT.Genome(0, inputs + 1, 0, outputs, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(g, params, True, 1.0, i)
    # pop.RNG.Seed(int(time.clock()*100))
    # pop.RNG.Seed(1234)

    generations = 0
    fitness_data = []
    net_data = []
    rnd.shuffle(initial_positions)
    for generation in range(max_generations):
        # genero nueva percepcion de entrada
        # self.simulator.restart_scenario()
        # self.sens_t = self.simulator.get_sensorization()#np.random.random(size=2)  # Percepcion nueva para evaluar en cada generacion
        if generation % batch_time == 0:
            rnd.shuffle(initial_positions)

        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        # fitness_list = EvaluateGenomeList_Parallel(genome_list, evaluate, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        generations = generation
        best = max(fitness_list)
        print("Best fitness: ", best, " in generation: ", generation)
        # if best > 15.0:
        #     break
        fitness_data.append((fitness_list))
        net = NEAT.NeuralNetwork()
        pop.GetBestGenome().BuildPhenotype(net)
        net_data.append((net.NumHiddenNeurons(),net.NumConnections()))

    net = NEAT.NeuralNetwork()
    pop.GetBestGenome().BuildPhenotype(net)

    # img = NEAT.viz.Draw(net)
    # cv2.imshow("current best", img)
    # cv2.waitKey(1)
    save_policy(pop.GetBestGenome(), i)

    plot_fitness(fitness_data)
    plot_net_evolution(net_data)

    return generations, net.NumHiddenNeurons(), net.NumConnections(), pop.GetBestGenome()

def run(max_runs, inputs, outputs, params, initial_positions, max_generations, batch_time, evaluate):
    # self.load_vf()
    gens = []
    for run in range(max_runs):
        curtime = time.time()

        gen, nodes, connections, best_genome = getbest(run, inputs, outputs, params, initial_positions, max_generations, batch_time, evaluate)
        # self.save_policy(best_genome, run)
        gens += [gen]

        elapsed = time.time() - curtime
        elapsedPerGen = (elapsed / gen) * 1000
        print('Run: {}/{}'.format(run, max_runs - 1), 'Generations to create policy:', gen,
              '| in %3.2f ms per gen, %3.4f s total' % (elapsedPerGen, elapsed),
              "complexity ({}, {})".format(nodes, connections))
    avg_gens = sum(gens) / len(gens)

    print('All:', gens)
    print('Average:', avg_gens)

def plot_fitness(data):
    mean_fitness = []
    max_fitness = []
    for i in range(len(data)):
        mean_fitness.append(sum(data[i])/len(data[i]))
        max_fitness.append(max(data[i]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(mean_fitness) + 1), mean_fitness, zorder=2, linewidth=0.5, label='mean')
    ax.scatter(range(1, len(mean_fitness) + 1), mean_fitness, zorder=1, marker='.')
    ax.plot(range(1, len(max_fitness) + 1), max_fitness, zorder=2, linewidth=0.5, label='max')
    ax.scatter(range(1, len(max_fitness) + 1), max_fitness, zorder=1, marker='.')
    ax.set_xlabel('Generation', size=12.0)
    ax.set_ylabel('Fitness', size=12.0)
    ax.set_title('Fitness evolution', size=12.0)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    for i in range(len(data)):
        if i % self.batch_time == 0:
            plt.axvline(i, linestyle='--', linewidth=0.1, color='grey')
    plt.show()

def plot_net_evolution(self, data):
    hidden = list(zip(*data))[0]
    connections = list(zip(*data))[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(data) + 1), hidden, zorder=2, linewidth=0.5, label='hidden neurons')
    ax.scatter(range(1, len(data) + 1), hidden, zorder=1, marker='.')
    ax2 = ax.twinx()
    ax2.plot(range(1, len(data) + 1), connections, zorder=2, linewidth=0.5, color='orange', label='connections')
    ax2.scatter(range(1, len(data) + 1), connections, zorder=1, color='orange', marker='.')
    ax.set_xlabel('Generation', size=12.0)
    ax.set_ylabel('Hidden Neurons', size=12.0)
    ax2.set_ylabel('Connections', size=12.0)
    ax.set_title('Net evolution', size=12.0)
    fig.legend()
    ax.grid(linestyle="--", alpha=0.5)
    for i in range(len(data)):
        if i % self.batch_time == 0:
            plt.axvline(i, linestyle='--', linewidth=0.1, color='grey')
    plt.show()

def plot_utility_actions(self):
    acciones = []
    utilities = []
    sens_t=self.simulator.get_sensorization()
    # for accion in range(-90, 90):
    utility_t = 1.0 - sens_t[1]
    for accion in range(0, 360):
        acciones.append(accion)
        sens_t1 = self.forward_model.predicted_state((accion,50), self.simulator.get_scenario_data())
        # sens_t1 = np.asarray(sens_t1 + (-1,))
        # utility = self.net.mlpfwd(sens_t1.reshape(1, 3))
        # utility = self.change_range(utility, old_max=1.0, old_min=0.4, new_max=1.0, new_min=0.0)
        utility_t1 = 1.0 - sens_t1[1]
        dif = float(utility_t1) - float(utility_t)
        if dif > 0.015:#0.055:
            utilities.append(2*dif)
        # elif dif < 0:
        #     utilities.append(3*dif)
        else:
            utilities.append(dif)
        # utilities.append(float(utility_t1) - float(utility_t))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acciones, utilities, zorder=2, linewidth=0.5)
    ax.set_xlabel('Action', size=12.0)
    ax.set_ylabel('Utility', size=12.0)
    ax.set_title('Candidate actions and expected utility: ' + str(sens_t), size=12.0)
    ax.grid(linestyle="--", alpha=0.5)

# def test_policy():
#     policy = self.load_policy('generated_policies/policy_4_batch_30_variable_exp_pob100_12.pckl')
#     self.simulator.visualize = True
#     net = NEAT.NeuralNetwork()
#     policy.BuildPhenotype(net)
#
#     for i in range(2000):
#
#         # net.Flush()
#         #
#         # sens_t = self.simulator.get_sensorization()
#         # net.Input(np.append(sens_t, 1.0))
#         # for _ in range(2):
#         #     net.Activate()
#         # action = net.Output()
#         # # Cambio de escala la accion: [0,1] -> [-90, 90]
#         # action_sim = self.change_range(action[0])
#         #
#         # sens_t1 = self.forward_model.predicted_state(action_sim, self.simulator.get_scenario_data())
#         # sens_t1 = np.asarray(sens_t1 + (-1,))
#         # utility = self.net.mlpfwd(sens_t1.reshape(1, 3))
#
#         x_ball = self.normalize_value(self.simulator.ball_1_get_pos()[0], max_value=2400, min_value=100)
#         y_ball = self.normalize_value(self.simulator.ball_1_get_pos()[1], max_value=800, min_value=50)
#         x_box = self.normalize_value(self.simulator.box1_get_pos()[0], max_value=2400, min_value=100)
#         y_box = self.normalize_value(self.simulator.box1_get_pos()[1], max_value=800, min_value=50)
#
#         entrada = np.array((x_ball, y_ball, x_box, y_box))
#         # self.plot_utility_actions()
#
#         # Evaluation
#         net.Flush()
#         net.Input(np.append(entrada, 1.0))  # net.Input(np.append(sens_t, 1.0))
#         for _ in range(2):
#             net.Activate()
#         action = net.Output()
#         # Cambio de escala la accion: [0,1] -> [-90, 90]
#         act1 = self.change_range(old_value=action[0], old_max=1.0, old_min=0.0, new_max=180.0, new_min=0.0)
#         act2 = self.change_range(old_value=action[1], old_max=1.0, old_min=0.0, new_max=1.0, new_min=-1.0)
#
#         act3 = self.change_range(old_value=action[2], old_max=1.0, old_min=0.0, new_max=50.0, new_min=0.0)
#
#         action_sim = (act1 * np.sign(act2), act3) # Para accion 0-180
#         #
#         # if act2 >= 0:
#         #     action_sim = (act1, act3) # Para accion -90 a 90
#         # else:
#         #     action_sim = (act1+180, act3)
#
#         self.simulator.apply_action(action_sim)
#
#         self.simulator.world_rules()
#         if self.simulator.get_reward():
#             self.simulator.restart_scenario2()
#             # self.simulator.restart_scenario3()

def save_policy(genome, number):
    f = open('generated_policies/policy_'+str(number)+'_batch_30_variable_exp_pob100_12.pckl', 'wb')
    pickle.dump(genome, f)
    f.close()

def load_policy(filename):
    f = open(filename, 'rb')
    policy = pickle.load(f)
    f.close()
    return policy

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
    n = 7500
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

    # policy = KerasNN()

    # Train Policy

    # batch_size = 2*1000
    # batches_per_epoch = 100
    # num_iterations = num_epochs * batches_per_epoch
    # iterations = 0
    # history = []
    # val_history = []
    #
    # num_epochs = 2*500#100
    # load_policy = False
    # if load_policy:
    #     policy.load_model('trained_models/policy_model_{}.h5'.format(experiment_label))
    # else:
    #     history = policy.train(input_data=in_data_train, output_data=out_data_train, batch_size=100, epochs=num_epochs,
    #                        validation_data=(in_data_valid, out_data_valid))
    #
    #     fig = plt.figure()
    #     plt.plot(history.history['loss'], color='red', label='loss')
    #     plt.plot(history.history['val_loss'], color='blue', label='val_loss')
    #     plt.legend()
    #     # plt.show()
    #     fig.savefig(
    #         'snapshots/policy_loss_{}_{}.png'.format("diff" if residual_forward else "nodiff", experiment_label),
    #         bbox_inches='tight')
    #     policy.save_model('trained_models/policy_model_{}.h5'.format(experiment_label))


    #####NEAT
    # # NEAT
    params = NEAT.Parameters()
    # Basic parameters
    params.PopulationSize = 100  # populationSize
    params.DynamicCompatibility = True
    params.MinSpecies = 2
    params.MaxSpecies = 7
    params.InnovationsForever = True

    # GA Parameters
    params.YoungAgeTreshold = 5
    params.YoungAgeFitnessBoost = 1.0
    # params.SpeciesMaxStagnation = 100
    params.SpeciesDropoffAge = 50
    params.StagnationDelta = 0.0
    params.OldAgeTreshold = 30
    params.OldAgePenalty = 1.0
    params.KillWorstAge = 40
    params.SurvivalRate = 0.3
    params.CrossoverRate = 0.6
    params.KillWorstSpeciesEach = 20
    params.OverallMutationRate = 0.1
    params.MultipointCrossoverRate = 0.5
    params.RouletteWheelSelection = False
    params.InterspeciesCrossoverRate = 0.001
    params.DetectCompetetiveCoevolutionStagnation = True

    # Structural Mutation parameters
    params.RecurrentProb = 0.1
    params.SplitRecurrent = True
    params.MutateAddLinkProb = 0.5
    params.RecurrentLoopProb = 0.01
    params.MutateRemLinkProb = 0.01
    params.MutateAddNeuronProb = 0.5
    params.SplitLoopedRecurrent = False
    params.MutateAddLinkFromBiasProb = 0.0
    params.MutateRemSimpleNeuronProb = 0.0

    # Parameter Mutation parameters
    params.MutateWeightsProb = 0.5
    params.MutateWeightsSevereProb = 0.1
    params.WeightMutationRate = 0.9
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 1.0
    params.MaxWeight = 10.0
    params.MutateActivationAProb = 0.0
    params.MutateActivationBProb = 0.0
    params.ActivationAMutationMaxPower = 0.0
    params.ActivationBMutationMaxPower = 0.0
    params.TimeConstantMutationMaxPower = 0.0
    params.BiasMutationMaxPower = 1.0
    params.MutateNeuronTimeConstantsProb = 0.0
    params.MutateNeuronBiasesProb = 0.1
    params.MinNeuronBias = -10.0
    params.MaxNeuronBias = 10.0
    params.EliteFraction = 0.02

    params.ExcessCoeff = 1.0
    params.DisjointCoeff = 1.0
    params.BiasDiffCoeff = 2.0
    params.CompatTreshold = 2.0
    params.WeightDiffCoeff = 2.0
    params.ActivationADiffCoeff = 0.0
    params.ActivationBDiffCoeff = 0.0
    params.TimeConstantDiffCoeff = 0.0
    params.CompatTresholdModifier = 0.1
    params.ActivationFunctionDiffCoeff = 0.0
    params.CompatTreshChangeInterval_Generations = 1

    # Pendiente
    params.MinActivationA = 1.0
    params.MaxActivationA = 1.0

    # Desplazamiento
    params.MinActivationB = 0.0
    params.MaxActivationB = 0.0

    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0

    max_runs = 5  # 10
    max_generations = 1000  # 100 ##150
    evaluated_points = 100  # 250
    # NET
    inputs = 64  # 2
    outputs = 1  # 2

    batch_time = 30

    initial_positions = in_data_train
    run(max_runs, inputs, outputs, params, initial_positions, max_generations, batch_time, evaluate)

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

############################################



class POLICY_TRAINING(object):

    def __init__(self):
        self.candidate_state_evaluator = CandidateStateEvaluator()
        self.simulator = Sim()
        self.simulator.visualize = False
        self.forward_model = ForwardModel()
        self.net = None

        # # NEAT
        self.params = NEAT.Parameters()
        # Basic parameters
        self.params.PopulationSize = 100#populationSize
        self.params.DynamicCompatibility = True
        self.params.MinSpecies = 2
        self.params.MaxSpecies = 7
        self.params.InnovationsForever = True

        # GA Parameters
        self.params.YoungAgeTreshold = 5
        self.params.YoungAgeFitnessBoost = 1.0
        # self.params.SpeciesMaxStagnation = 100
        self.params.SpeciesDropoffAge = 50
        self.params.StagnationDelta = 0.0
        self.params.OldAgeTreshold = 30
        self.params.OldAgePenalty = 1.0
        self.params.KillWorstAge = 40
        self.params.SurvivalRate = 0.3
        self.params.CrossoverRate = 0.6
        self.params.KillWorstSpeciesEach = 20
        self.params.OverallMutationRate = 0.1
        self.params.MultipointCrossoverRate = 0.5
        self.params.RouletteWheelSelection = False
        self.params.InterspeciesCrossoverRate = 0.001
        self.params.DetectCompetetiveCoevolutionStagnation = True

        # Structural Mutation parameters
        self.params.RecurrentProb = 0.1
        self.params.SplitRecurrent = True
        self.params.MutateAddLinkProb = 0.5
        self.params.RecurrentLoopProb = 0.01
        self.params.MutateRemLinkProb = 0.01
        self.params.MutateAddNeuronProb = 0.5
        self.params.SplitLoopedRecurrent = False
        self.params.MutateAddLinkFromBiasProb = 0.0
        self.params.MutateRemSimpleNeuronProb = 0.0

        # Parameter Mutation parameters
        self.params.MutateWeightsProb = 0.5
        self.params.MutateWeightsSevereProb = 0.1
        self.params.WeightMutationRate = 0.9
        self.params.WeightMutationMaxPower = 1.0
        self.params.WeightReplacementMaxPower = 1.0
        self.params.MaxWeight = 10.0
        self.params.MutateActivationAProb = 0.0
        self.params.MutateActivationBProb = 0.0
        self.params.ActivationAMutationMaxPower = 0.0
        self.params.ActivationBMutationMaxPower = 0.0
        self.params.TimeConstantMutationMaxPower = 0.0
        self.params.BiasMutationMaxPower = 1.0
        self.params.MutateNeuronTimeConstantsProb = 0.0
        self.params.MutateNeuronBiasesProb = 0.1
        self.params.MinNeuronBias = -10.0
        self.params.MaxNeuronBias = 10.0
        self.params.EliteFraction = 0.02

        self.params.ExcessCoeff = 1.0
        self.params.DisjointCoeff = 1.0
        self.params.BiasDiffCoeff = 2.0
        self.params.CompatTreshold = 2.0
        self.params.WeightDiffCoeff = 2.0
        self.params.ActivationADiffCoeff = 0.0
        self.params.ActivationBDiffCoeff = 0.0
        self.params.TimeConstantDiffCoeff = 0.0
        self.params.CompatTresholdModifier = 0.1
        self.params.ActivationFunctionDiffCoeff = 0.0
        self.params.CompatTreshChangeInterval_Generations = 1

        # Pendiente
        self.params.MinActivationA = 1.0
        self.params.MaxActivationA = 1.0

        # Desplazamiento
        self.params.MinActivationB = 0.0
        self.params.MaxActivationB = 0.0

        self.params.ActivationFunction_Tanh_Prob = 0.0
        self.params.ActivationFunction_SignedStep_Prob = 0.0
        self.params.ActivationFunction_SignedSigmoid_Prob = 0.0
        self.params.ActivationFunction_UnsignedSigmoid_Prob = 1.0

        self.max_runs = 5#10
        self.max_generations = 1000#100 ##150
        self.evaluated_points = 100#250
        # NET
        self.inputs = 64#2
        self.outputs = 1#2

        self.batch_time = 30

    def evaluate(self, genome):
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

        acum_utility = []

        for i in range(self.evaluated_points):
            # Perception
            self.simulator.baxter_larm_set_pos(self.initial_positions[i][0])
            self.simulator.ball_set_pos(self.initial_positions[i][1])
            self.simulator.box1_set_pos(self.initial_positions[i][2])
            self.simulator.ball_position = self.initial_positions[i][3]
            self.simulator.baxter_larm.angle = self.initial_positions[i][4]
            sens_t = self.simulator.get_sensorization()
            # entrada = np.append(self.initial_positions[i][0], self.initial_positions[i][4])

            x_ball = self.normalize_value(self.initial_positions[i][1][0], max_value=2400, min_value=100)
            y_ball = self.normalize_value(self.initial_positions[i][1][1], max_value=800, min_value=50)
            x_box = self.normalize_value(self.initial_positions[i][2][0], max_value=2400, min_value=100)
            y_box = self.normalize_value(self.initial_positions[i][2][1], max_value=800, min_value=50)

            entrada = np.array((x_ball, y_ball, x_box, y_box))
            # self.plot_utility_actions()
            sens_t = self.simulator.get_sensorization()
            # sens_t = np.asarray(sens_t + (-1,))
            # utility_t = self.net.mlpfwd(sens_t.reshape(1, 3))
            utility_t = 1.0-sens_t[1]#self.change_range(utility_t, old_max=1.0, old_min=0.4, new_max=1.0, new_min=0.0)

            # Evaluation
            net.Flush()
            net.Input(np.append(entrada, 1.0))# net.Input(np.append(sens_t, 1.0))
            for _ in range(2):
                net.Activate()
            action = net.Output()
            # Cambio de escala la accion: [0,1] -> [-90, 90]

            act1 = self.change_range(old_value=action[0], old_max=1.0, old_min=0.0, new_max=180.0, new_min=0.0)
            act2 = self.change_range(old_value=action[1], old_max=1.0, old_min=0.0, new_max=1.0, new_min=-1.0)
            # Vel 50-150
            act3 = self.change_range(old_value=action[2], old_max=1.0, old_min=0.0, new_max=50.0, new_min=0.0)

            action_sim = (act1 * np.sign(act2), act3) # Para accion 0-180

            # if act2 >= 0:
            #     action_sim = (act1, act3)  # Para accion -90 a 90
            # else:
            #     action_sim = (act1+180, act3)

            sens_t1 = self.forward_model.predicted_state(action_sim, self.simulator.get_scenario_data())
            # sens_t1 = np.asarray(sens_t1 + (-1,))
            # utility_t1 = self.net.mlpfwd(sens_t1.reshape(1, 3))
            utility_t1 = 1.0-sens_t1[1]#self.change_range(utility_t1, old_max=1.0, old_min=0.4, new_max=1.0, new_min=0.0)
            # acum_utility.append(float(utility))
            dif = float(utility_t1) - float(utility_t)
            if dif > 0.015:# if dif > 0.055:
                dif = 2 * dif
            # elif dif < 0:
            #     dif = 3*dif
            # acum_utility.append(float(utility_t1)-float(utility_t))
            acum_utility.append(dif)

        fitness = sum(acum_utility)/len(acum_utility)
        return fitness

    def getbest(self, i):
        g = NEAT.Genome(0, self.inputs + 1, 0, self.outputs, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, self.params, 0)
        pop = NEAT.Population(g, self.params, True, 1.0, i)
        # pop.RNG.Seed(int(time.clock()*100))
        # pop.RNG.Seed(1234)

        generations = 0
        fitness_data = []
        net_data = []
        rnd.shuffle(self.initial_positions)
        for generation in range(self.max_generations):
            # genero nueva percepcion de entrada
            # self.simulator.restart_scenario()
            # self.sens_t = self.simulator.get_sensorization()#np.random.random(size=2)  # Percepcion nueva para evaluar en cada generacion
            if generation % self.batch_time == 0:
                rnd.shuffle(self.initial_positions)

            genome_list = NEAT.GetGenomeList(pop)
            fitness_list = EvaluateGenomeList_Serial(genome_list, self.evaluate, display=False)
            # fitness_list = EvaluateGenomeList_Parallel(genome_list, evaluate, display=False)
            NEAT.ZipFitness(genome_list, fitness_list)
            pop.Epoch()
            generations = generation
            best = max(fitness_list)
            print("Best fitness: ", best, " in generation: ", generation)
            # if best > 15.0:
            #     break
            fitness_data.append((fitness_list))
            net = NEAT.NeuralNetwork()
            pop.GetBestGenome().BuildPhenotype(net)
            net_data.append((net.NumHiddenNeurons(),net.NumConnections()))

        net = NEAT.NeuralNetwork()
        pop.GetBestGenome().BuildPhenotype(net)

        # img = NEAT.viz.Draw(net)
        # cv2.imshow("current best", img)
        # cv2.waitKey(1)
        self.save_policy(pop.GetBestGenome(), i)

        self.plot_fitness(fitness_data)
        self.plot_net_evolution(net_data)

        return generations, net.NumHiddenNeurons(), net.NumConnections(), pop.GetBestGenome()

    def run(self):
        # self.load_vf()
        gens = []
        for run in range(self.max_runs):
            curtime = time.time()

            gen, nodes, connections, best_genome = self.getbest(run)
            # self.save_policy(best_genome, run)
            gens += [gen]

            elapsed = time.time() - curtime
            elapsedPerGen = (elapsed / gen) * 1000
            print('Run: {}/{}'.format(run, self.max_runs - 1), 'Generations to create policy:', gen,
                  '| in %3.2f ms per gen, %3.4f s total' % (elapsedPerGen, elapsed),
                  "complexity ({}, {})".format(nodes, connections))
        avg_gens = sum(gens) / len(gens)

        print('All:', gens)
        print('Average:', avg_gens)

    def plot_fitness(self, data):
        mean_fitness = []
        max_fitness = []
        for i in range(len(data)):
            mean_fitness.append(sum(data[i])/len(data[i]))
            max_fitness.append(max(data[i]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(mean_fitness) + 1), mean_fitness, zorder=2, linewidth=0.5, label='mean')
        ax.scatter(range(1, len(mean_fitness) + 1), mean_fitness, zorder=1, marker='.')
        ax.plot(range(1, len(max_fitness) + 1), max_fitness, zorder=2, linewidth=0.5, label='max')
        ax.scatter(range(1, len(max_fitness) + 1), max_fitness, zorder=1, marker='.')
        ax.set_xlabel('Generation', size=12.0)
        ax.set_ylabel('Fitness', size=12.0)
        ax.set_title('Fitness evolution', size=12.0)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)
        for i in range(len(data)):
            if i % self.batch_time == 0:
                plt.axvline(i, linestyle='--', linewidth=0.1, color='grey')
        plt.show()

    def plot_net_evolution(self, data):
        hidden = list(zip(*data))[0]
        connections = list(zip(*data))[1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(data) + 1), hidden, zorder=2, linewidth=0.5, label='hidden neurons')
        ax.scatter(range(1, len(data) + 1), hidden, zorder=1, marker='.')
        ax2 = ax.twinx()
        ax2.plot(range(1, len(data) + 1), connections, zorder=2, linewidth=0.5, color='orange', label='connections')
        ax2.scatter(range(1, len(data) + 1), connections, zorder=1, color='orange', marker='.')
        ax.set_xlabel('Generation', size=12.0)
        ax.set_ylabel('Hidden Neurons', size=12.0)
        ax2.set_ylabel('Connections', size=12.0)
        ax.set_title('Net evolution', size=12.0)
        fig.legend()
        ax.grid(linestyle="--", alpha=0.5)
        for i in range(len(data)):
            if i % self.batch_time == 0:
                plt.axvline(i, linestyle='--', linewidth=0.1, color='grey')
        plt.show()

    def plot_utility_actions(self):
        acciones = []
        utilities = []
        sens_t=self.simulator.get_sensorization()
        # for accion in range(-90, 90):
        utility_t = 1.0 - sens_t[1]
        for accion in range(0, 360):
            acciones.append(accion)
            sens_t1 = self.forward_model.predicted_state((accion,50), self.simulator.get_scenario_data())
            # sens_t1 = np.asarray(sens_t1 + (-1,))
            # utility = self.net.mlpfwd(sens_t1.reshape(1, 3))
            # utility = self.change_range(utility, old_max=1.0, old_min=0.4, new_max=1.0, new_min=0.0)
            utility_t1 = 1.0 - sens_t1[1]
            dif = float(utility_t1) - float(utility_t)
            if dif > 0.015:#0.055:
                utilities.append(2*dif)
            # elif dif < 0:
            #     utilities.append(3*dif)
            else:
                utilities.append(dif)
            # utilities.append(float(utility_t1) - float(utility_t))


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(acciones, utilities, zorder=2, linewidth=0.5)
        ax.set_xlabel('Action', size=12.0)
        ax.set_ylabel('Utility', size=12.0)
        ax.set_title('Candidate actions and expected utility: ' + str(sens_t), size=12.0)
        ax.grid(linestyle="--", alpha=0.5)

    @staticmethod
    # def change_range(old_value, old_max=1.0, old_min=0.0, new_max=90, new_min=-90):
    # def change_range(old_value, old_max=1.0, old_min=0.0, new_max=360, new_min=0):
    def change_range(old_value, old_max, old_min, new_max, new_min):
        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min
        return new_value

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    def load_vf(self):
        f = open('net_vf_ball_in_box.pckl', 'rb')
        self.net = pickle.load(f)
        f.close()

    def save_policy(self, genome, number):
        f = open('generated_policies/policy_'+str(number)+'_batch_30_variable_exp_pob100_12.pckl', 'wb')
        pickle.dump(genome, f)
        f.close()

    def test_policy(self):
        policy = self.load_policy('generated_policies/policy_4_batch_30_variable_exp_pob100_12.pckl')
        self.simulator.visualize = True
        net = NEAT.NeuralNetwork()
        policy.BuildPhenotype(net)

        for i in range(2000):

            # net.Flush()
            #
            # sens_t = self.simulator.get_sensorization()
            # net.Input(np.append(sens_t, 1.0))
            # for _ in range(2):
            #     net.Activate()
            # action = net.Output()
            # # Cambio de escala la accion: [0,1] -> [-90, 90]
            # action_sim = self.change_range(action[0])
            #
            # sens_t1 = self.forward_model.predicted_state(action_sim, self.simulator.get_scenario_data())
            # sens_t1 = np.asarray(sens_t1 + (-1,))
            # utility = self.net.mlpfwd(sens_t1.reshape(1, 3))

            x_ball = self.normalize_value(self.simulator.ball_1_get_pos()[0], max_value=2400, min_value=100)
            y_ball = self.normalize_value(self.simulator.ball_1_get_pos()[1], max_value=800, min_value=50)
            x_box = self.normalize_value(self.simulator.box1_get_pos()[0], max_value=2400, min_value=100)
            y_box = self.normalize_value(self.simulator.box1_get_pos()[1], max_value=800, min_value=50)

            entrada = np.array((x_ball, y_ball, x_box, y_box))
            # self.plot_utility_actions()

            # Evaluation
            net.Flush()
            net.Input(np.append(entrada, 1.0))  # net.Input(np.append(sens_t, 1.0))
            for _ in range(2):
                net.Activate()
            action = net.Output()
            # Cambio de escala la accion: [0,1] -> [-90, 90]
            act1 = self.change_range(old_value=action[0], old_max=1.0, old_min=0.0, new_max=180.0, new_min=0.0)
            act2 = self.change_range(old_value=action[1], old_max=1.0, old_min=0.0, new_max=1.0, new_min=-1.0)

            act3 = self.change_range(old_value=action[2], old_max=1.0, old_min=0.0, new_max=50.0, new_min=0.0)

            action_sim = (act1 * np.sign(act2), act3) # Para accion 0-180
            #
            # if act2 >= 0:
            #     action_sim = (act1, act3) # Para accion -90 a 90
            # else:
            #     action_sim = (act1+180, act3)

            self.simulator.apply_action(action_sim)

            self.simulator.world_rules()
            if self.simulator.get_reward():
                self.simulator.restart_scenario2()
                # self.simulator.restart_scenario3()

def main():
    instance = POLICY_TRAINING()
    instance.load_vf()
    instance.generate_random_initial_positions()
    instance.run()
    instance.test_policy()
    # instance.test_deliberative()
    instance.graph_paper_steps_goal()


if __name__ == '__main__':
    main()