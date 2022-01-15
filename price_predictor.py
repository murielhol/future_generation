from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse
import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import pandas as pd 
import json
import shutil
import pickle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import timeit

import seaborn as sns
from model import Model

from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio



np.random.seed(111)
tf.set_random_seed(111)
torch.manual_seed(111)

'''
TODO:
- generalize the params
'''

class PricePredictor(object):
    def __init__(self, config, dataset):
        # Initialize the model
        self._model = Model(config)
        self._config = config
        self._dataset = dataset
        # make a folder to keep all info about this model
        if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
        if not os.path.exists('saved_models/'+self._config.model_name):
                    os.makedirs('saved_models/'+self._config.model_name)
        self.save_path = 'saved_models/'+self._config.model_name

    @staticmethod
    def adjust_lr(optimizer, epoch, total_epoch, init_lr, end_lr):
        lr = end_lr + (init_lr - end_lr) * (0.5 * (1+math.cos(math.pi * float(epoch) / total_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    @staticmethod
    def adjust_kd(epoch, total_epoch, init_kd, end_kd):
        if epoch > total_epoch:
            return 1.
        return end_kd + (init_kd - end_kd) * math.cos(0.5 * math.pi * float(epoch) / total_epoch)

    @staticmethod
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.net.eval()
        model.opt.zero_grad()
        # imagine batch size is 10, seq_len is 1000 and 1 channel
        bs = config.batch_size
        seq_len = config.input_seq_length
        channels = 28
        x = np.ones([bs, seq_len, channels])
        # for pytorch convs it is [batch_size, channels, width, height]
        x = np.einsum('ijk->jik', x)
        y = x.copy()
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        y = Variable(torch.from_numpy(y).float())
        mask_x = Variable(torch.from_numpy(np.ones([seq_len, bs])).float())
        # self._model.net.eval()
        _, _, pars = model.net([x,y, mask_x])
        mu = pars[0]
        grad=torch.zeros(mu.size())
        # imagien only 1 output in the time axis has a gradient
        grad[-1, :, :] = 1.0
        mu.backward(gradient=grad)
        # see what this gradient is wrt the input
        zeros=np.where(x.grad.data!=0)
        RF = len(set(zeros[0]))
        print('RF: ', RF)
        return RF

    @staticmethod
    def evaluate(x, y, model, mask):
        x = np.einsum('ijk->jik', x)
        y = np.einsum('ijk->jik', y)
        model.eval()
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()
        mask_x = Variable(torch.from_numpy(mask).float())
        loss, kld_loss, outputs = model([x,y, mask_x]);
        total_loss = loss - kld_loss 
        total_loss = total_loss.item()
        # mean = outputs[0].detach().numpy() 
        # plt.plot(mean[:, 0 ,0])
        # plt.show()
        
        return total_loss, loss.item(), kld_loss.item(), outputs


    def _train(self):
        ################## build model ##############################
        t1 = time.time()
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        print('--------------------------------------------------------------------')
        print('NOTE: the receptive field is ', receptive_field, ' and your input is ', self._config.input_seq_length)
        print('--------------------------------------------------------------------')
        t2 = time.time()
        print('Finished building the model: ' + str(t2-t1) +' sec \n')
        # ################# get data ################################
        self._dataset.prepare_data(self._config, shuffle = self._config.shuffle, skip = self._config.input_seq_length - receptive_field)
        print('train size: ', np.shape(self._dataset._train_x))
        print('test size: ', np.shape(self._dataset._test_x))
        t3 = time.time()
        print('Finished preparing the dataset: ' + str(t3-t2) +' sec \n')
        ################### prepare log structure ##################
        log_folder = 'saved_models/'+self._config.model_name+'/logs/'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        else:
            for root, dirs, files in os.walk(log_folder):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        
        steps_per_epoch = len(self._dataset._train_y)//self._config.batch_size
        ############## initialize all the stuff ##################
        kld_step = 0.5
        kld_weight = kld_step
        
        print('train size: ', np.shape(self._dataset._train_x))
        print('test size: ', np.shape(self._dataset._test_x))
        costs = []
        test_costs = []
        mean_kl_costs = []
        mean_ll_costs = []
        t = timeit.default_timer()
        mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        mask[receptive_field:, :] = 1
        test_mask = np.zeros([self._config.input_seq_length, self._config.batch_size*10])
        test_mask[receptive_field:, :] = 1
        for epoch in range(1, int(self._config.epochs+1)):
            self._model.net.train()
            loss_sum = 0
            kld_loss_sum = 0
            logp_loss_sum = 0

            test_loss_sum = 0
            test_kld_loss_sum = 0
            test_logp_loss_sum = 0

            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                self._model.opt.zero_grad()
                x, y = self._dataset.get_batch(self._config.batch_size)
                x = np.einsum('ijk->jik', x)
                # y = np.einsum('ijk->jik', y)
                # inputs are rows 1 to 27
                # targets are rows 2 to 28
                y = Variable(torch.from_numpy(x[1:,:,:]).float())
                x = Variable(torch.from_numpy(x[:-1,:,:]).float())
                
                mask_x = Variable(torch.from_numpy(mask).float())
                if (self._config.kld == 'True'):
                    loss, kld_loss, _ = self._model.net([x,y, mask_x])
                    total_loss = loss - kld_loss * kld_weight
                    total_loss.backward();
                    total_loss = total_loss.item()
                    kld_loss_sum += kld_loss.item()
                    logp_loss_sum += loss.item()
                else:
                    pass
                    # all_loss = self._model.net([x,y])
                    # all_loss.backward()
                    # total_loss = all_loss.item()

                costs.append(total_loss)                
                torch.nn.utils.clip_grad_norm_(self._model.net.parameters(), 0.1, 'inf')
                self._model.opt.step()
                loss_sum += total_loss;


                ################ occasionally show the (test) performance #############  
                # if train_step % self._config.print_every == 0:

            x, y = self._dataset.get_batch(self._config.batch_size*10, test = True)
            test_cost, test_nll, test_kld_loss, test_pred = self.evaluate(x[:,:-1,:], y[:,1:,:], self._model.net, test_mask)
            
            if epoch in [1, 5, 10, 20, 100]:
                test_pred = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pred[0].detach().numpy()))).float()
                print(np.shape(test_pred))
                plt.plot(test_pred[0,:,:])
                plt.plot(y[0,:,:])
                plt.show()

            test_costs.append(test_cost)
            test_loss_sum += test_cost
            test_kld_loss_sum += test_kld_loss
            test_logp_loss_sum += test_nll
            mean_kl_costs.append(np.mean(test_kld_loss))
            mean_ll_costs.append(np.mean(test_nll))

            if epoch==20:
                reference_ll = mean_ll_costs[-1]
                reference_kl = mean_kl_costs[-1]
                improvements_ll, improvements_kl = [], []
                print('reference loss: ', reference)
            if epoch>20:
                improvement_ll = mean_ll_costs[-2] - mean_ll_costs[-1]
                improvement_kl = mean_kl_costs[-2] - mean_kl_costs[-1]
                if improvement_ll < reference_ll*0.01 and improvement_kl < reference_kl*0.01:
                    converged  += 1
                    print('converged: ',converged)
                else:
                    improvements_ll.append(improvement_ll)
                    improvements_kl.append(improvement_kl)
                    c = converged - 1
                    converged = np.max([0, c])
                
                print('improvement_ll: ', improvement_ll)
                print('improvement_kl: ', improvement_kl)


            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f, logp_loss:%f, kld_loss: %f,\
             \n                       test_loss: %f, test_logp_loss:%f, test_kld_loss: %f, kld_weight: %f' % (
                s-t, epoch, self._config.epochs, train_step, steps_per_epoch,
                -loss_sum / train_step, -logp_loss_sum/train_step, -kld_loss_sum/train_step,
                -test_loss_sum / train_step, -test_logp_loss_sum/train_step, -test_kld_loss_sum/train_step,
                kld_weight
                )
            print(log_line)
            
            # print("- batch {0:.1f} | {1:.8f} | {2:.8f} | ".format(train_step, 
            #                                         np.mean(costs),
            #                                         np.mean(test_costs) ))
            log.append([train_step, np.mean(costs), np.mean(test_costs)])

            # adjust the KL weight and also the learning rate
            print('Adjusting kld weight and learning rate')
            kld_weight = self.adjust_kd(epoch, self._config.epochs, kld_step, 1.)
            print('KL weight: ', kld_weight)
            self.adjust_lr(self._model.opt, epoch, self._config.epochs, self._config.learning_rate, 0.)

            if epoch%10==0:
                state = {
                    'epoch': epoch,
                    'state_dict': self._model.net.state_dict(),
                    'optimizer': self._model.opt.state_dict()                    
                    }

                torch.save(state, 'saved_models/'+self._config.model_name+str(epoch)+'.pth.tar')
                print('Saved model of epoch ', epoch)
                # dump confg json to keep track of model properties
                with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
                    json.dump(vars(self._config), fp)
                with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
                    pickle.dump( self._config, fp )
            # write results to a log file
            log = pd.DataFrame(np.array(log), columns = ['step', 'train loss', 'test loss'])
            log.to_csv('saved_models/'+self._config.model_name+'/epoch'+str(epoch)+'.csv')
            


    def _validate(self, steps = 5):
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        mask = np.zeros([receptive_field, 2500])
        mask[receptive_field-1:, :] = 1
        skip = self._config.input_seq_length - receptive_field
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config, skip = skip)
        ###### first build the graph again ########
        sigma = tf.placeholder(tf.float32, shape=1, name='sigma')

        MSE1, MSE5, MSE10 = [], [], []
        x, y,  dates = self._dataset.get_validation_set()
        
        preds, pars,  tars, ins = [], [], [], []
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+'550.pth.tar', map_location='cpu')

        self._model.net.load_state_dict(state['state_dict'])

        MLE = []

        # if 'ho' in self._config.file_path:
        #     a = 1.0
        #     for seed in range(50):
        #         np.random.seed(seed)
        #         torch.manual_seed(seed)
        #         # torch.cuda.manual_seed(seed)
        #         # torch.cuda.manual_seed_all(seed);
        #         X = x.copy()
        #         print('simulation ', seed)
                
        #         for step in range(100):
        #             # print(np.shape(x[:, step:step+receptive_field, :]))
        #             # print(np.shape(y[:, step:step+receptive_field]))
        #             _, _, _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], X[:, step:step+receptive_field], self._model.net, mask)

            #         X = np.concatenate([X, np.einsum('ijk->jik', test_pars[0].detach().numpy()[-1:,:, :])], axis = 1)
            #     plt.plot(X[2, : , :], alpha = a, c='b')
            #     a *=0.99
            # # plt.plot(batch_y[0, :, :])
            # plt.show()
            # a = 1.0
            # for seed in range(50):
            #     np.random.seed(seed)
            #     torch.manual_seed(seed)
            #     # torch.cuda.manual_seed(seed)
            #     # torch.cuda.manual_seed_all(seed);
            #     X = x.copy()
            #     print('simulation ', seed)
                
            #     for step in range(50):
            #         # print(np.shape(x[:, step:step+receptive_field, :]))
            #         # print(np.shape(y[:, step:step+receptive_field]))
            #         _, _, _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], X[:, step:step+receptive_field], self._model.net, mask)
            #         locs = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:, :]))).float()
            #         logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
            #         Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
            #         test_pred = Dist.sample()

            #         X = np.concatenate([X, test_pred], axis = 1)
            #     plt.plot(X[8, 1: , :], alpha = a, c='b')
            #     a *=0.97
            # plt.plot(y[8, :, :], c='r')
            # plt.show()
            # np.random.seed(111)
            # torch.manual_seed(111)


        recon, reglos, lowerbound = [],[],[]
        for step in range(steps):
            print('step: ', step)
            test_cost, test_nll, test_kld_loss, test_pars = self.evaluate(x[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], self._model.net, mask)
            
            locs = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:, :]))).float()
            logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
            Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
            test_pred = Dist.sample()

            loc = np.einsum('ijk->jik', test_pars[0].detach().numpy()[-1:,:, :])
            logvar = np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])
            P = np.concatenate([loc, logvar], axis=-1)
            x = np.concatenate([x, test_pred], axis = 1)

            # MLE.append(ll)
            recon.append(test_nll)
            reglos.append(test_kld_loss)
            lowerbound.append(test_cost)

        if len(preds) < 1:
            tars = y
            preds = x[:, -steps:, :]
            ins = x
            pars = P
        else:
            tars = np.vstack([tars, y])
            preds = np.vstack([preds, x[:, -steps:, :]])
            ins = np.vstack([ins, x])
            pars = np.vstack([pars, P])

        with tf.Session() as sess:
            print(np.shape(y))
            print(np.shape(preds))

            mean_mmd, mean_that = [], []
            for s in [0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
                mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(y[:,-steps:,:], 
                                preds, biased=False, sigmas=sigma), feed_dict={sigma:np.ones(1)*s})
                mean_mmd.append(mmd2)
                mean_that.append(that_np)

            print('MMDs : ', mean_mmd, 'MMD : ', np.mean(mean_mmd), ' THATs: ', mean_that, ' THAT: ', np.mean(mean_that))
            print('MLE: ', MLE)
            print('reglos', reglos)
            print('recon', recon)
            print('lowerbound', lowerbound)

        predictions, targets, mse, win, lose, total_return  = self._check_performance(pars, tars)

        # return predictions, targets, dates[:len(tars), -2],  mse, win, lose, total_return

    def _simulate(self, steps = 5):
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        mask = np.zeros([receptive_field, 200])
        mask[receptive_field-1:, :] = 1
        skip = self._config.input_seq_length - receptive_field
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config, skip = skip)
        ###### first build the graph again ########
        sigma = tf.placeholder(tf.float32, shape=1, name='sigma')
        x, y,  dates = self._dataset.get_validation_set()
        x = x[:,:,:]
        y = y[:,:,:]
        MMD = []
        THAT = []
        N_batches = 10#len(x)//self._config.batch_size
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+'990.pth.tar', map_location='cpu')
        self._model.net.load_state_dict(state['state_dict'])
        # for each batch get predictions
        for i in range(N_batches):
            
            print('simulation ', i)
            print('Testing ... ', (i/N_batches)*100 ,' %')
            index = i*self._config.batch_size
            batch_x = x[index:index+self._config.batch_size, :, :]
            batch_y = y[index:index+self._config.batch_size, :, :]
            a = .2
            seed = i
            np.random.seed(seed)
            tf.set_random_seed(seed)
            X = batch_x
            for step in range(steps):
                np.random.seed(step)
                torch.manual_seed(step)
                _, _, _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], X[:, step:step+receptive_field, :], self._model.net, mask)
                
                locs = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:, :]))).float()
                logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                Dist = torch.distributions.normal.Normal(locs[:,-1:,:], logvars[:,-1:,:], validate_args=None)
                test_pred = Dist.sample()
                X = np.concatenate([X, locs], axis = 1)

            with tf.Session() as sess:
                mean_mmd, mean_that = [], []
                for s in [0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
                    mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(batch_y, 
                                    X[:,1:,:], biased=False, sigmas=sigma), feed_dict={sigma:np.ones(1)*s})
                    MMD.append(mmd2)
                    THAT.append(that_np)
            if i == 0:
                VAR_real = np.std(batch_y, axis=0)
                VAR_fake = np.std(X[:,1:,:], axis=0)
            else:
                VAR_real += np.std(batch_y, axis=0)
                VAR_fake += np.std(X[:,1:,:], axis=0)
            
            fig1 = plt.figure('SWN_mode__X'+str(i))
            plt.style.use('dark_background')
            plt.plot(X[1:,:,0].T, alpha = a, c='c')
            plt.axvline(x=receptive_field)
            fig1.savefig('SWN_mode_X'+str(i)+'.png')

            fig2 = plt.figure('SWN_mode__Y'+str(i))
            plt.style.use('dark_background')
            plt.plot(batch_y[:,:,0].T, alpha = 0.1, c='y')
            fig2.savefig('SWN_mode_Y'+str(i)+'.png')

            # plt.show()

        fig = plt.figure('variance_mode')
        plt.plot(VAR_fake/10.0, label='fake')
        plt.plot(VAR_real/10.0, label='real')
        plt.legend()
        plt.show()
        print('DIFF: ', np.sum((VAR_fake - VAR_real))/(10.0*steps))
        print('Mean MMD: ', np.mean(MMD))
        print('Mean That: ', np.mean(THAT))


 

                

    @staticmethod
    def gaussian_loss(y_hat, y, log_scale_min=float(np.log(1e-14)), reduce=True):

        mean = y_hat[:,:,0:1]; logvar = y_hat[:,:,1:];
        logvar = tf.clip_by_value(logvar, -12., 12.)
        var = tf.exp(logvar);
        dis = tf.contrib.distributions.Normal(loc=mean, scale = tf.exp(logvar))
        log_prob = dis.log_prob(y)
        if reduce:
            log_prob = tf.reduce_mean(log_prob, axis=0)
        return log_prob

    def ll(self, x, pars):

        P = tf.placeholder(tf.float32, np.shape(pars),  name = 'scale')
        tars = tf.placeholder(tf.float32, np.shape(x),  name = 'logvar')
        log_prob = self.gaussian_loss(P, tars)
        with tf.Session() as sess:
            ll = sess.run(log_prob, feed_dict={P:pars, tars:x})
        return ll


    def _check_performance(self,pars,tars, show = True):
        '''
        Check the test accuracy based on investment threshold 0.5
        inputs
        - price index : column of the targets feature (the price)
        - show: boolean, if true then it will show some samples of sequences

        outputs
        - mse: mean squared error between predicted and true
        - f1 score
        - true positives, false positives, true negatives, false negatives
        '''
        print('joooe')
        print(np.shape(tars))
        print(np.shape(pars))       
        MLE = self.ll(tars[:,-np.shape(pars)[1]:,0], pars)
        print('MLE: ', MLE)
        plt.plot(np.mean(MLE, axis=0))
        plt.show()


        # if self._config.conversion == 'return':
        #     predicted_delta = 100*predictions[:, 0]
        #     true_delta = 100*targets
        # elif self._config.conversion == 'log_return':
        #     predicted_delta = 100* (np.exp(predictions[:, 0, 0]) - 1)
        #     true_delta = 100* (np.exp(targets) - 1)

        # mse = np.mean(np.power(predictions[:,:,0] - targets[:, -np.shape(predictions)[1]:, 0], 2))
        # print('MSE', mse)

        
        if show:
            for _ in range(10):
                i = np.random.randint(0, len(targets)-1)
                plt.plot(np.arange(np.shape(inputs)[1], np.shape(inputs)[1]+np.shape(predictions)[1]), predictions[i, :], label= 'predictions')
                plt.plot(np.arange(np.shape(inputs)[1], np.shape(inputs)[1]+np.shape(predictions)[1]), targets[i, -np.shape(predictions)[1]:, 0], label = 'targets')
                plt.plot(inputs[i, :, price_index], label = 'inputs')
                plt.legend()
                plt.show()
        
        

