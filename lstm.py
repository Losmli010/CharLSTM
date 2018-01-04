# coding: utf-8

import numpy as np
from utils import *

class LSTMLM:
    def __init__(self,word_dim,hidden_dim=128,bptt_truncate=-1):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        self.Wf = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, word_dim))
        self.Uf = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        self.bf=np.zeros(hidden_dim)
        self.Wi = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, word_dim))
        self.Ui = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        self.bi=np.zeros(hidden_dim)
        self.Wg = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, word_dim))
        self.Ug = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        self.bg=np.zeros(hidden_dim)
        self.Wo = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, word_dim))
        self.Uo = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, hidden_dim))
        self.bo=np.zeros(hidden_dim)
        self.V = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (word_dim, hidden_dim))
        self.b=np.zeros(word_dim)
        
    
    def forward_propagation(self,x):
        T=len(x)
        f=np.zeros((T,self.hidden_dim))
        i=np.zeros((T,self.hidden_dim))
        g=np.zeros((T,self.hidden_dim))
        c=np.zeros((T+1,self.hidden_dim))
        o=np.zeros((T,self.hidden_dim))
        h=np.zeros((T+1,self.hidden_dim))
        z=np.zeros((T,self.word_dim))
        
        for t in np.arange(T):
            inputs=np.zeros(self.word_dim)
            inputs[x[t]]=1
            f[t]=sigmoid(np.dot(self.Wf,inputs)+np.dot(self.Uf,h[t-1])+self.bf)
            i[t]=sigmoid(np.dot(self.Wi,inputs)+np.dot(self.Ui,h[t-1])+self.bi)
            g[t]=np.tanh(np.dot(self.Wg,inputs)+np.dot(self.Ug,h[t-1])+self.bg)
            c[t]=f[t]*c[t-1]+i[t]*g[t]
            o[t]=sigmoid(np.dot(self.Wo,inputs)+np.dot(self.Uo,h[t-1])+self.bo)
            h[t]=o[t]*np.tanh(c[t])
            z[t] = softmax(np.dot(self.V,h[t])+self.b)
        return (f,i,g,c,o,h,z)
    
    def predict(self,x):
        f,i,g,c,o,h,z=self.forward_propagation(x)
        return np.argmax(z,axis=1)
    
    def calculate_loss(self,x,y):
        f,i,g,c,o,h,z=self.forward_propagation(x)
        loss=0.0
        for i in np.arange(len(y)):
            correct_word_predictions = z[i, y[i]]
            loss+=-1.0*np.log(correct_word_predictions)
        return loss
    
    def calculate_total_loss(self,X,Y):
        L=0.0
        N=0
        for i in np.arange(len(Y)):
            L+=self.calculate_loss(X[i],Y[i])
            N+=len(Y[i])
        return L/N
    
    def bptt(self, x, y):
        T = len(y)
        f,i,g,c,o,h,z = self.forward_propagation(x)
    
        dWf = np.zeros(self.Wf.shape)
        dUf = np.zeros(self.Uf.shape)
        dbf = np.zeros(self.bf.shape)
        dWi = np.zeros(self.Wi.shape)
        dUi = np.zeros(self.Ui.shape)
        dbi = np.zeros(self.bi.shape)
        dWg = np.zeros(self.Wg.shape)
        dUg = np.zeros(self.Ug.shape)
        dbg = np.zeros(self.bg.shape)
        dWo = np.zeros(self.Wo.shape)
        dUo = np.zeros(self.Uo.shape)
        dbo = np.zeros(self.bo.shape)
        dV = np.zeros(self.V.shape)
        db = np.zeros(self.b.shape)
        
        delta_z = z
        delta_z[np.arange(len(y)), y] -= 1.0
        
        delta_h=np.zeros(h.shape)
        delta_c=np.zeros(c.shape)
        for t in np.arange(T)[::-1]:
            dV += np.outer(delta_z[t], h[t].T)
            db += delta_z[t]

            delta_h[t] = np.dot(self.V.T,delta_z[t])+delta_h[t+1]
            delta_o = delta_h[t]*np.tanh(c[t])
            delta_c[t] = delta_h[t]*o[t]*(1-np.tanh(c[t])**2)+delta_c[t+1]
            
            delta_i=delta_c[t]*g[t]*i[t]*(1-i[t])
            delta_g=delta_c[t]*i[t]*(1-g[t]**2)
            delta_f=delta_c[t]*c[t-1]*f[t]*(1-f[t])
            delta_o_net=delta_o*o[t]*(1-o[t])
                    
            inputs=np.zeros(self.word_dim)
            inputs[x[t]]=1
            dWf +=np.outer(delta_f,inputs.T)
            dUf +=np.outer(delta_f,h[t-1].T)
            dbf +=delta_f
            dWi +=np.outer(delta_i,inputs.T)
            dUi +=np.outer(delta_i,h[t-1].T)
            dbi +=delta_i
            dWg +=np.outer(delta_g,inputs.T)
            dUg +=np.outer(delta_g,h[t-1].T)
            dbg +=delta_g
            dWo +=np.outer(delta_o_net,inputs.T)
            dUo +=np.outer(delta_o_net,h[t-1].T)
            dbo +=delta_o_net
           
        return (dWf,dUf,dbf,dWi,dUi,dbi,dWg,dUg,dbg,dWo,dUo,dbo,dV,db)
    
    def sgd_step(self, x, y, learning_rate):
        dWf,dUf,dbf,dWi,dUi,dbi,dWg,dUg,dbg,dWo,dUo,dbo,dV,db = self.bptt(x, y)
        self.Wf -= learning_rate * dWf
        self.Uf -= learning_rate * dUf
        self.bf -= learning_rate * dbf
        self.Wi -= learning_rate * dWi
        self.Ui -= learning_rate * dUi
        self.bi -= learning_rate * dbi
        self.Wg -= learning_rate * dWg
        self.Ug -= learning_rate * dUg
        self.bg -= learning_rate * dbg
        self.Wo -= learning_rate * dWo
        self.Uo -= learning_rate * dUo
        self.bo -= learning_rate * dbo
        self.V -= learning_rate * dV
        self.b -= learning_rate * db