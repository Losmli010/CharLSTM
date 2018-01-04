# coding: utf-8

import numpy as np
import sys

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def softmax(x):
    values = np.exp(x - np.max(x))
    return values/ np.sum(values)

def save_model_parameters(model, outfile):
    np.savez(outfile,
        Wf=model.Wf,
        Uf=model.Uf,
        bf=model.bf,
        Wi=model.Wi,
        Ui=model.Ui,
        bi=model.bi,
        Wg=model.Wg,
        Ug=model.Ug,
        bg=model.bg,
        Wo=model.Wo,
        Uo=model.Uo,
        bo=model.bo,
        V=model.V,
        b=model.b)
    print("Saved model parameters to %s." % outfile)

def load_model_parameters(model,path):
    npzfile = np.load(path)
    Wf,Uf,bf,Wi,Ui,bi,Wg,Ug,bg,Wo,Uo,bo,V,b = npzfile["Wf"], npzfile["Uf"], npzfile["bf"], npzfile["Wi"], npzfile["Ui"], npzfile["bi"],npzfile["Wg"], npzfile["Ug"], npzfile["bg"], npzfile["Wo"], npzfile["Uo"], npzfile["bo"],npzfile['V'],npzfile['b']
    hidden_dim, word_dim = Wf.shape
    print("Building model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim))
    sys.stdout.flush()
    model = model(word_dim, hidden_dim=hidden_dim)
    model.Wf=Wf,
    model.Uf=Uf,
    model.bf=bf,
    model.Wi=Wi,
    model.Ui=Ui,
    model.bi=bi,
    model.Wg=Wg,
    model.Ug=Ug,
    model.bg=bg,
    model.Wo=Wo,
    model.Uo=Uo,
    model.bo=bo,
    model.V=V,
    model.b=b
    return model