#!/usr/bin/env python
import numpy
from nlptools.text import Vocab

def demo_seq():
    training_data = [('what is the illuminati'.split(), 'a world wide conspiracy'.split()),
             ('what is the illuminatti'.split(), 'a secret society that has supposedly existed for centuries'.split()),
             ('what is vineland'.split(), 'vineland is a novel by thomas pynchon'.split()),
             ('what is illiminatus'.split(), 'alleged world wide conspiracy theory'.split()),
             ('who wrote vineland'.split(), 'thomas pynchon'.split()),
             ('who is bilbo baggins'.split(), "is a character in tolkein's lord of the rings".split()),
             ('who is geoffrey chaucer'.split(), 'chaucer is best known for his canterbury tales'.split()),
             ('who are the illuminati'.split(), 'who is geoffrey chaucer'.split()),
             ('who is piers anthony'.split(), 'author of canturbury tales'.split())]
    vocab = Vocab()
    
    inputs, outputs = zip(*training_data)

    inputs = vocab(inputs, batch=True)
    outputs = vocab(outputs, batch=True)
    
    inputs = numpy.asarray([numpy.pad(i, (0, 1), 'constant', constant_values=Vocab.EOS_ID) for i in inputs], dtype=numpy.object)
    prev_outputs = numpy.asarray([numpy.pad(o, (1, 0), 'constant', constant_values=Vocab.BOS_ID) for o in outputs], dtype=numpy.object)
    outputs = numpy.asarray([numpy.pad(o, (0, 1), 'constant', constant_values=Vocab.EOS_ID) for o in outputs], dtype=numpy.object)

    vocab.reduce()

    return inputs, prev_outputs, outputs, vocab


if __name__ == '__main__':
    inputs, prev_outputs, outputs, vocab = demo_seq()
    print(inputs)
    print(prev_outputs)
    print(outputs)
 

