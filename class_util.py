### Adversarial Surprise
### Arnaud Fickinger, 2021

import inspect
from functools import wraps

import pdb

def member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__: 
    :returns: 
    :rtype: 

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)
    return wrapper

def hidden_member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__: 
    :returns: 
    :rtype: 

    """

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, '_' + name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, '_' + names[index]):
                    setattr(self, '_' + names[index], defaults[index])

        wrapped__init__(self, *args, **kargs)
    return wrapper

def tensor_member_initialize(wrapped__init__):
    """Decorator to initialize members of a class with the named arguments. (i.e. so D.R.Y. principle is maintained
    for class initialization).

    Modified from http://stackoverflow.com/questions/1389180/python-automatically-initialize-instance-variables
    :param wrapped__init__: 
    :returns: 
    :rtype: 

    """
    import tensorflow as tf

    names, varargs, keywords, defaults = inspect.getargspec(wrapped__init__)

    @wraps(wrapped__init__)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, tf.compat.v1.convert_to_tensor(arg))

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], tf.compat.v1.convert_to_tensor(defaults[index]))
        wrapped__init__(self, *args, **kargs)
    return wrapper

class classproperty(object):
    def __init__(self, f):
        """Decorator to enable access to properties of both classes and instances of classes

        :param f: 
        :returns: 
        :rtype: 

        """
        
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)
