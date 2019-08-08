import tensorflow as tf


class Optimization:

    @staticmethod
    def parse(lr_method):
        if "-" in lr_method:
            name_cvt = {'lr': 'learning_rate'}
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {} 
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                name = name_cvt.get(split[0], split[0])
                lr_method_parameters[name] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        return lr_method_name, lr_method_parameters

    @staticmethod
    def opt(lr_method):
        method, prmt = Optimization.parse(lr_method)
        if method == 'sgd':
            optmz = tf.train.GradientDescentOptimizer(**prmt)
        elif method == 'adagrad':
            optmz = tf.train.AdagradOptimizer(**prmt)
        else:
            raise ValueError('Invalid opt')

        return optmz

    @staticmethod
    def apply_opt(lr_method, loss, var_list=None, clip=None):
        optmz = Optimization.opt(lr_method)
        '''
        gvs = optmz.compute_gradients(loss, var_list)
        if clip:
            # gvs = [(tf.clip_by_value(grad, clip), var) for grad, var in gvs]
            gvs = tf.clip_by_global_norm(gradients,max_gradient_norm)
        train_op = optmz.apply_gradients(gvs)
        '''

        gradients, variables = zip(*optmz.compute_gradients(loss, var_list))
        if clip:
            gradients, _ = tf.clip_by_global_norm(gradients, clip)
        train_op = optmz.apply_gradients(zip(gradients, variables))

        return train_op, optmz
