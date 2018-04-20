import tensorflow as tf

class Model:
    def __init__(self, dim_feature, n_nodes_hl):
        n_nodes_hl1 = n_nodes_hl[0]
        n_nodes_hl2 = n_nodes_hl[1]
        n_nodes_out = 1

        def init_random_tensor(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.01, seed=0), name=name)

        def init_const_tensor(shape, value, name):
            return tf.Variable(value*tf.ones(shape), name=name)

        def batch_matmul(X,W,batch_size):
            W_shape = tf.shape(W)
            W = W + tf.zeros(shape=[batch_size,W_shape[0],W_shape[1]]) # Broadcast along batch dimension
            return tf.matmul(X,W)

        def model(G,w_h1,w_h2,w_out,b_h1,b_h2,b_out):
            this_batch_size = tf.shape(G)[0] # Using dynamic shape [https://blog.metaflow.fr/shapes-and-dynamic-dimensions-in-tensorflow-7b1fe79be363]
            with tf.name_scope("layer1"):
                h1 = batch_matmul(G,w_h1,this_batch_size) + b_h1 # E.g. (?,24,2)*(?,2,50) + (50) = (?,24,50)
                h1 = tf.layers.batch_normalization(h1, training=is_training)
                h1 = tf.nn.tanh(h1)
                h1 = tf.nn.dropout(h1, keep_prob_h1)
            with tf.name_scope("layer2"):
                h2 = batch_matmul(h1,w_h2,this_batch_size) + b_h2
                h2 = tf.layers.batch_normalization(h2, training=is_training)
                h2 = tf.nn.tanh(h2)
                h2 = tf.nn.dropout(h2, keep_prob_h2)
            with tf.name_scope("layer_out"): # Here the energies of each of the atoms should show up
                out = batch_matmul(h2,w_out,this_batch_size) + b_out
                tf.identity(out, "out")
            with tf.name_scope("sigma_E"): # ... the sum of energies is the total energy
                #out = tf.Print(out, [out], "out = ", summarize=24)
                out = tf.reshape(out, [this_batch_size,24]) # Reshape back to (?,24)
                #out = tf.Print(out, [out], "outReshaped = ", summarize=24)
                E_tot = tf.reduce_sum(out,axis=1, keepdims=True) # (?,1)
                #E_tot = tf.Print(E_tot, [E_tot[0]], "E_tot[0] = ", summarize=24)
                return E_tot

        # Init data tensors
        G = tf.placeholder('float',[None, 24, dim_feature], name="G")
        E = tf.placeholder('float',[None, 1], name="E")
        #batch_size = tf.Placeholder('int', [1], name="batch_size") # TODO: use G.set_shape instead of batch_matmul!
        keep_prob_h1 = tf.placeholder(tf.float32, name="keep_prob_h1")
        keep_prob_h2 = tf.placeholder(tf.float32, name="keep_prob_h2")
        is_training = tf.placeholder(tf.bool, name="is_training")

        # Init weights and biases
        w_h1 = init_random_tensor([dim_feature,n_nodes_hl1], "w_h1")
        w_h2 = init_random_tensor([n_nodes_hl1,n_nodes_hl2], "w_h2")
        w_out = init_random_tensor([n_nodes_hl2,n_nodes_out], "w_out")
        b_h1 = init_random_tensor([n_nodes_hl1], "b_h1")
        b_h2 = init_random_tensor([n_nodes_hl2], "b_h2")
        b_out = init_random_tensor([n_nodes_out], "b_out")

        # Add histogram summaries for the weights and biases
        with tf.name_scope("weights"):
            tf.summary.histogram("w_h1", w_h1)
            tf.summary.histogram("w_h2", w_h2)
            tf.summary.histogram("w_out", w_out)
        with tf.name_scope("biases"):
            tf.summary.histogram("b_h1", b_h1)
            tf.summary.histogram("b_h2", b_h2)
            tf.summary.histogram("b_out", b_out)

        # Create the model
        E_G = tf.identity(model(G,w_h1,w_h2,w_out,b_h1,b_h2,b_out), name="E_G")

        # Create cost function
        with tf.name_scope("loss"):
            # Ensures that batch normalization is back-prob. through as well
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                loss = tf.square(tf.losses.absolute_difference(labels=E,predictions=E_G))
                tf.identity(loss, name="absolute_difference_squared")
                train_optimzer = tf.train.AdamOptimizer().minimize(loss)
                #train_optimzer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
                #tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="Adam")
                tf.summary.scalar("absolute_difference_squared", loss) # Add summary to tensorboard

        with tf.name_scope("accuracy"):
            errors = tf.divide(E - E_G, E)
            accuracy = (1 - tf.reduce_mean(errors)) * 100 # Percentage
            tf.identity(accuracy, name="pct_accuracy")
            tf.summary.scalar("pct_accuracy",accuracy)

        self.loss = loss
        self.train_optimzer = train_optimzer
        self.E = E
        self.G = G
        self.E_G = E_G
        self.dim_feature = dim_feature
        self.keep_prob_h1 = keep_prob_h1
        self.keep_prob_h2 = keep_prob_h2
        self.is_training = is_training

    def nodes_in_graph(self):
        return tf.get_default_graph().as_graph_def().node

class Model2:
    def __init__(self, dim_feature, n_nodes_hl):
        n_nodes_out = 1

        def init_random_tensor(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.01, seed=0), name=name)

        def init_const_tensor(shape, value, name):
            return tf.Variable(value*tf.ones(shape), name=name)

        def model(G):
            
            with tf.name_scope("layer_in"):
                layer = tf.contrib.layers.fully_connected(inputs = G, # (?,24,fea_dim)
                                                          num_outputs = n_nodes_hl[0],
                                                          biases_initializer=tf.contrib.layers.xavier_initializer(), scope = 'fc_in')

            for i in range(1,len(n_nodes_hl)):
                with tf.name_scope("layer_" + str(i)):
                    layer = tf.contrib.layers.fully_connected(inputs = layer,
                                                              num_outputs = n_nodes_hl[i],
                                                              biases_initializer=tf.contrib.layers.xavier_initializer(), scope = 'fc_' + str(i))

            with tf.name_scope("layer_out"): # Here the energies of each of the atoms should show up
                out = tf.contrib.layers.fully_connected(inputs = layer,
                                                        num_outputs = 1,
                                                        #biases_initializer=tf.constant_initializer(value=-180),
                                                        activation_fn = None, scope = 'fc_out')
                tf.identity(out, "out")
                print(out)

            with tf.name_scope("sigma_E"): # ... the sum of energies is the total energy
                E_tot = tf.reduce_sum(out,axis=1) # (?,24,1) -> (?,1)
                #E_tot = tf.Print(E_tot, [E_tot[0]], "E_tot[0] = ", summarize=24)
                print(E_tot)
                return E_tot

        # Init data tensors
        G = tf.placeholder('float',[None, 24, dim_feature], name="G")
        E = tf.placeholder('float',[None, 1], name="E")
        #batch_size = tf.Placeholder('int', [1], name="batch_size") # TODO: use G.set_shape instead of batch_matmul!
        keep_prob_h1 = tf.placeholder(tf.float32, name="keep_prob_h1")
        keep_prob_h2 = tf.placeholder(tf.float32, name="keep_prob_h2")
        is_training = tf.placeholder(tf.bool, name="is_training")

        # Create the model
        E_G = tf.identity(model(G), name="E_G")

        # Add histogram for weights and biases
        all_vars = tf.global_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None

        tf.summary.histogram("w_in", get_var("fc_in/weights"))
        tf.summary.histogram("b_in", get_var("fc_in/bias"))
        for i in range(1,len(n_nodes_hl)):
            tf.summary.histogram("w_"+str(i), get_var("fc_" + str(i) + "/weights"))
            tf.summary.histogram("b_"+str(i), get_var("fc_" + str(i) + "/bias"))
        tf.summary.histogram("w_out", get_var("fc_out/weights"))
        tf.summary.histogram("b_out", get_var("fc_out/bias"))

        # Create cost function
        with tf.name_scope("loss"):
            # Ensures that batch normalization is back-prob. through as well
            loss = tf.losses.mean_squared_error(labels=E,predictions=E_G)
            #loss = tf.Print(loss,[E],"E = ")
            loss = tf.Print(loss,[E_G],"E_G = ")
            loss = tf.Print(loss,[loss],"loss = ")
            # loss = tf.Print(loss, [E], "E = ", summarize=50)
            # loss = tf.Print(loss, [E_G], "E_G = ", summarize=50)
            # loss = tf.Print(loss, [loss], "loss = ")
            # loss = tf.Print(loss, [], "")
            tf.identity(loss, name="mean_squared_error")
            train_optimzer = tf.train.AdamOptimizer(0.01).minimize(loss)
            #train_optimzer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
            #tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="Adam")
            tf.summary.scalar("mean_squared_error", loss) # Add summary to tensorboard

        with tf.name_scope("accuracy"):
            errors = tf.divide(E - E_G, E)
            accuracy = (1 - tf.reduce_mean(errors)) * 100 # Percentage
            tf.identity(accuracy, name="pct_accuracy")
            tf.summary.scalar("pct_accuracy",accuracy)

        self.loss = loss
        self.train_optimzer = train_optimzer
        self.E = E
        self.G = G
        self.E_G = E_G
        self.dim_feature = dim_feature
        self.keep_prob_h1 = keep_prob_h1
        self.keep_prob_h2 = keep_prob_h2
        self.is_training = is_training
