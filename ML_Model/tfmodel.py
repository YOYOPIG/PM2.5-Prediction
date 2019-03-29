import tensorflow as tf

# Variables to be trained
a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
c = tf.Variable([.3], dtype=tf.float32)
d = tf.Variable([.3], dtype=tf.float32)
e = tf.Variable([.3], dtype=tf.float32)
f = tf.Variable([.3], dtype=tf.float32)

# Declare datas as placeholder, changing when the session runs
pm10 = tf.placeholder(tf.float32)
pm25 = tf.placeholder(tf.float32)
pm100 = tf.placeholder(tf.float32)
temp = tf.placeholder(tf.float32)
humid = tf.placeholder(tf.float32)

# Prediction values
PM25_model = a * pm10 + b * pm25 + c * pm100 + d * temp + e * humid + f

y = tf.placeholder(tf.float32) # Real values

# Calculate loss function
loss = tf.reduce_sum(tf.square(PM25_model - y)) # sum of squares

# Optimize by gradient decent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
pm10_train = [1, 2, 3, 4]
pm25_train = [1, 2, 3, 4]
pm100_train = [1, 2, 3, 4]
temp_train = [1, 2, 3, 4]
humid_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Initialize the model and run
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training in loop
for i in range(1000):
  sess.run(train, {pm10:x_train, pm25:pm25_train, pm100:pm100_train,
  				   temp:temp_train, humid:humid_train, y:y_train})

# Output
cur_a, cur_b, cur_c, cur_d, cur_e, cur_f, curr_loss = sess.run([a, b, c, d, e, f, loss],
	{pm10:x_train, pm25:pm25_train, pm100:pm100_train, temp:temp_train, humid:humid_train, y:y_train})
print("a: %s b: %s c: %s d: %s e: %s f: %s loss: %s"%(cur_a, cur_b, cur_c, cur_d, cur_e, cur_f, curr_loss))