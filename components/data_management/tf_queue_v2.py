import tensorflow as tf
def main():
    #assemble a graph
    dummy_input = tf.random_normal([5],mean=0,stddev=1)

    dummy_input = tf.Print(dummy_input,[dummy_input],message='New dummy is created: ')

    q = tf.FIFOQueue(capacity=4, dtypes=tf.float32)
    enqueue_op = q.enqueue_many(dummy_input)
    qr = tf.train.QueueRunner(q,[enqueue_op]*1)
    tf.train.add_queue_runner(qr)
    data = q.dequeue()
    data = tf.Print(data,[q.size()],message='This is how many items are left in q: ')

    fg = data+1


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        #print(sess.run(dummy_input))
        #sess.run(enqueue_op)
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))

        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        print(sess.run(fg))
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    main()
