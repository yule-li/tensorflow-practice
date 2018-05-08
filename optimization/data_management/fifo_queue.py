import tensorflow as tf

def first_fifo():
    q = tf.FIFOQueue(3,"float")
    #init = q.enqueue_many(([0.,0.,0.],[5.,5.,5.]))
    init = q.enqueue_many(tf.convert_to_tensor([0.,0.,0.],dtype="float"))
    x = q.dequeue()
    y = x+1
    q_inc = q.enqueue([y])
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
        print(sess.run([q_inc,y]))
    
def fifo_queue_no_coord():
    dummpy_input = tf.random_normal([3],mean=0,stddev=1)
    dummpy_input = tf.Print(dummpy_input, data=[dummpy_input],
                                message="New dummpy inputs have been created: ", summarize=6)

    q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
    enqueue_op = q.enqueue_many(dummpy_input)
    data = q.dequeue()
    data = tf.Print(data,data=[q.size()],message="This is how many in q: ")
    fg = data + 1
    with tf.Session() as sess:
        sess.run(enqueue_op)
        sess.run(fg)
        sess.run(fg)
        sess.run(fg)
        sess.run(fg)
        print("Can not run here!")

def fifo_queue_with_coord():
    dummpy_input = tf.random_normal([3],mean=0,stddev=1)
    dummpy_input = tf.Print(dummpy_input, data=[dummpy_input],
                                message="New dummpy inputs have been created: ", summarize=6)

    q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
    enqueue_op = q.enqueue_many(dummpy_input)
    qr = tf.train.QueueRunner(q,[enqueue_op]*1)
    tf.train.add_queue_runner(qr)
    data = q.dequeue()
    data = tf.Print(data,data=[q.size()],message="This is how many in q: ")
    fg = data + 1
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(enqueue_op)
        sess.run(fg)
        sess.run(fg)
        sess.run(fg)
        sess.run(fg)
        print("We run here!")
        coord.request_stop()
        coord.join(threads)

def main():
    #first_fifo()
    #fifo_queue_no_coord()
    fifo_queue_with_coord()

if __name__ == '__main__':
    main()
