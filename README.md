# FaceRecognition-tensorflow
基于TensorFlow训练的人脸识别神经网络

文章地址：https://www.cnblogs.com/mu---mu/p/FaceRecognition-tensorflow.html
Github地址：https://github.com/seathiefwang/FaceRecognition-tensorflow
##################################################################

我做的修改

1 为了兼容 Tensorflow 2.0. 修改了Tensorflow的引用代码：
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

2 按照原著者的建议，修改了is_my_face.py部分语句导致的Saver_path报错：
        
    #saver = tf.train.Saver()  
    #sess = tf.Session()  
    #saver.restore(sess, tf.train.latest_checkpoint('.'))  
    init_op = tf.initialize_all_variables() 
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        sess.run(init_op)
        save_path = saver.save(sess, "./tmp/model.ckpt")
        #print "Model saved in file: ",save_path
        saver.restore(sess, "./tmp/model.ckpt")
    
    # the following code lines are all backwards behind with tf.Session() as sess:
       
        def is_my_face(image):  
            res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
            if res[0] == 1:  
                return True  
            else:  
                return False  

        #使用dlib自带的frontal_face_detector作为我们的特征提取器
        detector = dlib.get_frontal_face_detector()

        cam = cv2.VideoCapture(0)  

        while True:  
            _, img = cam.read()  
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            if not len(dets):
                #print('Can`t get face.')
                cv2.imshow('img', img)
                key = cv2.waitKey(30) & 0xff  
                if key == 27:
                    sys.exit(0)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1,x2:y2]
                # 调整图片的尺寸
                face = cv2.resize(face, (size,size))
                print('Is this my face? %s' % is_my_face(face))

                cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
                cv2.imshow('image',img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)

        sess.close() 
