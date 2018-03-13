Here I give some instruction to install TensorFlow with Virtualenv, other installation methods can refer to [tensorflow document](https://tensorflow.google.cn/install/).

1. Install pip and Virtualenv
```
$ sudo apt-get install python-pip python-dev python-virtualenv
```

2. Creat a Virtualenv environment at ```~/tf```
```
$ virtualenv --system-site-packages ~/tf
```

3. Activate the Virtualenv enviroment by change your prompt to the following:
```
$ source ~/tf/bin/activate
```

4. Install Tensorflow in the active Virtualenv enviroment
```
pip install --upgrade tensorflow # for cpu
pip install --upgrade tensorflow-gpu # for gpu
```

## Tips
1. Install the specified version like 1.2.1:
    - ```pip install tensorflow-gpu==1.2.1```
2. Install TF from the specified pipi source such as ```pypi.douban.com```
    - ```pip install -i http://pypi.douban.com/simple/ tensorflow-gpu```
