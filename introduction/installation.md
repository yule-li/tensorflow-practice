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
