# A Deep-Learning-Based Chinese Speech Recognition System
基于深度学习的中文语音识别系�?

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

ReadMe Language 中文�?[English](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md) 

[查看本项目的Wiki页面](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki) (正在完善�?

## Introduction 简�?

本项目使用Keras、TensorFlow基于深度卷积神经网络和长短时记忆神经网络、注意力机制以及CTC实现�?

This project uses Keras, TensorFlow based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement.

本项目目前已经可以正常进行训练了�?

通过git克隆仓库以后，需要将datalist目录下的文件全部拷贝到dataset目录下，也就是将其跟数据集放在一起�?
```shell
$ cp -rf datalist/* dataset/
```

目前可用的模型有24�?5�?51

本项目开始训练请执行�?
```shell
$ python3 train_mspeech.py
```
本项目开始测试请执行�?
```shell
$ python3 test_mspeech.py
```
测试之前，请确保代码中填写的模型文件路径存在�?

ASRT API服务器启动请执行�?
```shell
$ python3 asrserver.py
```

如果要训练和使用模型251，请在代码中 `import SpeechModel` 的相应位置做修改�?

如果程序运行期间或使用中有什么问题，可以及时在issue中提出来，我将尽快做出答复�?

提问前可以先 [查看常见问题](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/issues) 

## Model 模型

### Speech Model 语音模型

CNN + LSTM/GRU + CTC

* 关于下载已经训练好的模型的问�?

可以在Github本仓库下[releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases)里面的查看发布的各个版本软件的压缩包里获得完整源程序�?

### Language Model 语言模型

基于概率图的最大熵隐马尔可夫模�?

## About Accuracy 关于准确�?

当前，最好的模型在测试集上基本能达到80%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做�?7%，所以正确率仍有待于进一步提�?

* 目前可知的可以继续提高准确率的一个方案就是纠正数据集标注错误，尤其是ST-CMDS里面关于syllable文件中拼音的错误，这里面有一定比例的错误标注，如果走过路过的各位有意愿尽自己的能力帮助纠正一些数据标注错误的，我将非常欢迎，可以通过提交Pull Request来纠正，并且将登上本仓库的贡献者名单�?

样例：`不是�?bu4 shi4 -> bu2 shi4` `一个：yi1 ge4 -> yi2 ge4` `了解：le5 jie3 -> liao3 jie3`

* 已订正部分：

ST-CMDS

train:  20170001P00001A    20170001P00001I    20170001P00002A

## Python Import
Python的依赖库

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py

## Data Sets 数据�?
* 清华大学THCHS30中文语音数据�?

data_thchs30.tgz 
<http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz>
<http://www.openslr.org/resources/18/data_thchs30.tgz>

test-noise.tgz 
<http://cn-mirror.openslr.org/resources/18/test-noise.tgz>
<http://www.openslr.org/resources/18/test-noise.tgz>

resource.tgz 
<http://cn-mirror.openslr.org/resources/18/resource.tgz>
<http://www.openslr.org/resources/18/resource.tgz>

* Free ST Chinese Mandarin Corpus

ST-CMDS-20170001_1-OS.tar.gz 
<http://cn-mirror.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>
<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>

特别鸣谢！感谢前辈们的公开语音数据�?

如果提供的数据集链接无法打开和下载，请点击该链接 [OpenSLR](http://www.openslr.org)

## Log
日志

链接：[进展日志](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)

