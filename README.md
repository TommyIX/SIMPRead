## SIMPRead | 易读

#### 介绍

基于piggyreader，TextSUM (Based on LSTM / Attention Deep-learning Model)与Flask开发的Chrome插件，用于实时精简网站内容，并以阅读器方式展现，便于用户快速阅读。



#### 使用方式

运行本地Server，在本地虚拟环境上安装好requirements.txt中的内容，比如使用：

```bash
conda create -n SIMPRead python=3.8
conda activate SIMPRead
pip install -r requirements.txt
```

然后[前往老王的谷歌硬盘](https://drive.google.com/drive/folders/1F2UK9hFVd3NTXSf-17SWT2EY3Or9_rsW?usp=sharing)，下载model.pth，放在与SIMPRead_server.py相同的目录下，并新建/data目录，放置data下的3个csv文件。

```bash
python SIMPRead_server.py
```

这时，SIMPRead服务器将会自动加载模型，加载完成后，将会持续监听本地8999端口，此时运行插件端，插件的简化请求将会被服务器接受并处理。



#### 特别鸣谢

十分感谢piggyreader原作者公开的阅读器基础代码，本代码仅作为学术探究成果的Demo展示，不做任何其它商业用途，不在任何应用商店上发布以及设置长期服务。如果您支持本项目，请支持piggyreader的作者：[huntbao/piggyreader: Chrome 插件 Piggy Reader (github.com)](https://github.com/huntbao/piggyreader)

