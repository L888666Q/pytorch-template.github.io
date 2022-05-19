# pytorch-template.github.io
深度学习pytorch框架

config.json: 配置文件

训练：
python3 train.py

测试：
python3 test.py -c 配置文件路径(和模型同一目录下) -r 模型路径

剪枝优化：
python3 prune.py --model saved/models/SPOOF-NET/ --save saved/models/SPOOF-NET/**/pruned.pth --percent 0.7
