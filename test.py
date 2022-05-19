import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.mobilenetv3 as module_arch_mobilenetv3
from parse_config import ConfigParser

from math import exp
import datetime
import os 
from shutil import copyfile


def main(config):
    logger = config.get_logger('test')

    # 数据
    data_loader = getattr(module_data, config['test_data_loader']['type'])(
        config['test_data_loader']['args']['data_dir'],
        batch_size=config['test_data_loader']['args']['batch_size'],
        shuffle=config['test_data_loader']['args']['shuffle'],
        num_workers=config['test_data_loader']['args']['num_workers']
    )

    # 模型
    #VGG
    model = config.init_obj('arch', module_arch)
    #mobilenetv3
    #model = config.init_obj('arch_mobilenetv3', module_arch_mobilenetv3)
    logger.info(model)

    # 同样的
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    #if config['n_gpu'] > 1:
        #model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # 准备测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    img_num=0
    finger_num=0
    miss_num=0
    omit_num=0

    imgsort = 1
    thr_mode = 1
    finger_thr = 3277
    spoof_thr = 40000

    with torch.no_grad():
        for i, (data, target, path) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            time = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
            prediction = torch.max(output, 1)
            confusion_out_path = "./saved/confusion_test%s"%time
            if not os.path.isdir(confusion_out_path):
                os.mkdir(confusion_out_path)
            for i in range(len(target)):
                img_num += 1
                img_type = path[i].split('/')

                if target[i] == 0:
                    finger_num += 1
                #print(path[i])
                #print("\n%d: output[%d][0]:%f  output[%d][1]:%f path: %s \n"%(i, i, output[i][0], i, output[i][1], path[i]))
                if output[i][0] > output[i][1]:
                    c=output[i][0]
                else:
                    c=output[i][1]
                finger_score = exp(output[i][0] -c)/(exp(output[i][0]-c)+exp(output[i][1]-c))
                if thr_mode == 1:
                    finger_score = finger_score*65536
                    if target[i] == 0:
                        prediction.indices[i] = (finger_score < finger_thr)
                    else:
                        prediction.indices[i] = (finger_score < spoof_thr)
                #confusionmap[target[i]][prediction.indices[i]] += 1
                if target[i] != prediction.indices[i]:
                #if 1:
                    if imgsort:
                        target_path = "./saved/confusion_test%s/%d" % (time,target[i])
                        #if not os.path.isdir(target_path):
                            #os.mkdir(target_path)
          
                        img_type = path[i].split('/')
                        pathout=path[i].split('train/')[-1].replace('/', '_');
                        #print(pathout)
                        if target[i] == 0:
                            out_path = target_path
                            miss_num += 1
                        else:
                            out_path = target_path
                            omit_num += 1
                        if not os.path.isdir(out_path):
                            os.mkdir(out_path)
                        #copyfile(path[i], out_path + "/%03d.bmp"%num)
                        #copyfile(path[i], out_path + "/%d_%s"%(finger_score*65536,img_type[-1]))
                        copyfile(path[i], out_path + "/%d_%s"%(finger_score,pathout))
                        #if target[i] == 0:
                        #os.remove(path[i])
                        #copyfile(path[i], out_path + "/%s"%(img_type[-1]))
                        #move(path[i], out_path + "/%s"%(img_type[-1]))
                        #os.remove(path[i])
                    #if logprint: 
                        #log.append("\ntarget:%d  num:%d path: %s \n"%(target[i], num, path[i]))
                    #num += 1;
            #

            # computing loss, metrics on test set
            #loss = loss_fn(output, target)
            #my_loss = loss_fn()
            #loss=my_loss(output,target)
            batch_size = data.shape[0]
            #total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    notfinger=img_num-finger_num;
    logger.info("img_num=%d,finger_num:%d,notfinger_num:%d,miss_num=%d,omit_num=%d"%(img_num,finger_num,notfinger,miss_num,omit_num))
    #print("\nimg_num=%d,finger_num:%d,notfinger_num:%d,miss_num=%d,omit_num=%d\n"%(img_num,finger_num,notfinger,miss_num,omit_num))

    if(finger_num==0):
        finger_num=1;
    if(notfinger==0):
        notfinger=1;
    #print("\naccuracy:%f  miss_percent:%f  omit_percent:%f \n"%((img_num-miss_num-omit_num)/img_num*100,miss_num/finger_num*100, (omit_num)/(notfinger)*100))
    logger.info("accuracy:%f  miss_percent:%f  omit_percent:%f"%((img_num-miss_num-omit_num)/img_num*100,miss_num/finger_num*100, (omit_num)/(notfinger)*100))

    n_samples = len(data_loader.sampler)
    log = {'loss': 0 / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
