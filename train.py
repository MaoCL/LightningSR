import os.path as osp
# from train_pipeline_novisdom import train_pipeline
from train_pipeline_novisdom import train_pipeline
import archs
import data
import models


if __name__ == '__main__':
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path ='/home/zhangfang/zhangfang/CycleSRGAN-mao'
    train_pipeline(root_path)


