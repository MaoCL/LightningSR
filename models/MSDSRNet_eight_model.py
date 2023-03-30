import re
import numpy as np
import random
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from basicsr.utils import  USMSharp
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.metrics import calculate_metric
from basicsr.data.degradations import random_add_gaussian_noise_pt,random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from data.transforms import cut_image
from collections.abc import Iterable
import time
from thop import profile
from ptflops import get_model_complexity_info
# from visdom import Visdom
@MODEL_REGISTRY.register()
class MSDSRNet_eight_Model(BaseModel):
    """用来测试随机lensblur和新损失函数可行性,仅L1loss"""
    def __init__(self, opt):
        super(MSDSRNet_eight_Model, self).__init__(opt)
        self.batch = 21
        self.time = 0.0
        self.params = 0.0
        self.flops = 0.0
        self.hight = 0
        self. width = 0
        self.area = 0
        # define network
        self.net_P = build_network(opt['network_P'])
        self.net_P = self.model_to_device(self.net_P)

        self.net_U = build_network(opt['network_U'])
        self.net_U = self.model_to_device(self.net_U)
       
        # self.print_network(self.net_U)

        load_path4 = self.opt['path'].get('pretrain_network_U', None)
        if load_path4 is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_U, load_path4, self.opt['path'].get('strict_load_g', True), param_key)


        load_pathl = self.opt['path'].get('pretrain_network_P', None)
        if load_pathl is not None:
            param_key = self.opt['path'].get('param_key_p', 'params')
            self.load_network(self.net_P, load_pathl, self.opt['path'].get('strict_load_g', True), param_key)

        # load_pathu = self.opt['path'].get('pretrain_network_du', None)
        # if load_pathu is not None:
        #     param_key = self.opt['path'].get('param_key_g', 'params')
        #     self.load_network(self.net_du, load_pathu, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()



    def model_emaU(self,decay=0.999):
        net_g = self.get_bare_model(self.net_U)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_U_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def model_emaP(self,decay=0.999):
        net_g = self.get_bare_model(self.net_P)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_P_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            self.net_U_ema = build_network(self.opt['network_U']).to(self.device)
            # load pretrained model
            load_path4 = self.opt['path'].get('pretrain_network_U', None)
            if load_path4 is not None:
                self.load_network(self.net_U_ema, load_path4, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_emaU(0)  # copy net_g weight
            self.net_P_ema = build_network(self.opt['network_P']).to(self.device)
            # load pretrained model
            load_path1 = self.opt['path'].get('pretrain_network_P', None)
            if load_path1 is not None:
                self.load_network(self.net_P_ema, load_path1, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_emaP(0)  # copy net_g weight
            self.net_U_ema.eval()
        #-------------------------------------------------#
        self.net_P.train()
        self.net_U.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        self.cri_regress = torch.nn.CrossEntropyLoss().to(self.device)
       
        # if train_opt.get('regress_opt'):
        #     self.cri_regress = build_loss(train_opt['regress_opt']).to(self.device)
        # else:
        #     self.cri_pix = None
        self.cri_perceptual = None
        self.cri_gan = None
       




        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_U.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        #  # optimizer d
        # optim_type = train_opt['optim_d'].pop('type')
        # self.optimizer_d = self.get_optimizer(optim_type, self.net_du.parameters(), **train_opt['optim_d'])
        # self.optimizers.append(self.optimizer_d)


    @torch.no_grad()
    def feed_data(self, data):
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)



    def optimize_parameters(self, current_iter, g_scaler):
        # train_opt = self.opt['datasets']
        # for p in self.net_du.parameters():
        #     p.requires_grad = False

        # freeze_by_names(self.net_U,('blocks'))
        # freeze_by_names(self.net_U,('upsample'))
        # freeze_by_names(self.net_U,('blocks.5.'))
        # freeze_by_names(self.net_U,('blocks.6.'))
        # freeze_by_names(self.net_U,( 'blocks.7.'))
        num_block = 8
        self.outfakesrlist = []
        self.accumulation_steps = 1
        self.Predict = self.net_P(self.lq)
        # Predict = torch.max(F.softmax(Predict),1)[1] + 1
        # Predict =int(Predict)
        self.outfakesrlist = self.net_U(self.lq,self.Predict)
        loss_dict = OrderedDict()
        #进行每个block的比较
        PSNR_block = []
       
        for i in range(num_block):
            PSNR_block.append(self.caluate_psnr(self.outfakesrlist[i],self.gt))

        #返回[batch,1]个预测标签
        self.Predict_block = self.judgement(PSNR_block,num_block)
        self.Predict_block = self.Predict_block.to(self.device)     

        # print('Predict_block',self.Predict_block)

        # print('self.outfakesrlist[0] size',self.outfakesrlist[0].size())
        # print('self.outfakesrlist[1] size',self.outfakesrlist[1].size())
        # print('self.outfakesrlist[2] size',self.outfakesrlist[2].size())


        with torch.cuda.amp.autocast():
            l_g_total = 0
            if self.cri_pix:
                l_g0_pix = self.cri_pix(self.outfakesrlist[0], self.gt)
                l_g1_pix = self.cri_pix(self.outfakesrlist[1], self.gt)
                l_g2_pix = self.cri_pix(self.outfakesrlist[2], self.gt)
                l_g3_pix = self.cri_pix(self.outfakesrlist[3], self.gt)
                l_g4_pix = self.cri_pix(self.outfakesrlist[4], self.gt)
                l_g5_pix = self.cri_pix(self.outfakesrlist[5], self.gt)
                l_g6_pix = self.cri_pix(self.outfakesrlist[6], self.gt)
                l_g7_pix = self.cri_pix(self.outfakesrlist[7], self.gt)
    
                l_g_total += l_g0_pix + l_g1_pix +  l_g2_pix +  l_g3_pix +  l_g4_pix +  l_g5_pix  +  l_g6_pix +  l_g7_pix
                loss_dict['l_g0_pix'] = l_g0_pix
                loss_dict['l_g1_pix'] = l_g1_pix
                loss_dict['l_g2_pix'] = l_g2_pix
                loss_dict['l_g3_pix'] = l_g3_pix
                loss_dict['l_g4_pix'] = l_g4_pix
                loss_dict['l_g5_pix'] = l_g5_pix
                loss_dict['l_g6_pix'] = l_g6_pix
                loss_dict['l_g7_pix'] = l_g7_pix
       
                
            if self.cri_regress:
                l_regression = self.cri_regress(self.Predict, self.Predict_block.long())
                l_g_total += l_regression
                loss_dict['l_regression'] = l_regression
            loss_dict['l_g_total'] = l_g_total
        g_scaler.scale(l_g_total).backward()
        if((current_iter+1)%self.accumulation_steps)==0:
            g_scaler.step(self.optimizer_g)
            g_scaler.update()
            self.optimizer_g.zero_grad(set_to_none=True)


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:

                self.model_emaU(decay=self.ema_decay)
                self.model_emaP(decay=self.ema_decay)
    # 测试流程
    def test(self):
        if hasattr(self, 'net_g1_ema'):

            self.net_U_ema.eval()
            self.net_P_ema.eval()

            with torch.no_grad():

                Predict = self.net_P_ema(self.lq)
                print("Predict_ema is", Predict)
                self.outfakesrlist = self.net_U_ema(self.lq,Predict)
                self.outfakesr =  self.outfakesrlist[0]
        else:
            self.net_U.eval()
            self.net_P.eval()
            
            # names =[item[0] for item in self.net_U._modules.items()]
            # print(names)
            with torch.no_grad():
         

            #    self.hight = int(self.lq.size()[2] /32)
            #    self.width = int(self.lq.size()[3] /32)
            #    self.area += self.hight * self.width
            #    print('total area is ',self.area)
# --------------------------------------------------------------------#
               torch.cuda.synchronize()
               tn = time.time()
# --------------------------------------------------------------------#
               Predict = self.net_P(self.lq)

           
            #    print("Predict is", Predict)
                
               Predict = torch.max(Predict,1)[1]+1
            
            #    print("Predict is", Predict)
               Predict = 8
              
               self.outfakesrlist = self.net_U(self.lq,Predict)
# --------------------------------------------------------------------#
               torch.cuda.synchronize()
               dn1 = time.time() - tn
# --------------------------------------------------------------------#
               self.outfakesr =  self.outfakesrlist[0]
            # --------------------------------------------------------------------#
            #    print('dn1 difference is ',dn1)
    
# # --------------------------------------------------------------------#
            #    self.time += dn1
            #    print('total  time is ',self.time)
# --------------------------------------------------------------------#
            # --------------------------------------------------------------------#
            # flops1,params1 = profile(self.net_P,(self.lq,))
            # flops2,params2 = profile(self.net_U,(self.lq,Predict,))
            # # print("flops1 is",flops1)
            # # print("params1 is",params1)
            # # print("flops2 is",flops2)
            # # print("params2 is",params2)
            # # flops = flops1 + flops2
            # # params = params1 + params2
            # # # print("total flops is",flops)
            # # print("total params is",params)
            
            # self.flops += flops2 + flops1
            # print('total flops is ',self.flops)
            # --------------------------------------------------------------------#


            # flops1,params1 = get_model_complexity_info(self.net_P,(3,self.lq.size()[2],self.lq.size()[3]))
            # flops2,params2 = get_model_complexity_info(self.net_U,(3,self.lq.size()[2],self.lq.size()[3],Predict))
            # flops = flops1 + flops2
            # params = params1 + params2
            # # 计算单次
            # # print("flops is",flops)
            # # print("params is",params)


            # self.flops += flops2 + flops1
            # print('total flops is ',self.flops)

            self.net_U.train()
            self.net_P.train()



    # 多卡做validation
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    # 单卡做validation
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')


        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            # print(img_name)
            self.test()

            visuals = self.get_current_visuals()

            #-----------------------------------------------------------------------------------------#
            # print([visuals['result']].shape,[visuals['gt']])
            # mse = F.mse_loss(visuals['result'],visuals['gt'], reduction='mean')
            #-----------------------------------------------------------------------------------------#
            sr_img = tensor2img([visuals['result']])

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            # del self.Predict_block
            # del self.Predict
            del self.lq
            del self.outfakesrlist
            del self.outfakesr
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img,save_img_path)


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.is_train = True

    # 控制如何打印validation的结果
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    # 得到网络输出的结果
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.outfakesr.detach().cpu()

        return out_dict

    def save(self, epoch, current_iter):

        if hasattr(self, 'net_U_ema'):
            self.save_network([self.net_U, self.net_U_ema], 'net_U', current_iter, param_key=['params', 'params_ema'])
            self.save_network([self.net_P, self.net_P_ema], 'net_P', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_U, 'net_U', current_iter)
            self.save_network(self.net_P, 'net_P', current_iter)

        # self.save_network(self.net_du, 'net_du', current_iter)
        self.save_training_state(epoch, current_iter)

    def visdom (self, dataloader):
        self.is_train = False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            mse = F.mse_loss(visuals['result'],visuals['gt'], reduction='mean')
        self.is_train = True
        return mse

    def caluate_psnr(self,input1,input2):
        sr = input1
        gt = input2
        mse = F.mse_loss(sr,gt,reduction='none')
        mse = mse.reshape(self.batch,-1)
        mse = torch.mean(mse,1)
        psnr = 20 * torch.log10(1. / mse.sqrt())
        # print('sr.size',sr.size())
        # print('gt.size',gt.size())
        # print('mse.size',mse.size())
        # print('psnr.size',psnr.size())
        return psnr

    def judgement(self,PSNR,block):
        Predict_block  = torch.zeros(self.batch)
        #注意修改self.batch
        for i in range(self.batch):
            ema = []
            v_ema = []
            v_ema_corr =[]
            sum = 0
            count = 1

            beta = 0.9
            v_pre = 0
            for b in range(1,block):
                
                val = PSNR[b][i] - PSNR[b-1][i]
                ema.append(val)
                v_t = beta * v_pre + (1-beta) *val
                v_ema.append(v_t)
                v_pre = v_t
            
            for i, t in enumerate(v_ema):
                v_ema_corr.append(t/(1-np.power(beta, i+1)))
            
            
            result = 7
            if(PSNR[7][i] - PSNR[0][i] > 1.0):
                # print('PSNR large')
                result = 7
            # elif(PSNR[5][i] - PSNR[0][i] < 0.1):
            #     result = 0
            # elif(PSNR[5][i] - PSNR[1][i] < 0.1):
            #     result = 1
            # val = PSNR[b][i] - PSNR[b-1][i]
            #     sum += val
            #     ema_val = sum / count
            #     count += 1
            else : 
                for c in range(2,block-1):
                    # print(PSNR[c][i].item() - PSNR[c-1][i].item())
                    # print(v_ema_corr[c-2].item())
                    # print(PSNR[c][i].item() - PSNR[c-1][i].item() < v_ema_corr[c-2].item())
                    # print(PSNR[c+1][i].item() - PSNR[c][i].item() < v_ema_corr[c-1].item())
                    if(ema[c-1].item() < v_ema_corr[c-2].item()) :
                        if(ema[c].item() < v_ema_corr[c-1].item()):
                            result = c
                            break
                        else:
                            continue
                    
            
            Predict_block[i] = result
            # print('ema is',ema)
            # print('v_ema_error is',v_ema_corr)
            # print('predict is',result)
        return Predict_block

    # 固定网络参数
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]

    for name, child in model.named_children():
        if name not in layer_names:
            # print(name,'no name')
            continue
        for param in child.parameters():
            # print(name,'yes')
            param.requires_grad = not freeze
        
def freeze_by_names(model, layer_names):  
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)