import glob
import logging
import os
import random
import warnings
from collections import OrderedDict
from multiprocessing.managers import Namespace

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from skimage import morphology, measure
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import gaussian_filter
from torchvision import transforms

import common
from sklearn.metrics import roc_curve, roc_auc_score

import metrics
from utils import PatchMaker, rand_perlin_2d_np, read_and_convert_image, rand_augment_abnormal, save_pytorch_model

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# def init_weight(m):
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.xavier_normal_(m.weight)
#     elif isinstance(m, torch.nn.Conv2d):
#         torch.nn.init.xavier_normal_(m.weight)
def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2),
                                     torch.nn.Dropout(p=0.5),
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x

class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1

class PlainNet(torch.nn.Module):
    def __init__(self,
                 extractor,
                 device,
                 train_dataloader,
                 test_dataloader,
                 type_,
                 dataset_name,
                 args : Namespace
                 ):
        super(PlainNet, self).__init__()

        self.device = device
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self.type_ = type_
        self.dataset_name = dataset_name
        self.args = args

        if self.dataset_name == 'MVTec-AD':
            self._weight = self.args.weight[0]
        elif self.dataset_name == 'MPDD':
            self._weight = self.args.weight[1]
        elif self.dataset_name == 'VisA':
            self._weight = self.args.weight[2]
        else:
            self._weight = self.args.weight[3]

        self._layers_extract = extractor['layers_extract']
        self._backbone = extractor['backbone']
        self._no_grad_modules = torch.nn.ModuleDict({})
        self._update_modules = torch.nn.ModuleDict({})
        self._patch_makers = PatchMaker(self.args.patchsize, self.args.patchstride)
        self._meta_epochs, self._gan_epochs = self.args.meta_epochs, self.args.gan_epochs
        self._aed_meta_epochs, self._epochs = self.args.aed_meta_epochs, self.args.epochs
        self._train_backbone = self.args.train_backbone
        self._intput_shape = self.args.intput_shape
        self._init_adapter_lr, self._init_discriminator_lr = self.args.adapter_lr, self.args.discriminator_lr
        self._opt_adapter, self._opt_discriminator = None, None
        self._opt_discriminator_sch = None
        self._margin = self.args.margin
        self._model_dir, self._ckpt_dir, self._tb_dir, self._logger = "", "", "", None
        self._pse_idx = None
        self._inv_normalize = transforms.Normalize(mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],std=[1 / s for s in [0.229, 0.224, 0.225]])
        self._process_image = transforms.Compose(  # 经过这个处理的都会变成B, C, H, W
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._adaptive_pool = nn.AdaptiveAvgPool1d(self.args.channel_num_reduced)
        self._sdas_file_list = sorted(glob.glob(os.path.join('./data', self.dataset_name, 'sdas', self.type_, '*.jpg')))
        self._dtd_file_list = sorted(glob.glob(os.path.join('./data/DTD', 'images', '*', '*.jpg')))
        self._anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=self.args.intput_shape[-2:])

        self._config_model()
        self._pse_image, self._pse_gt = self._get_pse_image()

    def _train_module(self):
        _ = self._no_grad_modules.eval()
        self._update_modules['adapter'].train()
        self._update_modules['discriminator'].train()

        for i_epochs in range(self._epochs): # 10
            train_loss = 0.
            all_sample = 0
            with tqdm.tqdm(self._train_dataloader, desc=f"Training {i_epochs + 1} / {self._epochs}...", leave=False) as data_iterator:
                for i_batch, (image_normal, gt_normal) in enumerate(data_iterator):
                    self._opt_adapter.zero_grad()
                    self._opt_discriminator.zero_grad()
                    image_normal = image_normal.to(self.device)
                    #gt_normal = gt_normal.to(self.device)
                    #is_normal = True
                    batch_size = image_normal.size(0)
                    all_sample += batch_size

                    features_normal, patch_shape_normal = self._embed(image_normal) # [B x 28 x 28, 1536]
                    features_pse, patch_shapes_pse = self._embed(self._pse_image[i_batch])  # [B x 28 x 28, 1536]

                    features_pse_reduced = self._adaptive_pool(features_pse.unsqueeze(1)).squeeze(1) #这里后面换成idx
                    features_normal_reduced = self._adaptive_pool(features_normal.unsqueeze(1)).squeeze(1)  # 这里后面换成idx

                    features_normal_adapted = self._update_modules['adapter'](features_normal_reduced) # [B x 28 x 28, 1024]
                    features_pse_adapted = self._update_modules['adapter'](features_pse_reduced)  # [B x 28 x 28, 1024]

                    # pse_mask
                    pse_mask = self._pse_gt[i_batch]
                    pse_mask_ = pse_mask.flatten()
                    down_ratio_y = int(pse_mask.shape[2] / patch_shapes_pse[0][0])
                    down_ratio_x = int(pse_mask.shape[3] / patch_shapes_pse[0][1])
                    anomaly_mask = torch.nn.functional.max_pool2d(pse_mask, (down_ratio_y, down_ratio_x)).float()
                    anomaly_mask = anomaly_mask.reshape(-1, 1)

                    loss_gau = self._cal_loss_gaussian(features_normal_adapted)
                    loss_pse = self._cal_loss_pse(features_pse_adapted, pse_mask_, patch_shapes_pse, batch_size)
                    loss_sim = self._sim_loss(features_normal_adapted, features_pse_adapted, anomaly_mask)

                    loss = self._weight[0] * loss_gau + self._weight[1] * loss_pse + self._weight[2] * loss_sim
                    train_loss += loss
                    loss.backward()
                    self._opt_adapter.step()
                    self._opt_discriminator.step()

            self._opt_discriminator_sch.step()
            self._opt_adapter_sch.step()
            torch.cuda.empty_cache()

    def train_self(self):

        best_record = None
        for i_mepoch in range(self.args.meta_epochs): # 15
            self._train_module()
            is_anormal_list, predict_img_list_imgscore, ground_truth_pixel, predict_pixel = self._predict()
            metrics_img = metrics.compute_imagewise_retrieval_metrics(predict_img_list_imgscore, is_anormal_list, False)
            metrics_pix = metrics.compute_pixelwise_retrieval_metrics(predict_pixel, ground_truth_pixel, False)
            aupro = metrics.cal_pro_score(ground_truth_pixel, predict_pixel)
            if best_record is None:
                best_record = [metrics_img['auroc'], metrics_pix['auroc'], aupro]
            else:
                if metrics_img['auroc'] + metrics_pix['auroc'] + aupro > best_record[0] + best_record[1] + best_record[2]:
                    best_record = [metrics_img['auroc'], metrics_pix['auroc'], aupro]
                    save_pytorch_model(self._update_modules['adapter'], self._update_modules['discriminator'],
                                       self.type_, metrics_img['auroc'], metrics_pix['auroc'], aupro, i_mepoch, data_set_name=self.dataset_name)
            print(f"----- {i_mepoch} I-AUROC:{round(metrics_img['auroc'] * 100, 3)}(MAX:{round(best_record[0] * 100, 3)})"
                  f"  P-AUROC{round(metrics_pix['auroc'] * 100, 3)}(MAX:{round(best_record[1] * 100, 3)})"
                  f"  AUPRO{round(aupro * 100, 3)}(MAX:{round(best_record[2] * 100, 3)})-----")

    def _predict(self, path=None):
        _ = self._no_grad_modules.eval()
        self._update_modules['discriminator'].eval()
        self._update_modules['adapter'].eval()
        predict_img_list_imgscore = []
        ground_truth_pixel = []
        predict_pixel = []
        is_normal_list = []
        with torch.no_grad():
            with tqdm.tqdm(self._test_dataloader, desc="Inferring...", leave=False) as data_iterator:
                for i, sample in enumerate(data_iterator):
                    image = sample[0].to(self.device)
                    ground_truth = sample[1].to(self.device) # tensor[1, 1, 224, 224]
                    is_normal = sample[2].to(self.device) # tensor(True/False)
                    is_normal_list.append(is_normal.item())
                    features_image, patch_shapes = self._embed(image)
                    features_image_reduced = self._adaptive_pool(features_image.unsqueeze(1)).squeeze(1)  # 这里后面换成idx
                    features_image_reduced_adapted = self._update_modules['adapter'](features_image_reduced)
                    image_scores = -self._update_modules['discriminator'](features_image_reduced_adapted)
                    image_scores_map = self._cal_map(image_scores, patch_shapes) # numpy[224, 224]

                    ground_truth_pixel.append(ground_truth.squeeze(0).squeeze(0).cpu().numpy()) # 真实的像素级ground_truth图，numpy[224, 224]
                    predict_pixel.append(image_scores_map) # 预测的像素级ground_truth图，numpy[224, 224]

                    predict_img_list_imgscore.append(np.max(image_scores_map).item()) #列表[1, .....]

            is_normal_list = np.array([int(not a) for a in is_normal_list])
            ground_truth_pixel = np.array(ground_truth_pixel)
            predict_pixel = np.array(predict_pixel)
            predict_img_list_imgscore = np.array(predict_img_list_imgscore)

        predict_img_list_imgscore, predict_pixel = self._normalize(predict_img_list_imgscore, predict_pixel)
        return is_normal_list, predict_img_list_imgscore, ground_truth_pixel, predict_pixel

    @staticmethod
    def _normalize(pr_list_sp, pr_px):
        pr_list_sp = np.array(pr_list_sp)
        min_sp = pr_list_sp.min(axis=-1).reshape(-1, 1)
        max_sp = pr_list_sp.max(axis=-1).reshape(-1, 1)
        pr_list_sp = (pr_list_sp - min_sp) / (max_sp - min_sp)
        pr_list_sp = np.mean(pr_list_sp, axis=0)

        pr_px = np.array(pr_px)
        min_scores = pr_px.reshape(len(pr_px), -1).min(axis=-1).reshape(-1, 1, 1, 1)
        max_scores = pr_px.reshape(len(pr_px), -1).max(axis=-1).reshape(-1, 1, 1, 1)
        pr_px = (pr_px - min_scores) / (max_scores - min_scores)
        pr_px = np.mean(pr_px, axis=0)

        return pr_list_sp, pr_px

    def _cal_map(self, scores, patch_shape): #要求测试的batch_size必须为1
        scores = scores.reshape(1, patch_shape[0][0], patch_shape[0][1]).unsqueeze(1)
        scores = F.interpolate(scores, size=self._intput_shape[1], mode='bilinear', align_corners=False)
        anomaly_map = scores.reshape(self._intput_shape[1], self._intput_shape[2]).cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        return anomaly_map

    def _config_model(self):
        feature_aggregator = common.NetworkFeatureAggregator(self._backbone, self._layers_extract, self.device)
        '''
        {
            "layer2": tensor(batch_size, 512, 28, 28),
            "layer3": tensor(batch_size, 1024, 14, 14),
        }
        '''
        feature_dimensions = feature_aggregator.feature_dimensions(self._intput_shape)
        '''
        [512, 1024] 输出的是通道数列表
        '''
        preprocessing = common.Preprocessing(feature_dimensions, self.args.embed_dimension)
        '''
        feature_dimensions: 是一个维度列表[512, 1024],
        self.args.embed_dimension: 是要统一的维度数1536比如
        作用是将[512, 1024]全部统一成[1536, 1536]的维度，然后拼接
        输出是Tensor[batch_size, len(feature_dimensions), self.args.embed_dimension]
        '''
        preadapt_aggregator = common.Aggregator(target_dim=self.args.embed_dimension)
        '''
        输出是Tensor(batch_size, self.args.embed_dimension)
        是将Tensor[batch_size, len(feature_dimensions), self.args.embed_dimension]压缩到Tensor(batch_size, self.args.embed_dimension)
        '''

        feature_aggregator = feature_aggregator.to(self.device)
        preprocessing= preprocessing.to(self.device)
        preadapt_aggregator = preadapt_aggregator.to(self.device)

        self._no_grad_modules["feature_aggregator"] = feature_aggregator
        self._no_grad_modules["preprocessing"] = preprocessing
        self._no_grad_modules["preadapt_aggregator"] = preadapt_aggregator

        adapter = Projection(self.args.channel_num_reduced, self.args.channel_num_reduced,
                             self.args.adapter_layer_number, self.args.adapter_layer_type)
        discriminator = Discriminator(self.args.channel_num_reduced, self.args.discriminator_layer,
                             self.args.discriminator_hidden)
        adapter = adapter.to(self.device)
        discriminator = discriminator.to(self.device)
        self._update_modules["adapter"] = adapter
        self._update_modules["discriminator"] = discriminator

        self._opt_adapter = torch.optim.Adam(self._update_modules["adapter"].parameters(), self._init_adapter_lr)
        # self._opt_discriminator = torch.optim.Adam(self._update_modules["discriminator"].parameters(),
        #                                            lr=self._init_discriminator_lr,weight_decay=1e-5)
        self._opt_discriminator = torch.optim.SGD(self._update_modules["discriminator"].parameters(), self._init_discriminator_lr, momentum=0.9, weight_decay=1e-5)
        self._opt_discriminator_sch = torch.optim.lr_scheduler.CosineAnnealingLR(self._opt_discriminator,
                                                                                 (self._meta_epochs - self._aed_meta_epochs) * self._epochs,
                                                                                 self._init_discriminator_lr*.1)
        self._opt_adapter_sch = torch.optim.lr_scheduler.CosineAnnealingLR(self._opt_adapter,
                                                                           (self._meta_epochs - self._aed_meta_epochs) * self._epochs,
                                                                           self._init_adapter_lr * .01)

    def _embed(self, images):
        images = images.to(self.device)
        _ = self._no_grad_modules.eval()

        with torch.no_grad():
            features = self._no_grad_modules["feature_aggregator"](images)

        features_org = [features[layer] for layer in self._layers_extract]

        features = [
            self._patch_makers.patchify(x, return_spatial_info=True) for x in features_org
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        '''
        features = [
            tensor(324, 512, 3, 3),  # layer2
            tensor(64, 1024, 3, 3),   # layer3
        ]
        '''
        features = self._no_grad_modules["preprocessing"](features)
        features = self._no_grad_modules["preadapt_aggregator"](features)
        return features, patch_shapes

    def _get_pse_image(self):
        pse_dir = os.path.join('./data', self.dataset_name, 'pse_image', self.type_, 'pse')
        mask_dir = os.path.join('./data', self.dataset_name, 'pse_image', self.type_, 'mask')
        if not os.path.exists(pse_dir):
            os.makedirs(pse_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

        pse = []
        mask = []

        for image_normal, _ in self._train_dataloader:
            sub_pse = []
            sub_mask = []
            for one_image_normal_C_H_W in image_normal:
                one_image_normal_C_H_W = self._inv_normalize(one_image_normal_C_H_W)
                one_image_normal_H_W_C_numpy = one_image_normal_C_H_W.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                sub_pse_image, sub_pse_mask = self._generate_anomaly_image(one_image_normal_H_W_C_numpy)
                sub_pse_image = self._process_image(sub_pse_image)
                sub_pse_mask = torch.tensor(np.expand_dims(sub_pse_mask * 255.0, axis=2).astype(np.uint8)).permute(2, 0, 1)
                sub_pse_mask = sub_pse_mask / 255
                sub_pse.append(sub_pse_image.unsqueeze(0))
                sub_mask.append(sub_pse_mask.unsqueeze(0))

            sub_pse = torch.cat(sub_pse, dim=0)
            sub_mask = torch.cat(sub_mask, dim=0)
            pse.append(sub_pse)
            mask.append(sub_mask)

        return pse, mask

    def _cal_loss_gaussian(self, features_adapt):
        noise_idxs = torch.randint(0, self.args.mix_noise, torch.Size([features_adapt.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.args.mix_noise).to(self.device)  # (N, K)
        noise = torch.stack([
            torch.normal(0, self.args.noise_std * 1.1 ** k, features_adapt.shape)
            for k in range(self.args.mix_noise)], dim=1).to(self.device)  # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = features_adapt + noise
        scores_denoise = self._update_modules['discriminator'](torch.cat([features_adapt, fake_feats]))
        true_scores = scores_denoise[:len(features_adapt)]
        fake_scores = scores_denoise[len(fake_feats):]
        true_loss = torch.clip(-true_scores + self.args.margin, min=0)
        fake_loss = torch.clip(fake_scores + self.args.margin, min=0)
        loss_gaussian = true_loss.mean() + fake_loss.mean()
        return loss_gaussian

    def _cal_loss_pse(self, pse_features, pse_mask, pse_patch_shape, batch_size):
        scores_pse = self._update_modules['discriminator'](pse_features)
        scores_pse = scores_pse.reshape(batch_size, pse_patch_shape[0][0], pse_patch_shape[0][1]).unsqueeze(1)
        scores_pse = F.interpolate(scores_pse, size=self._intput_shape[1], mode='bilinear', align_corners=False)
        scores_pse_ = - scores_pse.reshape(batch_size, -1)
        scores_pse_ = torch.sigmoid(scores_pse_)
        scores_cls, _ = torch.max(scores_pse_, dim=1)
        loss_l2 = nn.MSELoss()
        loss_cls = loss_l2(scores_cls, torch.zeros_like(scores_cls) + 1.0)
        scores_pse = scores_pse.flatten()
        if torch.sum(pse_mask) == 0:
            normal_loss = torch.clip(self.args.margin - scores_pse, min=0)
            loss_pixel = normal_loss.mean()
        else:
            anomaly_scores = scores_pse[pse_mask == 1]
            normal_scores = scores_pse[pse_mask == 0]
            anomaly_loss = torch.clip(self.args.margin + anomaly_scores, min=0)
            normal_loss = torch.clip(self.args.margin - normal_scores, min=0)
            loss_pixel = anomaly_loss.mean() + normal_loss.mean()
        return 0.5 * loss_pixel + 0.5 * loss_cls

    def _sim_loss(self, features_adapt, features_pse_adapt, mask):
        mask = mask.to(self.device)
        if torch.sum(mask) == 0:
            loss_sim = 0
        else:
            def reshape(features, mask):
                features = mask * features
                nozero = torch.any(features != 0, dim=1)
                features = features[nozero]
                return features

            features_adapt = reshape(features_adapt, mask)
            features_pse_adapt = reshape(features_pse_adapt, mask)
            sim = torch.nn.functional.cosine_similarity(features_pse_adapt, features_adapt)
            loss_sim = sim.mean()
        return loss_sim

    def _generate_anomaly_image(self, img):
        """
        step 1. generate mask
            - target foreground mask
            - perlin noise mask

        step 2. generate texture or structure anomaly
            - texture: load DTD
            - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation,
            and hue on the input image  퐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid
            and randomly arranged to obtain the disordered image  퐼

        step 3. blending image and anomaly source
        """

        target_foreground_mask = self._generate_target_foreground_mask(img)
        # Image.fromarray(target_foreground_mask*255).convert('L').save("foreground.jpg")

        ## perlin noise mask
        perlin_noise_mask = self._generate_perlin_noise_mask()

        ## mask
        mask = perlin_noise_mask * target_foreground_mask  # 0 and 1

        # step 2. generate texture or structure anomaly
        anomaly_img = self._anomaly_source(img=img,
                                           mask=mask,
                                           anomaly_type=self.args.anomaly_type).astype(np.uint8)

        return anomaly_img, mask

    def _generate_target_foreground_mask(self, image: np.ndarray):
        if self.dataset_name == 'MVTec-AD':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.type_ in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
                return np.ones_like(img_gray)
            if self.type_ == 'pill':
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int64)
            elif self.type_ in ['hazelnut', 'metal_nut', 'toothbrush']:
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int64)
            elif self.type_ in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
                _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_background_mask = target_background_mask.astype(np.bool_).astype(np.int64)
                target_foreground_mask = 1 - target_background_mask
            else:
                raise NotImplementedError("Unsupported foreground segmentation category")
            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(6))
            target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(6))
            return target_foreground_mask

        elif self.dataset_name == 'MPDD':
            if self.type_ in ['bracket_black', 'bracket_brown', 'connector']:
                img_seg = image[:, :, 1]
            elif self.type_ in ['bracket_white', 'tubes']:
                img_seg = image[:, :, 2]
            else:
                img_seg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(np.bool_).astype(np.int32)

            if self.type_ in ['bracket_white', 'tubes']:
                target_foreground_mask = target_background_mask
            else:
                target_foreground_mask = 1 - target_background_mask

            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(6))
            return target_foreground_mask

        elif self.dataset_name == 'VisA':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.type_ in ['capsules']:
                return np.ones_like(img_gray)
            if self.type_ in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
                _, target_foreground_mask = cv2.threshold(image[:, :, 2], 100, 255,
                                                          cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int32)
                target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
                target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
                return target_foreground_mask
            else:
                _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                target_foreground_mask = target_foreground_mask.astype(np.bool_).astype(np.int32)
                target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
                target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
                return target_foreground_mask

        elif self.dataset_name == 'BTAD':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.type_ in ['02']:
                return np.ones_like(img_gray)
            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_foreground_mask = target_background_mask.astype(np.bool_).astype(np.int32)
            target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(15))
            target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(6))
            return target_foreground_mask

    def _generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.args.imagesize, self.args.imagesize), (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)

        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.args.perlin_noise_threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise)
        )
        return mask_noise

    def _sdas_source(self) -> np.ndarray:
        path = random.choice(self._sdas_file_list)
        sdas_source_img = read_and_convert_image(path, self.args.imagesize)
        return sdas_source_img.astype(np.float32)

    def _dtd_source(self) -> np.ndarray:
        idx = np.random.choice(len(self._dtd_file_list))
        dtd_source_img = read_and_convert_image(self.dtd_file_list[idx], self.args.resize)
        dtd_source_img = rand_augment_abnormal()(image=dtd_source_img)
        return dtd_source_img.astype(np.float32)  # 返回经过随机增强的异常源图, 224

    def _anomaly_source(self, img: np.ndarray,
                        mask: np.ndarray,
                        anomaly_type: str):

        if anomaly_type == 'sdas':
            anomaly_source = self._sdas_source()
            factor = np.random.uniform(*self.args.sdas_transparency_range, size=1)[0]

        elif anomaly_type == 'dtd':
            anomaly_source = self._dtd_source()
            factor = np.random.uniform(*self.args.dtd_transparency_range, size=1)[0]
        else:
            raise NotImplementedError("unknown ano")

        mask_expanded = np.expand_dims(mask, axis=2)
        anomaly_img = factor * (mask_expanded * anomaly_source) + (1 - factor) * (mask_expanded * img)
        anomaly_img = ((- mask_expanded + 1) * img) + anomaly_img
        return anomaly_img  # 返回标准伪异常图像

    @staticmethod
    def calculate_image_metrics(y_true, y_score):
        """
        计算图像的评价指标，包括 AUROC、TPR、FPR、阈值等。

        参数:
        y_true (array-like): 真实的图像标签，形状为 (n_samples,)
        y_score (array-like): 模型预测的概率，形状为 (n_samples,)

        返回:
        dict: 包含评价指标的字典
        """
        # 计算 AUROC
        auroc = roc_auc_score(y_true, y_score)

        # 计算 FPR、TPR 和阈值
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        metrics = {
            'AUROC': auroc,
            'FPR': fpr,
            'TPR': tpr,
            'thresholds': thresholds
        }

        return metrics