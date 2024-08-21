
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from Vessel_Segmentation.train import CMD_Net_Train
from Vessel_Segmentation.test import CMD_Net_VS_Test
from Gland_Segmentation.main import train as CMD_Net_GS_Train
from Gland_Segmentation.main import CMD_Net_GS_Test
from networks.UNet import UNet
from networks.FCN import FCN
from networks.FCNsa import FCNsa
from networks.DeepLab import DeepLabV3
from networks.SegNet import SegNet
from networks.ConvUNeXt import ConvUNeXt
from networks.CMDNet_FCN import CMDNet_FCN
from networks.CMDNet_FCNsa import CMDNet_FCNsa
from networks.CMDNet_UNet import CMDNet_UNet
from networks.CMDNet_SegNet import CMDNet_SegNet
from networks.CMDNet_ConvUNeXt import CMDNet_ConvUNeXt
from networks.CMDNet_DeepLab import CMDNet_DeepLab


if __name__ == '__main__':

    parser = ArgumentParser(description="Evaluation script for CMD-Net models",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phase', default="train", type=str, help='input phase')
    parser.add_argument('--input-size', default=192, type=int, help='Images input size')
    parser.add_argument('--dataset', default="DRIVE", type=str, help='Images input size')
    parser.add_argument('--image-path', default='', type=str, help='DIRVE dataset path')
    parser.add_argument('--model', default='CMDNet_FCNsa', type=str, help='train model')
    parser.add_argument('--save-path', default='outcome/DRIVE', type=str, help='store the trained model')
    parser.add_argument('--alpha', default=0.0001, type=float, help='the empirical coefficient for diversity')
    parser.add_argument('--epoch', default=300, type=int, help='')
    parser.add_argument('--batchsize', default=4, type=int, help='Batch per GPU')
    parser.add_argument('--n-channels', default=3, type=int, help='n channels')
    parser.add_argument('--n-classes', default=1, type=int, help='classes')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--early-stop', default=20, type=int, help=' ')
    parser.add_argument('--update-lr', default=10, type=int, help=' ')
    parser.add_argument('--epochloss_init', default=10000, type=int, help=' ')
    parser.add_argument('--model-path', default='outcome/DRIVE/CMD-Net_DRIVE.th', type=str, help='store the trained model')
    parser.add_argument('--test-path', default='', type=str, help='store the test image')
    parser.add_argument('--gt-path', default='./dataset/DRIVE/test/1st_manual', type=str, help='store the test label')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

    image_path = args.image_path
    if image_path == '' and args.phase == 'train':
        if args.dataset == 'DRIVE':
            image_path = "./dataset/DRIVE"
        elif args.dataset == 'CRAG':
            image_path = "./dataset/CRAG"
        elif args.dataset == 'CHASEDB':
            image_path = "./dataset/CHASEDB1"
        elif args.dataset == 'GlaS':
            image_path = "./dataset/GlaS"
        else:
            print("please assign the dataset.")
            sys.exit()

    test_path = args.test_path
    gt_path = args.gt_path 
    if test_path == '' and args.phase == 'test':
        if args.dataset == 'DRIVE':
            test_path = "./dataset/DRIVE/test/images/"
            gt_path = "./dataset/DRIVE/test/1st_manual"
        elif args.dataset == 'CHASEDB':
            test_path = "./dataset/CHASEDB1/test/images/"
            gt_path = "./dataset/CHASEDB1/test/labels1"
        elif args.dataset == 'CRAG':
            test_path = "./dataset/CRAG/"
        elif args.dataset == 'GlaS':
            test_path = "./dataset/GlaS/" 
        else:
            print("please assign the dataset.")
            sys.exit()

        
    if (args.model == 'FCN'):
        net = FCN(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'UNet'):
        net = UNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'FCNsa'):
        net = FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'DeepLab'):
        net = DeepLabV3(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'SegNet'):
        net = SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'ConvUNeXt'):
        net = ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_FCN'):
        net = CMDNet_FCN(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_FCNsa'):
        net = CMDNet_FCNsa(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_UNet'):
        net = CMDNet_UNet(n_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_SegNet'):
        net = CMDNet_SegNet(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_DeepLab'):
        net = CMDNet_DeepLab(in_channels = args.n_channels, n_classes = args.n_classes).cuda()
    elif (args.model == 'CMDNet_ConvUNeXt'):
        net = CMDNet_ConvUNeXt(in_channels = args.n_channels, num_classes = args.n_classes).cuda()
    else:
        print("the model name must be in CMDNet_FCN, CMDNet_FCNsa, CMDNet_UNet, CMDNet_ConvUNeXt.")
        sys.exit()


    if "train" in args.phase:
        if  "DRIVE" in image_path or "CHASE" in image_path:
            CMD_Net_Train(net, [], [],  args.input_size,  image_path,  args.save_path, args.alpha, args.epoch, args.batchsize, args.lr,  args.early_stop, args.update_lr, args.epochloss_init )
        else:
            CMD_Net_GS_Train(net, args.batchsize, args.n_channels, args.input_size, args.lr, args.epoch, image_path, args.save_path)
    elif "test" in args.phase:
        if  "DRIVE" in test_path or "CHASE" in test_path:
            CMD_Net_VS_Test(net, args.model_path, test_path, gt_path, args.gpuid)
        else:
            CMD_Net_GS_Test(net, args.model_path, test_path, args.save_path, args.input_size, args.gpuid)
    else:
        print("input phase error.")
