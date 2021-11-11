from UGATIT import UGATIT
import argparse
from utils import *

dist_framework = os.getenv("DISTRIBUTED_FRAMEWORK", "").lower()
if dist_framework == "byteps":
    import byteps.torch as bps
elif dist_framework == "horovod":
    import horovod.torch as bps
else:
    import torch.distributed as bps
    def local_rank():
        return int(os.getenv("LOCAL_RANK", "-1"))
    def local_size():
        return int(os.getenv("LOCAL_WORLD_SIZE", "-1"))
    def rank():
        return bps.get_rank()
    def size():
        return bps.get_world_size()
    def init():
        bps.init_process_group(backend="nccl")
        return None

    bps.local_rank = local_rank
    bps.local_size = local_size
    bps.rank = rank
    bps.size = size
    bps.init = init


"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='dataset dir path')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--fix_aug', type=str2bool, default=False, help='No resize and random crop when train')
    parser.add_argument('--list_mode', type=str2bool, default=False, help='load image list')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    if bps.rank() == 0:
        check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
        check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
        check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    # try:
        # assert args.epoch >= 1
    # except:
        # print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    bps.init()
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    print(f'xxxx bps.local_rank() {bps.local_rank()}', flush=True)
    torch.cuda.set_device(bps.local_rank())
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
