import os
import glob
from xy_scripts.utils import Visualizer

def demo_image(root_dir):
    result_root = os.path.join(root_dir, '*')
    root = '/'.join(result_root.split('/')[:-4])
    epoch_root = os.path.join(root, 'test_outputs')
    visual_output = os.path.join(root, 'visual')
    exp_name = root_dir.split('/')[-1]
    result_names = [x.split('/')[-1] for x in sorted(glob.glob(result_root))]
    epoch_names = [x.split('/')[-1] for x in sorted(glob.glob(os.path.join(epoch_root, '*')))]

    for result_name in result_names:
        visualizer = Visualizer(os.path.join(visual_output), size=(180,320), demo_name=result_name + '.html', col=10)
        for epoch_name in epoch_names:
            cur_root = os.path.join(epoch_root, epoch_name, exp_name, result_name)
            images = sorted(glob.glob(os.path.join(cur_root, '*.png')))
            epoch = epoch_name.split('_')[1].zfill(3)
            for i in range(0, len(images)):
                visual_pth = '/' + '/'.join(images[i].split('/')[-7:])
                visual_name = epoch + '_' + result_name + '_' + visual_pth.split('/')[-1]
                visualizer.insert(visual_pth, visual_name)
        visualizer.write(_sorted=True)