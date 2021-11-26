# Import libraries
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy import mean

from process_user_input import Process, TransformAll


class Predict:
    def __init__(self, model_path, img_pth):
        self.path = model_path
        self.im = img_pth

    def predict(self, original_image, kernel_size, min_thresh, n, dim):
        # Load Model
        model = torch.load(self.path)

        # Load and transform Data
        pro = Process(self.im, original_image, kernel_size, min_thresh, n, dim)
        tra = TransformAll(self.im, original_image, kernel_size, min_thresh, n, dim)
        images = tra.split_numbers()

        tensor_images = tra.distribute_collect_images()
        img = pro.resize(True)

        probs = []
        preds = []
        raw_im = []
        for im, rim in tensor_images:
            # Predict
            model.eval()
            im = im.view(im.shape[0], -1)
            output = model(im)

            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)

            pred = top_class[0][0].item()
            prob = torch.softmax(output, dim=1)[0][pred].item()

            probs.append(prob)
            preds.append(pred)
            raw_im.append(rim)

        return int(mean(probs) * 100), ' '.join([str(i) for i in preds]), images, raw_im
