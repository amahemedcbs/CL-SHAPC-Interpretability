### Load compare dict
import shap
import numpy as np
import addcopyfighandler

# Custom Imports
from saliency_generator import SalGenArgs, iTAMLArgs, MnistArgs
import saliency_dataloader as sdl



algorithm = "iTAML"
dataset = "cifar10"
num = 10 if dataset == "cifar100" else 5


if algorithm == "iTAML":
   SalGenArgs.args = iTAMLArgs
else: SalGenArgs.args = MnistArgs

if dataset == "cifar100":
    SalGenArgs.class_per_task = 10
    SalGenArgs.num_class = 100
else:
    SalGenArgs.class_per_task = 2
    SalGenArgs.num_class = 10

SalGenArgs.algorithm = algorithm
SalGenArgs.dataset = dataset
SalGenArgs.args.dataset = SalGenArgs.dataset
SalGenArgs.args.num_class = SalGenArgs.num_class


shap_values_loaded = np.load(f"analysis/noshuffle/{algorithm}/shap_values_first_last_1000.npy", allow_pickle=True)  # ['shap_dict']
#shap_values_loaded = np.load(f"analysis/{algorithm}/{dataset}/shap_values_first_last_1000.npy", allow_pickle=True)  # ['shap_dict']
num_imgs = len(shap_values_loaded[()].keys())
shap_dict = {}
for i in range(num_imgs):
    shap_dict[f'{i}'] = shap_values_loaded[()][f'{i}']

'''
Bad Sample: 80
Good Sample: 132

Samples to Try: 55
'''

sample = 303
test_sample = shap_dict[f'{sample}']
test_sess = list(test_sample.keys())
test_sess.remove(test_sess[1])
print(test_sess)
ses = int(test_sess[0][-1])

'''
if dataset == "mnist":
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                  shap_dict[f'{sample}']['ses4']['shap_values'].reshape(28,28,1)]
else:
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze().reshape(32,32,3),
                  shap_dict[f'{sample}']['ses4']['shap_values'].squeeze().reshape(32,32,3)]
    print(test_shaps[0].shape)
'''
if dataset == "mnist":
    test_shaps = [shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].reshape(28,28,1),
                  shap_dict[f'{sample}']['ses4']['shap_values'].reshape(28,28,1)]
else:
    test_shaps =[shap_dict[f'{sample}'][f'ses{ses}']['shap_values'].squeeze(-1).transpose([0,2,3,1]),
                  shap_dict[f'{sample}']['ses4']['shap_values'].squeeze(-1).transpose([0,2,3,1])]
# Get test dataset
sal_dataloader = sdl.SalDataloader(SalGenArgs)
if dataset == "cifar100":
    test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data(range(ses * 10, (ses * 10) + 10), 100, batch_size=10000)
else:
    test_imgs, test_labels, _, STD, MEAN = sal_dataloader.load_data([ses*2, (ses*2)+1], 100, batch_size=10000)
print("Len of sal_imgs:", len(test_imgs))

# Get test image
test_img = test_imgs[sample-(ses*200)]
if dataset != "mnist":
    test_img = sal_dataloader.denormalize(test_img)
test_img_np = np.transpose(test_img.numpy(), [1, 2, 0])

shap.image_plot(np.concatenate(test_shaps), np.stack([test_img_np,test_img_np]), true_labels=test_sess)


# For a model with high SHAPC, but low accuracy:
# Look for a sample that was predicted correctly, feature consistency should be high
# Look for a sample that was predicted incorrectly, but check if feature consistency is still high -> indicates trustworthiness

