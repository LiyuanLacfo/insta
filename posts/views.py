from django.shortcuts import render
import cv2
import os
import torch
import os
from .networks import define_G
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np

# Create your views here.
# posts/views.py
from django.views.generic import ListView, CreateView
from django.urls import reverse_lazy # new
from .forms import PostForm # new
from .models import Post
from django.conf import settings


class HomePageView(ListView):
    model = Post
    template_name = 'home.html'


def pix2pix_index(request):
    images1 = [Post.objects.latest('uploaded_on')]
    images2 = [Post.objects.get(title="white image")]
    context = {'images1': images1,
               'images2': images2,
              }
    return render(request, 'home.html', context)

def output(request):
    image = Post.objects.latest('uploaded_on')
    img_name = image.cover.name
    img_path = os.path.join(settings.MEDIA_ROOT, img_name)
    new_image_prefix = img_name.split('.')[0]
    new_img_name = new_image_prefix+"_denoised"+".jpg"
    new_image = denoise(img_path, new_img_name)
    images1 = [image]
    images2 = [new_image]
    print(len(Post.objects.all()))
    return render(request, 'output.html', {'images1': images1, 'images2': images2})

def process_image(image, new_image_name):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img_path = os.path.join(settings.MEDIA_ROOT, new_image_name)
    cv2.imwrite(gray_img_path, gray_img)
    new_image = Post()
    new_image.title = "new gray image"
    new_image.cover = new_image_name
    return new_image

def denoise(image_path, new_image_name):
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = "unet_256"
    norm = "batch"
    no_dropout = False
    init_type = "normal"
    init_gan = 0.02
    model = define_G(input_nc, output_nc, ngf, netG, norm, not no_dropout, init_type, init_gan)
    model_name = "latest_net_G.pth"
    model_path = os.path.join(settings.MEDIA_ROOT, model_name)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    my_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    my_dataset = MyDataset(image_path, my_transforms)
    new_img_path = os.path.join(settings.MEDIA_ROOT, new_image_name)
    for i in range(len(my_dataset)):
        sample = my_dataset[i]
        a = model.forward(sample.unsqueeze(0))
        result_img = tensor2im(a)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(new_img_path, result_img)
    new_image = Post()
    new_image.title = "denoised image"
    new_image.cover = new_image_name
    return new_image
        

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

class MyDataset(Dataset):
    def __init__(self, img_path, transform = None):
        self.transform = transform
        self.img_path = img_path
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        image = Image.open(self.img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class CreatePostView(CreateView): # new
    model = Post
    form_class = PostForm
    template_name = 'post.html'
    success_url = reverse_lazy('pix2pix')
