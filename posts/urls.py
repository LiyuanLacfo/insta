# posts/urls.py
from django.urls import path

from .views import HomePageView, CreatePostView, pix2pix_index, output

urlpatterns = [
    path('', pix2pix_index, name='pix2pix'),
    path('output/', output, name='script'),
    # path('output/', output, name='script'),
    # path('', HomePageView.as_view(), name='pix2pix'), #here name must be the same as route in project urls.py
    path('post/', CreatePostView.as_view(), name='add_post') # new
]