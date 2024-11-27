
import torchvision


test_dataset = torchvision.datasets.VOCSegmentation(
    root= "/work/dlclarge2/dasb-Camvid", 
    year = '2007', 
    image_set= 'train', 
    download= False, 
)

print(test_dataset[0])