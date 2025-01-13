from torchvision import transforms

#  Preprocessing pipeline (replace with actual preprocessing) 
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return preprocess(image)