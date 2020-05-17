import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
from imutils import build_montages

def plot_distribution(paths:list,portion:str):
    healthy_path = paths[0]
    pneumonia_path = paths[1]

    numbers = [len(os.listdir(healthy_path)),len(os.listdir(pneumonia_path))]
    labels = ["Healthy","Pneumonia"]

    plt.bar(labels,numbers)
    plt.title(portion+" set distribution")
    plt.savefig(portion+"_distribution.png")
    plt.show()

def plot_montage(paths:list):
    h_files = paths[0]
    p_files = paths[1]

    healthy_images = []
    pneumonia_images = []

    for path in h_files:
        h_img = cv2.imread("chest_xray\\train\\NORMAL\\"+path)
        healthy_images.append(h_img)
    for path in p_files:
        p_img = cv2.imread("chest_xray\\train\\PNEUMONIA\\"+path)
        pneumonia_images.append(p_img)
    
    healthy_images = np.array(healthy_images)
    pneumonia_images = np.array(pneumonia_images)
    cv2.imshow("",healthy_images[0])
    cv2.waitKey(0)
    healthy = build_montages(healthy_images,(300,300),(3,3))
    pneumonia = build_montages(pneumonia_images,(300,300),(3,3))
    
    for montage in healthy:
        cv2.imshow("Healthy", montage)
        cv2.waitKey(0)

    for montage in pneumonia:
        cv2.imshow("Pneumonic", montage)
        cv2.waitKey(0)


plot_distribution(["chest_xray\\train\\NORMAL","chest_xray\\train\\PNEUMONIA"],"train")
plot_distribution(["chest_xray\\test\\NORMAL","chest_xray\\test\\PNEUMONIA"],"test")

plot_montage([list(os.listdir("chest_xray\\train\\NORMAL")[:9]), list(os.listdir("chest_xray\\train\\PNEUMONIA")[:9])])