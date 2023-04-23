# Uprava obrazka
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Nadchádzajúca implementácia v Pythone je vyvinutá z otvoreného zdrojového kódu matlabovej implementácie BCET
def BalanceContrastEnhancementTechnique(gray_image):
    # normalizacia pixelov medzi 0 a 1
    obrazok_in = normalizacia_img_double(gray_image)
    l = np.min(obrazok_in.ravel())                      # pixel jas min
    h = np.max(obrazok_in.ravel())                      # pixel jas max
    e = np.mean(obrazok_in)                             # pixel jas priemer
    s = np.mean(pow(obrazok_in, 2))                     # rozptyl pixelov
    L = 0                                               # MINIMUM očakavany obrázok
    H = 255                                             # MAXIMUM
    E = 85                                              # MEAN  > 80 (Odporucane)

    # rovnice BCET
    b_menovatel = pow(h, 2) * (E - L) - s * (H - L) + pow(l, 2) * (H - E)
    b_citatel = 2 * (h * (E - L) - e * (H - L) + l * (H - E))

    b = b_menovatel / b_citatel
    a = (H - L) / ((h - l) * (h + l - 2 * b))
    c = L - a * pow((l - b), 2)

    y = a * pow((obrazok_in - b), 2) + c                # PARABOLICKA FUNCKIA
    # kedže pixel = 0-255 všetky prvky v poli prevedieme na také čisla
    y = y.astype(np.uint8)

    return y

def normalizacia_img_double(im):
    # pretranformovanie pola do jedneho riadku najdeme min a max a normalizujeme od 0-1
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)

    return out

def change_color_fuzzycmeans(cluster_membership, clusters):
 img = []
 for pix in cluster_membership.T:
     img.append(clusters[np.argmax(pix)])

 return img

def segmentation(fcm_image,ths_image):
     brain = fcm_image
     tumor = ths_image
     segimg = cv2.addWeighted(brain, 0.5, tumor, 0.7, 0)

     return segimg

def dataanalysis(seg_img):
 try:
    detected_edges = cv2.Canny(seg_img, 10, 10 * 3, 5)
    cv2.applyColorMap(seg_img, cv2.COLORMAP_JET)

    return detected_edges
 except Exception as e:
    print("Error=" + e.args[0])
    tb = sys.exc_info()[2]
    print(tb.tb_lineno)

def FCM(img_bcet):
    list_img = []
    img = cv2.imread(str(img_bcet))
    rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
    list_img.append(rgb_img)
    clusters = [2]

    for index, rgb_img in enumerate(list_img):
        img = np.reshape(rgb_img, (resize_val, resize_val, 3)).astype(np.uint8)
        shape = np.shape(img)
        for i, cluster in enumerate(clusters):
            center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None, seed=42)

            novy_obrazok = change_color_fuzzycmeans(u, center)
            fuzzy_obrazok = np.reshape(novy_obrazok, shape).astype(np.uint8)
            # treshold pre oblast tumoru v mozgu
            ret, seg_img = cv2.threshold(fuzzy_obrazok, np.max(fuzzy_obrazok) - 1, 255, cv2.THRESH_BINARY)
            seg_img_1d = seg_img[:, :, 1]

            return seg_img_1d

            # nacitanie obrazka a uprava do 256 px a grayscale
resize_val = 256

# nacitanie obrazka do pamate v podobe intenzity pixelov
# extrakcia obrysov
image1 = cv2.imread('tumor_MRI_1.jpg')
image1 = cv2.resize(image1, (resize_val, resize_val), interpolation = cv2.INTER_AREA)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# plt.imshow(image)
# plt.show()

# odstránenie šumu z obrazu pomocou medianBlur z knižnice openCV
img_median = cv2.medianBlur(image, 5)
# plt.imshow(img_median)
# plt.show()

# segmentácia okolia mozgu a okolia tumoru v mozgovej casti
# normalna oblast mozgu sa rieši pomoocu FCM
# BCET zvyšenie kontrastu obrázka
image_bcet = BalanceContrastEnhancementTechnique(img_median)
cv2.imwrite('img_bcet.jpg', image_bcet)

#FCM segmentation for normal region of brain
image_FCM = FCM('img_bcet.jpg')
# plt.imshow(image_FCM)
# plt.show()

blur = cv2.GaussianBlur(image_bcet,(5,5),0)
# najdenie oblasti iba tumoru podla tresh hodnoty
T, thresh_f = cv2.threshold(blur,200,255, cv2.THRESH_BINARY)
# plt.imshow(thresh_f)
# plt.show()

#Data analysis
seg_img=segmentation(image_FCM,thresh_f)
edges=dataanalysis(seg_img)
plt.imshow(edges)
plt.show()







