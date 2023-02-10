#! /usr/bin/env python
# -*- coding: utf8 -*-

import rospy
from sensor_msgs.msg import PointCloud2

import numpy as np
import ros_numpy as rnp
import cv2

import tf2_ros

import glob
import os

from tensorflow.keras.models import Sequential  # Model type to be used
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten 

from geometry_msgs.msg import TransformStamped

############################################################
# Detekcja obszarów ze znakami i wycinanie symboli
############################################################

class Sign:
    crop = None
    mask = None
    symbol = None
    label = 'unknown'
    corners = -1
    x = 0
    y = 0
    w = 0
    h = 0

def color_segmentation(img):
    h_min = 100
    h_max = 150
    s_min = 50
    s_max = 255 
    v_min = 100
    v_max = 200

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    return mask

def mask_refinement(mask):
    morph_elem = cv2.MORPH_ELLIPSE

    element = cv2.getStructuringElement(morph_elem, (5, 5), (2, 2))
    dst = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

    element = cv2.getStructuringElement(morph_elem, (3, 3), (1, 1))
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(morph_elem, (7, 7), (3, 3))
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, element)

    return dst

def select_contours(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])

    # Get the mass centers
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (int(mu[i]['m10'] / (mu[i]['m00'] + 1e-5)), int(mu[i]['m01'] / (mu[i]['m00'] + 1e-5)))


    contour_candidates = []
    
    print('\nDetected contours:')
    for i in range(len(contours)):
        ca = cv2.contourArea(contours[i])
        cl = cv2.arcLength(contours[i], True)

        if ca < 10:
            continue
        
        print(' * Contour[%3d] - Area: %8.0f - Length: %4.0f' % (i, ca, cl))

        if ca > 500:
            contour_candidates.append(i)

    print('\nCandidate contours:')
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    ratio_thr = 0.8
    selected_contours = []

    # Draw contours + hull results
    for c_id in contour_candidates:
        cl = cv2.arcLength(contours[c_id], True)
        hl = cv2.arcLength(hull_list[c_id], True)
        ratio = hl/cl
        if ratio > ratio_thr:
            selected_contours.append(c_id)

        print(' * Contour[%3d] - Length: %4.0f - Hull: %4.0f - Ratio: %.3f' % (c_id, cl, hl, ratio))

    return [contours[s] for s in selected_contours]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def count_corners(contour):
    from scipy.signal import find_peaks

    mu = cv2.moments(contour)
    (cx, cy) = (int(mu['m10'] / (mu['m00'] + 1e-5)), int(mu['m01'] / (mu['m00'] + 1e-5)))

    ds, phis = [], []
    for i in range(len(contour)):
        x, y = contour[i][0]
        d, rho = cart2pol(x-cx, y-cy)
        ds.append(d)
        phis.append(rho)

    min_id = np.argmin(ds)
    ds = np.roll(ds, -min_id)
    phis = np.roll(phis, -min_id)
    phis = np.unwrap(phis)
    phis = phis - min(phis)

    ds = cv2.blur(ds, (1,15))
    ds = np.reshape(ds, (ds.shape[0],))
    pks, _ = find_peaks(ds, width=5, prominence=5)

    return len(pks)

def crop_contour(img, contour):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, (255), -1)
    
    #crop = cv2.copyTo(img, mask)
    crop = img.copy()
    crop[mask == 0, :] = 0

    s = Sign()

    x,y,w,h = cv2.boundingRect(contour)

    s.x = x
    s.y = y
    s.w = w
    s.h = h

    s.crop = crop[y:y+h, x:x+w,:]
    s.mask = mask[y:y+h, x:x+w]

    return s

def detect_signs(img):
    ret = []

    mask = color_segmentation(img)
    mask = mask_refinement(mask)

    selected_contours = select_contours(mask)

    print('\nCorners count:')
    for i, cnt in enumerate(selected_contours):
        s = crop_contour(img, cnt)

        c = count_corners(cnt)
        s.corners = c
        print(' * Contour[%3d] - Corners: %d' % (i, c))

        thr = 200
        _, thr_out = cv2.threshold(cv2.cvtColor(s.crop, cv2.COLOR_BGR2GRAY), thr, 255, cv2.THRESH_OTSU)

        s.symbol = thr_out & s.mask
        # OPENING to eliminate the border
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
        s.symbol = cv2.morphologyEx( s.symbol, cv2.MORPH_OPEN, element)

        ret.append(s)

    return ret

############################################################
# Rozpoznawanie symboli numerycznych (z trójkątów)
############################################################

network = None

def load_network(path):
    model = Sequential()                                 # Linear stacking of layers

    # Convolution Layer 1
    model.add(Conv2D(16, (5, 5), input_shape=(50,50,1))) # 16 different 5x5 kernels -- so 16 feature maps
    model.add(Activation('relu') )                       # activation
    model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

    # Convolution Layer 2
    model.add(Conv2D(32, (5, 5)))                        # 32 different 5x5 kernels -- so 32 feature maps
    model.add(Activation('relu'))                        # activation
    model.add(MaxPooling2D(pool_size=(2,2)))             # Pool the max values over a 2x2 kernel

    model.add(Flatten())                                 # Flatten final output matrix into a vector

    # Fully Connected Layer 
    model.add(Dense(128))                                # 128 FC nodes
    model.add(Activation('relu'))                        # activation

    # Fully Connected Layer                        
    model.add(Dense(10))                                 # final 10 FC nodes
    model.add(Activation('softmax'))                     # softmax activation
    #model.summary()

    model.load_weights(path)

    # Check its architecture
    model.summary()

    return model

def recognize_number(sign):
    desired_size = 50
    im = sign.symbol

    cur_size = im.shape[:2]
    ratio = float(desired_size)/max(cur_size)
    new_size = tuple([int(x*ratio) for x in cur_size])

    # resize to fit 50x50 box
    img = cv2.resize(im, (new_size[1], new_size[0]))

    # pad borders with black pixels
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = 0
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    print(img.shape)

    pred = network.predict(img)
    return str(np.argmax(pred))

############################################################
# Rozpoznawanie symboli graficznych (z okręgów)
############################################################

models = []

def load_models(path):
    ret = []
    model_files = glob.glob(os.path.join(path, '*'))
    for mf in model_files:
        model = {
            'label': os.path.splitext(os.path.basename(mf))[0], 
            'img': cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        }
        ret.append(model)
        rospy.loginfo("Model loaded [%d x %d] %s", model['img'].shape[1], model['img'].shape[0], model['label'])

   
    return ret

def recognize_symbol(sign, models):
    dm = np.zeros((len(models)))
    for i,m in enumerate(models):
        d = cv2.matchShapes(m['img'], sign.symbol, cv2.CONTOURS_MATCH_I3, 0)
        dm[i] = d

    return models[np.argmin(dm)]['label']


############################################################
# Funkcje pomocnicze do wizualizacji danych
############################################################

def draw_signs(img, signs):
    for s in signs:
        cv2.rectangle(img, (s.x, s.y), (s.x + s.w, s.y+s.h), (0, 0, 255))
        cv2.putText(img, s.label, (s.x, s.y-2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

def colorize(xyz, zmin = 0, zmax = 3):
    zrng = zmax - zmin
    tmpz = xyz[:,:,2].copy()
    tmpz = ((tmpz - zmin) / zrng * 255)
    tmpz = np.clip(tmpz, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(tmpz, cv2.COLORMAP_JET)

def pc2rgbxyz(points_msg):
    pc = rnp.numpify(points_msg)
    w = points_msg.width
    h = points_msg.height
    xyz = pc.view('<f4').reshape(h,w,pc.itemsize // 4)[:,:,0:3]
    rgb = pc['rgb'].copy().view('<u1').reshape(h,w,4)[:,:,0:3]

    return rgb, xyz

def tf2np(tr):
    from pyquaternion import Quaternion as Q
    ret = np.eye(4,4)
    r = tr.transform.rotation
    t = tr.transform.translation
    R = Q(r.w, r.x, r.y, r.z).rotation_matrix
    T = np.array([t.x, t.y, t.z])

    return R, T

def sym(w):
    from scipy.linalg import sqrtm, inv
    return w.dot(inv(sqrtm(w.T.dot(w))))

def transform(cen, ori, frame, parent, stamp):
    from pyquaternion import Quaternion as Q
    t = TransformStamped()

    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = frame
    t.transform.translation.x = cen[0]
    t.transform.translation.y = cen[1]
    t.transform.translation.z = cen[2]
    q = Q(matrix=sym(ori))
    t.transform.rotation.x = q.x
    t.transform.rotation.y = q.y
    t.transform.rotation.z = q.z
    t.transform.rotation.w = q.w

    return t

def publish_sign(sign, xyz, trans):
    import pyransac3d as pyrsc
    
    xyz_crop = xyz[sign.y:sign.y+sign.h, sign.x:sign.x+sign.w]
    xyz_points = xyz_crop[sign.mask != 0]
    cen = np.nanmean(xyz_points, axis=0)

    # dopasowanie płaszczyzny do punktów wybranych z chmury
    plane = pyrsc.Plane()
    p, _ = plane.fit(xyz_points, 0.05, 100, 100)
    
    # patrzymy na znak, więc wektor normalny powinien być skierowany do kamery
    k = np.array(p[0:3])
    if k[2] > 0:
        k = -k

    # oś y symbolu skierowana w górę
    j = np.array([0, -1, 0])
    # oś x wyznaczona ze znanej osi y i z
    i = np.cross(j, k)
    i = i / np.linalg.norm(i)
    j = np.cross(k, i)

    # złożenie osi układu współrzędnych w macierz obrotu
    ori = np.transpose(np.vstack((i, j, k)))

    # wyznaczenie macierzy obrotu i wektora przesunięcia
    R, T = tf2np(trans)

    # przekształcenie położenia znaku
    cen = np.matmul(R, cen) + T
    # przekształcenie orientacji znaku
    ori = np.matmul(R, ori)

    # publikacja transformacji, tym razem w układzie bazowym (odom)
    t = transform(cen, ori, "symbol_"+sign.label, trans.header.frame_id, trans.header.stamp)
    br = tf2_ros.TransformBroadcaster()
    br.sendTransform(t)

############################################################
# Przetwarzanie chmury punktów
############################################################

def callback(points_msg):
    global models

    while True:
        try:
            trans = tfBuffer.lookup_transform(target_frame, points_msg.header.frame_id, points_msg.header.stamp)
            break
        except:
            print("Retry tf lookup")
            rospy.sleep(0.1)
    
    # zamiana chmury punktów na obraz kolorowy oraz macierz współrzędnych
    rgb, xyz = pc2rgbxyz(points_msg)
    
    # detekcja symboli w obrazie
    signs = detect_signs(rgb)
    for i, s in enumerate(signs):
        cv2.imshow(str(i) + "_crop", s.crop)
        cv2.imshow(str(i) + "_mask", s.mask)
        cv2.imshow(str(i) + "_symb", s.symbol)

        if s.corners == 3:
            s.label = recognize_number(s)

        if s.corners < 3:
            s.label = recognize_symbol(s, models)

        publish_sign(s, xyz, trans)

    # wizualizacja wykrytych znaków
    vis = rgb.copy()
    draw_signs(vis, signs)

    cv2.imshow("dep", colorize(xyz))
    cv2.imshow("rgb", vis)

    cv2.waitKey(10)

    return


if __name__ == '__main__':
    rospy.init_node('symbol_detector', anonymous=True)

    target_frame = rospy.get_param('~target_frame', 'camera_rgb_optical_frame')

    # ładowanie modeli symboli graficznych
    model_path = rospy.get_param('~model_path', 'models')
    models = load_models(model_path)
    if len(models) < 1:
        rospy.logerr("No models found! Set the [model_path] param.")
        exit(1)

    # ładowanie modelu sieci neuronowej
    network_path = rospy.get_param('~network_path', 'network')
    network = load_network(network_path)

    # przygotowanie i wstępne wypełnienie bufora transformacji
    tfBuffer = tf2_ros.Buffer(rospy.Duration(120))
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.loginfo("Filling TF buffer")
    rospy.sleep(2)

    rospy.loginfo("Subscribing to topics")

    points_sub = rospy.Subscriber('points', PointCloud2, callback)

    rospy.spin()
