#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDisk version 2.0
@author: Sihan Wang（swang59@ncsu.edu）
@author: Kun Yang (yangkun118@sjtu.edu.cn)
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
import copy
import math
from skimage import feature
from skimage.feature import blob_log
from skimage.io import imsave
from skimage.transform import resize
from scipy import stats,signal




###################################################################
# This file includes the utilities of AutoDisk, an automated diffraction 
# pattern analysis method for 4D-STEM. This version covers the functions
# for diffraction disk recognition, lattice parameter estimation and
# lattice strain mapping.
#
# For details about the method, please refer to the manuscript:
# "AutoDisk: Automated Diffraction Processing and Strain Mapping in 4D-STEM"
# by Sihan Wang, Tim Eldred, Jacob Smith and Wenpei Gao.
###################################################################
#generateInt  drawCircles PM_slope print  generateInt_new  generateAvg find_center_of_mass tilt_int bresenham_line
#detect_ellipse generateAdf  radial_intensity move_center saveData gauss_fit ctrRadiusIni calcStrain array_average
#radGradMax detAng latFit latBack delArti average_minimum_distance rotate_image bresenham_area backmodel_denoising
#array_average calculate_radius vector_select readDatadat rotate_image generate_vectors cal_weight readData save
#generate_vectors histogram_analysis generateInt_new generateTilt calculate_radius ctrDet  average_minimum_distance
#drawCircles  generateKernel generateAvg ndimage array_average  calculate_radius generate_vectors

def visual(image, outdir=None, point=None, colorbar=False, normalization=False, title=None):
    """
    Convert a 2D array of int or float to an int8 array of image and visualize it.

    Parameters
    ----------
    image : 2D array of int or float
    outdir : str
    plot : bool, optional
        Ture if the image need to be ploted. The default is True.

    Returns
    -------
    image_out: 2D array of int8

    """
    if normalization is True:
        image_out = (((image - image.min()) / (image.max() - image.min())) * 255).astype(np.uint8)
    else:
        image_out = image
    fig, ax = plt.subplots(figsize = (5,5))
    im = ax.imshow(image_out,cmap='grey')  # viridis gray
    if point is not None:
        ax.plot(point[0][1], point[0][0], 'ro', markersize=point[1])
    if title is not None:
        plt.title(title)
        plt.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax)
    plt.show()
    if outdir is not None:
        directory = os.path.split(outdir)[0]
        if directory:
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
        fig.savefig(outdir,format='png',dpi=300)
    return image_out

import glob
def create_video_from_images(image_folder, video_path, frame_duration):
    """
    Creates a video from a sequence of images.

    :param image_folder: The folder where the PNG images are stored (do not end path with a slash).
    :param video_path: The full path where the output video will be saved.
    :param frame_duration: The duration each image will be shown in the video, in seconds.
    """
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print("No images found. Please check the path.")
        return
    frame = cv2.imread(images[0])
    if frame is None:
        print("Failed to read the first image. Please check the image file.")
        return
    height, width, layers = frame.shape
    frame_rate = 1 / frame_duration
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Can be changed to *'mp4v', *'X264', etc.
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    video.release()
    print("Video creation complete.")

def rearrange(image):
    """
    Parameters
    ----------
    image : 2D array of int or float
    
    Returns
    -------
    empty_array : 2D array of int or float

    """
    size=len(image)
    ctr=size//2
    empty_array = np.zeros((size, size))
    empty_array[0:ctr,0:ctr] = image[ctr:size,ctr:size]
    empty_array[ctr:size,0:ctr] = image[0:ctr,ctr:size]
    empty_array[0:ctr,ctr:size] = image[ctr:size,0:ctr]
    empty_array[ctr:size,ctr:size] = image[0:ctr,0:ctr]
    return empty_array

def readDatadat(dname,dim=2,pixel=480,pixel2=None,Normalized=True):   
    """
    Read in a 2D-STEM data file.
    
    Parameters
    ----------
    dname : str
        Name of the data file.

    Returns
    -------
    data: 2D array of int or float
        The read-in 2D-STEM data.
    """
    dimy=pixel
    if pixel2==None:
        dimx=pixel
    else:
        dimx=pixel2
    pro_dim = dim
    file = open(dname,'rb') 
    data = np.fromfile(file, np.float32) 
    if pro_dim==1:
        data = np.reshape(data, (dimy, dimx))
        if Normalized:
            data = data / np.sum(data)
        else:
            data = data
    else:
        data = np.reshape(data, (pro_dim, dimy, dimx))
        #print(np.sum(data[1])) 
        if pro_dim==2:
            data=data[1]
            if Normalized:
                data = data / np.sum(data)
            else:
                data = data
        else:
            data=data
    file.close()
    return data    
        
def readData(dname):   
    """
    Read in a 4D-STEM data file.
    
    Parameters
    ----------
    dname : str
        Name of the data file.

    Returns
    -------
    data: 4D array of int or float
        The read-in 4D-STEM data.
    
    """
    dimy = 130
    dimx = 128

    file = open(dname,'rb') 
    data = np.fromfile(file, np.float32)          
    pro_dim = int(np.sqrt(len(data)/dimx/dimy))
    
    data = np.reshape(data, (pro_dim, pro_dim, dimy, dimx))
    data = data[:,:,0:dimx, :]
    file.close()
    
    return data    

def bresenham_line(x0, y0, x1, y1):
    '''
    Aalculate all the integer coordinates that form a straight line between these two points on a 2D grid

    Parameters
    ----------
    x0 y0: int
        one of the position.
    x1 y1: int
        another position.

    Returns
    -------
    points : 1D list 
        All position between these two points.

    '''
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def circle_points(cx, cy, radius):
    '''
    Generate a list of points that form a circle.

    Parameters:
    cx : int
        x-coordinate of the center of the circle.
    cy : int
        y-coordinate of the center of the circle.
    radius : int
        Radius of the circle.

    Returns:
    list
        List of points representing the circle.
    '''
    points = []
    x, y = radius, 0
    P = 1 - radius
    while x >= y:
        points.extend([(cx + x, cy + y), (cx - x, cy + y), (cx + y, cy + x), (cx - y, cy + x),
                       (cx + x, cy - y), (cx - x, cy - y), (cx + y, cy - x), (cx - y, cy - x)])
        if x == y:
            break
        if P <= 0:
            P += 2 * y + 1
        else:
            x -= 1
            P += 2 * y - 2 * x + 1
        y += 1
    return points

def bresenham_angle(data_map, com, radius, specific_angle, plot=True, outdir=None):
    '''
    Find the closest boundary point on a circle to a specific angle and plot the line segment using Bresenham's algorithm.

    Parameters
    ----------
    data_map : numpy.ndarray
        Input image data.
    com : tuple
        Center coordinates of the circle.
    radius : int
        Radius of the circle.
    specific_angle : float
        Specific angle in degrees.
    plot : bool, optional
        Whether to display the plot. Default is True.
    outdir : str, optional
        Output directory to save the plot image. Default is None.

    Returns
    -------
    list
        List of points representing the line segment.
    float
        Closest angle to the specific angle.
    '''
    cy, cx = com
    # Get all possible integer points on the circle
    boundary_points = circle_points(cx, cy, radius)
    #print(boundary_points)
    # Find the closest boundary point to a particular angle 
    closest_point = None
    smallest_angle = float('inf')
    smallest_angle_difference = float('inf')
    for point in boundary_points:
        current_angle = math.degrees(math.atan2(cy - point[1], point[0] - cx))
        current_angle = current_angle if current_angle >= 0 else current_angle + 360
        angle_difference = abs(current_angle - specific_angle)
        if angle_difference < smallest_angle_difference:
            smallest_angle = round(current_angle, 2)
            smallest_angle_difference = angle_difference
            closest_point = point
    #print(cx, cy, closest_point[0], closest_point[1])# Getting the set of coordinate points on a line segment by Bresenham's algorithm
    line_points = bresenham_line(cx, cy, closest_point[0], closest_point[1])
    line_points.sort(key=lambda point: ((point[0] - cx) ** 2 + (point[1] - cy) ** 2))
    
    if plot==True:
        fig, ax = plt.subplots()
        ax.imshow(data_map, cmap='gray')
        x_values = [p[0] for p in line_points]
        y_values = [p[1] for p in line_points]
        plt.plot(x_values, y_values, marker='o', markersize=1)
        plt.plot(cx, cy, 'ro', markersize=1)  
        plt.gca().set_aspect('equal', adjustable='box') 
        plt.show()
        if outdir is not None:
            directory = os.path.split(outdir)[0]
            if directory:
                if not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)
            fig.savefig(outdir,format='png',dpi=300)
    return line_points, smallest_angle

def bresenham_area(X, Y, x0, y0, x1, y1, width):
    '''
    Calculate the area using Bresenham algorithm.
    
    Parameters
    ----------
    X : int
        Width of the area.
    Y : int
        Height of the area.
    x0 : int
        Starting x-coordinate.
    y0 : int
        Starting y-coordinate.
    x1 : int
        Ending x-coordinate.
    y1 : int
        Ending y-coordinate.
    width : int
        Width of the area.

    Returns
    -------
    position : numpy.ndarray
        Array representing the calculated area.
    '''
    position = []
    line_points = bresenham_line(x0, y0, x1, y1)
    for i in range(len(line_points)):
        local = line_points[i]
        N = (local[1])*X+local[0]
        for j in range(0,width,1):
            position.append(N+j)
    position = np.array(position).reshape(len(line_points),width)
    return position

def saveData(out_dir, data, overwrite=False, dataformat=None):
    """
    Save debye-waller facter into '.dat's.

    Parameters
    ----------
    out_dir : str
    data: 2D array of float
    overwrite: bool, optional
        Whether to overwrite the existing content. Default is False.
    dataformat: str
        If you want open the dat file with DigitalMicrograph in real 4 byte, you should let dataformat='np.float32'.
        
    Returns
    -------
    None.
    
    """
    if out_dir is not None:
        directory = os.path.split(out_dir)[0]
        if directory:
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
    if dataformat=='np.float32':
        arr = np.array(data, dtype=np.float32)
        try:
            with open(out_dir, 'wb' if overwrite else 'ab') as f:
                arr.tofile(f)
        except OSError as e:
            print(f"Error occurred while writing to {out_dir}: {e}")
    else:
        if overwrite and os.path.exists(out_dir):  
            os.remove(out_dir)
        mode = 'w' if overwrite else 'a'
        with open(out_dir, mode) as f:
            for row in data:
                # Check if row is iterable, if not, convert it to a list
                if not hasattr(row, '__iter__'):
                   row = [row]
                f.write(' '.join([str(elem) for elem in row]) + '\n')
    pass


def savePat(out_dir, data, ext ='.tif'):
    """
    Save diffraction patterns into '.tif's.

    Parameters
    ----------
    out_dir : str
        The name of the save folder.
    data : 2D array of int or float
        Array of a 4D dataset.
    ext : str, optional
        Extension of the output pattern. The default is '.tif'.

    Returns
    -------
    None.

    """
    pro_dim,pro_dim = data.shape[:2]
    out_dir = os.path.join(out_dir)
    for i in range(pro_dim):
        for j in range(pro_dim):
            pattern = data[i,j]       
            imsave(out_dir+np.str_(i)+'_'+np.str_(j)+".tif", pattern, plugin="tifffile")
    
    pass


from skimage import measure
def generateAdf(data,in_rad,out_rad,center=None,angle=None,show_mask=False,outdir=None,dataformat='np.float32', imagedata=False): 
    """
    Generate an annular dark-field image from the diffraction patterns.

    Parameters
    ----------
    data : 4D array of int or float
        The 4D dataset.
    in_rad : int
        Inner collection angle.
    out_rad : int
        Outer collection angle.

    Returns
    -------
    None.

    """
    imgh,imgw,pxh,pxw = data.shape
    i = imgh//2
    j = imgw//2
    data[np.where(np.isnan(data)==True)] = 0
    data[i,j,:,:] -= np.min(data[i,j,:,:])
    data[i,j,:,:] += 0.0000000001
    if center is None:
        cx=pxh//2
        cy=pxw//2
    else:
        #ctr_ori = find_center_of_mass(data,threshold_value = 0.001)
        cx=center[0]  #lie
        cy=center[1]  #hang
    y, x = np.meshgrid(np.arange(pxh), np.arange(pxw), indexing='ij')
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    angles = np.degrees(np.arctan2(y - cy, x - cx)) % 360
    mask_img = np.zeros((pxh, pxw))
    if angle is not None:
        for k in range(0, len(angle), 2):
            start_angle = angle[k]
            end_angle = angle[k + 1]
            mask_img[(distance >= in_rad) & (distance <= out_rad) & 
                     (angles >= start_angle) & (angles <= end_angle)] = 1
    else:
        mask_img[(distance >= in_rad) & (distance <= out_rad)] = 1
    adf = np.mean(data * mask_img, axis=(-2, -1))
    plt.imshow(adf,cmap='gray')
    if outdir is not None:
        folder = os.path.dirname(outdir)
        if not os.path.exists(folder):  # Check if folder exists, if not, create it
            os.makedirs(folder)
        outdir_tif = outdir + '.png'
        plt.savefig(outdir_tif,format='png',dpi=300)
        plt.show()
        outdir_dat = outdir + '.txt'
        if dataformat=='np.float32':
            saveData(outdir_dat,adf,overwrite=True,dataformat='np.float32')
        elif dataformat=='int':
            saveData(outdir_dat,adf,overwrite=True)
    if show_mask==True:
        avg_image = generateAvg(data)
        # Draw the mask contours
        fig, ax = plt.subplots()
        #ax.scatter(cx-1, cy-1, color='blue', s=10, label='Scatter Points')
        ax.imshow(np.sqrt(np.sqrt(np.sqrt(avg_image))), cmap='gray')
        contours = measure.find_contours(mask_img, level=0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)
        plt.title('Average Image with Mask Contours')
        if outdir is not None:
            outdir_mask = outdir +'_mask.png'
            plt.savefig(outdir_mask,format='png',dpi=300)
        plt.show()
    if imagedata==True:
        return np.array(adf)
    else:
        pass

def generateAvg(data, normalization=False):
    """
    Generate an average (sum) pattern from the 4D dataset.

    Parameters
    ----------
    data : 4D array of int or float
        Array of the 4D dataset.

    Returns
    -------
    avg_pat: 2D array of int or float
        An average (sum) difffraction pattern.

    """
    pro_y,pro_x = data.shape[:2]
    avg_pat = data[0,0]*1
    avg_pat[:,:] = 0
    for row in range (pro_y):
        for col in range (pro_x):
            avg_pat += data[row,col]
    #avg_pat = (((avg_pat - avg_pat.min()) / (avg_pat.max() - avg_pat.min())) * 255).astype(np.uint8)
    #avg_pat = avg_pat/(pro_y*pro_x)
    if normalization==True:
        avg_pat = avg_pat/np.sum(avg_pat)
    return avg_pat

def ctrRadiusIni(pattern):
    """
    Find the center coordinate and the radius of the zero-order disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.

    Returns
    -------
    ctr : 1D array of int or float
        Array of the center coordinates [row,col].
    avg_r : float
        Radius of the center disk in unit of pixels.

    """
    h,w = pattern.shape
    ctr = h//2
    pix_w = pattern[ctr,:]
    pix_h = pattern[:,ctr]
    
    fir_der_w = np.abs(pix_w[:1]-pix_w[1:])
    sec_dir_w_r = np.array(fir_der_w[w//2:-1]-fir_der_w[w//2+1:])
    sec_dir_w_l = np.array(fir_der_w[1:w//2]-fir_der_w[:w//2-1])
    avg_pos1_w = np.where(sec_dir_w_r==sec_dir_w_r.max())[0][0]
    avg_pos2_w = np.where(sec_dir_w_l==sec_dir_w_l.max())[0][0]
    avg_r_w = np.mean([avg_pos1_w+1,len(sec_dir_w_l)-avg_pos2_w])
    ctr_w =  np.mean([w//2 + avg_pos1_w + 1,avg_pos2_w + 2])
    
    fir_der_h = np.abs(pix_h[:1]-pix_h[1:])
    sec_dir_h_b = np.array(fir_der_h[h//2:-1]-fir_der_h[h//2+1:])
    sec_dir_h_u = np.array(fir_der_h[1:h//2]-fir_der_h[:h//2-1])
    avg_pos1_h = np.where(sec_dir_h_b==sec_dir_h_b.max())[0][0]
    avg_pos2_h = np.where(sec_dir_h_u==sec_dir_h_u.max())[0][0]
    avg_r_h = np.mean([avg_pos1_h+1,len(sec_dir_h_u)-avg_pos2_h])
    ctr_h =  np.mean([h//2 + avg_pos1_h + 1, avg_pos2_h+2])
    
    avg_r = np.mean([avg_r_w,avg_r_h])
    ctr = np.array([ctr_h,ctr_w])
    
    return ctr,avg_r  

from scipy.signal import find_peaks
def calculate_radius(main_spot, center, thre=0.25, r=None, plot=True):
    """
    Calculate the radius of a spot in an image based on its gradient magnitude.

    Parameters:
    ----------
        main_spot (ndarray): The input 2D array representing the spot.
        center (tuple): The coordinates (x, y) of the center of the spot.
        r (float, optional): The maximum distance from the center to consider for radius calculation.

    Returns:
    ----------
        float: The calculated radius of the spot.
    """
    '''
    x_center, y_center = center
    # Calculate gradient magnitude
    dx = np.gradient(main_spot, axis=0)
    dy = np.gradient(main_spot, axis=1)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    # Calculate distances from each pixel to the center
    x, y = np.meshgrid(np.arange(main_spot.shape[0]), np.arange(main_spot.shape[1]))
    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    # Apply radius constraint if provided
    if r is not None:
        gradient_magnitude = gradient_magnitude * (distances <= r)
    # Find peaks in the gradient magnitude
    peaks, _ = find_peaks(gradient_magnitude.flatten())
    rows, cols = np.unravel_index(peaks, gradient_magnitude.shape)
    #plt.scatter(cols, rows, color='r', marker='o')
    # Calculate the mean distance from the center to the peaks as the radius
    distances = np.sqrt((rows - x_center)**2 + (cols - y_center)**2)
    radius = np.mean(distances)
    if plot==True:
        fig, ax = plt.subplots(figsize = (5,5))
        ax.imshow(main_spot,cmap='gray')
        y,x = center
        c = plt.Circle((x, y), radius, color='red', linewidth=1, fill=False)
        ax.add_patch(c)
        plt.show()
    '''
    EDdata, center = move_center(main_spot , center=center, square_size=main_spot.shape[0], plot=False)
    distances, intensities = radial_intensity(EDdata, bin_width=1)
    mask_x = (distances <= main_spot.shape[0]/2)
    distances = distances[mask_x]
    intensities = intensities[mask_x]
    def smooth(data, window_size=3):
        if window_size < 1:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    smoothed_intensities = smooth(intensities, window_size=2)  # 获取平滑后的 intensities
    x_values = np.arange(len(smoothed_intensities))  # 创建 x 值数组
    peaks, _ = find_peaks(smoothed_intensities)
    outpixel = int((peaks[0]+peaks[1])/2)
    if plot==True:
        plt.figure(figsize=(8, 6))
        plt.plot(distances, intensities, color='black')
        plt.plot(x_values, smoothed_intensities, color='red')
        plt.plot(peaks, smoothed_intensities[peaks], "o", label='Peaks')
        plt.axvline(x=outpixel, color='green', linestyle='--', label=f'x={outpixel}')
        plt.xlabel('Distance from center (pixels)', fontname='Arial', fontweight='bold', fontsize=16)
        plt.ylabel('Integrated Intensity', fontname='Arial', fontweight='bold', fontsize=16)
        plt.title('Radial Intensity Profile', fontname='Arial', fontweight='bold', fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.xticks(fontname='Arial', fontweight='bold', fontsize=12)
        plt.yticks(fontname='Arial', fontweight='bold', fontsize=12)
        plt.show()
    from skimage.draw import line
    def extract_gray_values(image, start, end, width=1):
        rr, cc = line(start[0], start[1], end[0], end[1])  # 获取直线的行列坐标
        gray_values = []
        for i in range(len(rr)):
            y=rr[i]
            x=cc[i]
            total_sum = 0
            for dy in range(-width // 2, width // 2 + 1):
                new_y = y + dy
                if 0 <= new_y < image.shape[0]:
                    total_sum += image[new_y, x]
            gray_values.append(total_sum)
        return np.array(gray_values), rr, cc
    gray_values1, rr, cc = extract_gray_values(EDdata, (center[0], center[1]-outpixel), (center[0], center[1]+outpixel), width=5)
    x_values1 = np.arange(len(gray_values1))  # 创建 x 值数组
    from scipy.signal import resample
    def half_h(gray_values):
        new_length = len(gray_values) * 4
        gray_values = resample(gray_values, new_length)
        max_value = np.max(gray_values)
        half_max = max_value * thre
        indices_above_half_max = np.where(gray_values >= half_max)[0]
        if len(indices_above_half_max) > 0:
            fwhm = (indices_above_half_max[-1] - indices_above_half_max[0])/4  # FWHM = 右边界 - 左边界
        else:
            fwhm = 0
        return fwhm
    fwhm = half_h(gray_values1)
    if plot==True:
        ED = np.sqrt(np.sqrt(np.sqrt(EDdata)))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(ED, cmap='viridis')
        axs[0].plot(cc, rr, color='red', linestyle=':', linewidth=2, label='Selected Area Contour')  # Overlay the contour
        axs[0].legend(loc='upper right', prop={'family': 'Arial', 'weight': 'bold', 'size': 16})
        axs[1].plot(x_values1, gray_values1, label='Gray Value Changes')
        axs[1].axvline((max(x_values1)+1)/2 - fwhm / 2, color='orange', linestyle='--', label='FWHM')
        axs[1].axvline((max(x_values1)+1)/2 + fwhm / 2, color='orange', linestyle='--')
        #for ax in axs:
        #    ax.set_xticklabels(ax.get_xticks(), fontname='Arial', fontweight='bold', fontsize=12)
        #    ax.set_yticklabels(ax.get_yticks(), fontname='Arial', fontweight='bold', fontsize=12)
        axs[1].legend(loc='upper right', prop={'family': 'Arial', 'weight': 'bold', 'size': 16})
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.show() 
    return fwhm/2

from scipy.ndimage import center_of_mass
from scipy.ndimage import label
def find_center_of_mass(data, blobs=None, r=None, num=None, threshold_value=None,mask=None,
                        plot=True, outdir=None, pad_width=None):
    '''
    Find the main transmission spot in the given 2D list.
    Parameters
    ----------
    data (list): The 2D list representing the diffraction pattern.
    threshold_value (float): The threshold value for enhancing the main spot. Default is 0.4.

    Returns
    -------
    tuple: A tuple containing the center coordinates of the main spot and its radius.
    '''
    data = np.array(data)  # Convert the list to a numpy array
    if pad_width is not None:
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
        blobs = blobs + pad_width
    if threshold_value is not None:
        spot_array = data
        main_spot = spot_array > threshold_value  # Apply threshold to enhance the main spot
        labeled_array, num_features = label(main_spot)  # Label the connected components of the main spot
        if num_features!=1:
            print('Please change centerofMass:threshold_value,',num_features,'spots have been found from threshold_value!')  
        center = center_of_mass(main_spot, labeled_array, range(1, num_features+1))  # Find the center of mass of the main spot
        #center=center[0]
        if center:  # 检查列表是否非空
            center = center[0]
        else:
            print("Error: No center of mass found!")
            center = [0, 0]  # 设置默认值  
    if mask is not None:  #mask=[x,y,r]
        y, x = np.ogrid[:data.shape[0], :data.shape[1]]
        area = np.zeros_like(data, dtype=bool)
        area[(y - mask[0])**2+(x - mask[1])**2 < mask[2]**2] = True
        ctr_position = center_of_mass(data, area)
        center = np.array(ctr_position)  
    if blobs is not None:
        center=[]
        y, x = np.ogrid[:data.shape[0], :data.shape[1]]
        for blob in blobs:
            area = np.zeros_like(data, dtype=bool)
            area[(y - blob[0])**2+(x - blob[1])**2 < r**2] = True
            if plot is True:
                fig, ax = plt.subplots(figsize = (5,5))
                ax.imshow(np.sqrt(np.sqrt(np.sqrt(data))), cmap='gray')
                y,x = blob
                c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
                ax.add_patch(c)
                plt.axis('off')
            ctr_position = center_of_mass(data, area)
            center.append(ctr_position)
        if plot is True:
            if outdir is not None:
                directory = os.path.split(outdir)[0]
                if directory:
                    if not os.path.isdir(directory):
                        os.makedirs(directory, exist_ok=True)
                name, ext = os.path.splitext(os.path.split(outdir)[1])
                plt.title(f"{name}", fontsize=12)
                plt.tight_layout(pad=0)
                plt.savefig(outdir, dpi=300)
            plt.show()
    center = np.array(center)
    if pad_width is not None:
        center = center - pad_width
    return center

def area_int(data,ref_ctr,ctr,r,num=None):
    """
    This function calculates the intensity of the square area and position of pixels around a given center point.
    
    Parameters
    ----------
    data: The input data containing pixel values
    ref_ctr: The reference center points
    ctr: The current center point
    r: The radius around the center point
    num: The number of center points to consider
    
    Returns:
    ----------
    arr_int: List of intensities calculated for each center point
    position_pixel_all: List of positions of pixels for each center point
    """
    ref_ctr = np.array(ref_ctr)
    ctr_vec = ref_ctr[:,:2]
    #print(len(ctr_vec))
    if num is not None:
        distances = np.linalg.norm(ctr_vec - ctr, axis=1)
        sorted_indices = np.argsort(distances)
        ctr_vec = ctr_vec[sorted_indices[:num]]
        ctr_vec = sorted(ctr_vec, key=lambda x: (x[0], x[1]))
        #print(np.array(ctr_vec[:num]))# - np.array(ctr))
    arr_int = generateInt_new(data,ctr_vec,r)
    position_pixel_all = []
    for j in ctr_vec:
        position_pixel = []
        y = int(j[0]-r)
        x = int(j[1]-r)
        for m in range(int(r+0.5)*2+2):
            for n in range(int(r+0.5)*2+2):
                position_pixel.append((y+m, x+n))     
        position_pixel_all.append(position_pixel)
    return arr_int, position_pixel_all

def average_minimum_distance(points, avg_num=7):
    """Calculate the average minimum distance among all points in the set using vectorized operations"""
    points = points[np.argsort(np.linalg.norm(points - points[0], axis=1))]
    avg_distance=0
    for value in points[:avg_num]:
        sorted_points = points[np.argsort(np.linalg.norm(points - value, axis=1))]
        distances = np.linalg.norm(sorted_points[1:5] - value, axis=1)
        avg_distance += np.mean(distances) 
    avg_distance=avg_distance/avg_num
    return avg_distance
    
    '''
    # Compute all pairwise distances
    dist_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # Set diagonal to infinity to ignore zero distances from points to themselves
    np.fill_diagonal(dist_matrix, np.inf)
    # Find the minimum distance for each point
    min_distances = np.min(dist_matrix, axis=1)
    # Compute the average of the minimum distances
    return np.mean(min_distances)
    '''
def PM_slope(intensity, distance, atom):
    '''

    Parameters
    ----------
    intensity : 1D array of int or float
        The intensity of diffraction pattern.
    distance : 1D array of int or float
        The pixel between the diffraction disks and the center disk.

    Returns
    -------
    the slope(the fitting of Paul. Midgley).

    '''    # Acta Cryst. (1991). A47, 590-597
    distance = np.array(distance)
    s = []
    f = []
    if atom =='Si':
        s = [d / (2*12*5.44) for d in distance]
        f = [(0.02395*14/4.5*(1-math.exp(-1.737*m**2))+0.02395*14/4.5*(1-math.exp(-3.043*m**2))+0.02395*14/4.5*(1-math.exp(-30.57*m**2))
              +0.5*0.02395*14/4.5*(1-math.exp(-0.0507*m**2))+0.5*0.02395*14/4.5*(1-math.exp(-0.9918*m**2))
              +0.5*0.02395*14/4.5*(1-math.exp(-86.18*m**2)))/m**2 for m in s]
    elif atom =='Si110':
        s = distance
        f = [(0.02395*14/4.5*(1-math.exp(-1.737*m**2))+0.02395*14/4.5*(1-math.exp(-3.043*m**2))+0.02395*14/4.5*(1-math.exp(-30.57*m**2))
              +0.5*0.02395*14/4.5*(1-math.exp(-0.0507*m**2))+0.5*0.02395*14/4.5*(1-math.exp(-0.9918*m**2))
              +0.5*0.02395*14/4.5*(1-math.exp(-86.18*m**2)))/m**2 for m in s]
    elif atom =='C':
        s = distance/2
        f = [(0.02395*6/4.5*(1-math.exp(-0.2946*m**2))+0.02395*6/4.5*(1-math.exp(-3.934*m**2))+0.02395*6/4.5*(1-math.exp(-24.98*m**2))
              +0.5*0.02395*6/4.5*(1-math.exp(-25.28*m**2))+0.5*0.02395*6/4.5*(1-math.exp(-25.47*m**2))
              +0.5*0.02395*6/4.5*(1-math.exp(-46.7*m**2)))/m**2 for m in s]
        #f = [0.0893*math.exp(-0.2465*m**2)+0.2563*math.exp(-1.71*m**2)+0.757*math.exp(-6.4094*m**2)+1.0487*math.exp(-18.6113*m**2)
        #       +0.3575*math.exp(-50.2523*m**2) for m in s]
    elif atom=='C_simulation':
        s = distance
        f = [(0.02395*6/4.5*(1-math.exp(-0.2946*m**2))+0.02395*6/4.5*(1-math.exp(-3.934*m**2))+0.02395*6/4.5*(1-math.exp(-24.98*m**2))
              +0.5*0.02395*6/4.5*(1-math.exp(-25.28*m**2))+0.5*0.02395*6/4.5*(1-math.exp(-25.47*m**2))
              +0.5*0.02395*6/4.5*(1-math.exp(-46.7*m**2)))/m**2 for m in s]
    elif atom =='P':
        s = [d / (2*7*7.86) for d in distance]
        f = [(0.02395*15/4.5*(1-math.exp(-0.1795*m**2))+0.02395*15/4.5*(1-math.exp(-2.632*m**2))+0.02395*15/4.5*(1-math.exp(-2.676*m**2))
              +0.5*0.02395*15/4.5*(1-math.exp(-34.57*m**2))+0.5*0.02395*15/4.5*(1-math.exp(-36.78*m**2))
              +0.5*0.02395*15/4.5*(1-math.exp(-54.06*m**2)))/m**2 for m in s]
    s = [i**2 for i in s]
    f = [j**2 for j in f]
    q = []
    q = [math.log(f[i]/I) for i, I in enumerate(intensity)]
    s = np.array(s)
    q = np.array(q)
    coefficients = np.polyfit(s, q, 1)
    slope, intercept = coefficients
    # calculation R²
    y_pred = s * slope + intercept  # 预测值
    r_squared = 1 - (np.sum((q - y_pred) ** 2) / np.sum((q - np.mean(q)) ** 2))
    return slope,s,q,r_squared
    
    
    
def generateKernel(pattern,ctr,r,c=0.7,pad=2,pre_def = False):
    """
    Generate the kernel for cross-correlation based on thee center disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        An array of a diffraction pattern.
    ctr : 1D array of float
        Array of the row and column coordinates of the center.
    r : float
        Radius of a disk.
    c : float, optional
        An coefficient to modify the kernel size. The default is 0.7.
    pad : int, optional
        A hyperparameter to change the padding size out of the feature. The default is 2.
    pre_def: bool, optional
        If True, read the pre-defined ring kernel. The default is False.

    Returns
    -------
    fil_ring : 2D array of float
        Array of the kernel.
    """
    if pre_def == True:
        ring = np.load("kernel_cir.npy")
        f_size = int(2*r*c)
        ring = resize(ring, (f_size, f_size))
        fil_ring = np.zeros((len(ring)+2*pad,len(ring)+2*pad),dtype=float)
        fil_ring[pad:-pad,pad:-pad] = ring
        return fil_ring
    
    y_st = int(ctr[0]-r+0.5-pad*2)
    y_end = int(ctr[0]+r+0.5+pad*2)
    x_st = int(ctr[1]-r+0.5-pad*2)
    x_end = int(ctr[1]+r+0.5+pad*2)
    # +0.5 to avoid rounding errors (always shift to right, so 0,5 is modified to 1.5)
    
    if y_end-y_st==x_end-x_st:
        ctr_disk = pattern[y_st:y_end,x_st:x_end] 
    elif y_end-y_st>x_end-x_st:
        ctr_disk = pattern[y_st+1:y_end,x_st:x_end] 
    else:
        ctr_disk = pattern[y_st:y_end,x_st+1:x_end] 
        
    ctr_disk = (((ctr_disk - ctr_disk.min()) / (ctr_disk.max() - ctr_disk.min())) * 255).astype(np.uint8)  
    edge_det = feature.canny(ctr_disk, sigma=1)
    
    dim = len(ctr_disk)
    dim_hf = dim/2
    fil_ring = np.zeros((dim,dim))
    for i in range (dim):
        for j in range (dim):
            if edge_det[i,j]==True:
                if (i-dim_hf)**2+(j-dim_hf)**2>int(r-2)**2 and (i-dim_hf)**2+(j-dim_hf)**2<int(r+2)**2:
                    fil_ring[i,j] = 1
    
    coef = int(c*r)
    f_size = 2*coef
    fil_ring = resize(fil_ring, (f_size, f_size))
    
    return fil_ring



def crossCorr(pattern,kernel):
    """
    Cross correlate the pattern with the kernal.

    Parameters
    ----------
    pattern : 2D array of int or float
        Array of a diffraction pattern to be cross correlated.
    kernel : 2D array of float
        Array of the kernel.

    Returns
    -------
    cro_img_out : 2D array
        Cross correlated result of the input pattern.

    """
    cro_cor_img = signal.correlate2d(pattern, kernel, boundary='symm', mode='same')
    #cro_img_out = np.sqrt(cro_cor_img)
    

    return cro_cor_img



def samePadding(img,kernel):
    """
    Generate a padding outside of the image with the average intensity on the boundary of the image.

    Parameters
    ----------
    img : 2D array of int or float
        Array of the image.
    kernel : 2D array of float
        Array of the kernel.

    Returns
    -------
    constant : 2D array
        The image with a constant padding.
        
    """
    f_size = len(kernel)
    constant = np.empty((img.shape[0]+2*f_size,img.shape[1]+2*f_size))
    bcgd = np.mean(img[:f_size,f_size:])
    #constant[0:f_size,:] = constant[-f_size:img.shape[0]+2*f_size,:] = constant[:,0:f_size] = constant[:,f_size:img.shape[1]+2*f_size] = bcgd
    constant[0:f_size,:] = constant[f_size:img.shape[0]+2*f_size,:] = constant[:,0:f_size] = constant[:,f_size:img.shape[1]+2*f_size] = bcgd
    constant[f_size:img.shape[0]+f_size,f_size:img.shape[1]+f_size] = img
    
    return constant



def ctrDet(pattern, r, kernel, n_sigma=10, thred=0.2, ovl=0):
    """
    Detect disks on a pattern.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    r : float
        Radius of a disk.
    kernel : 2D array of float
        Kernel used for cross correlation.
    n_sigma : int, optional
        The number of intermediate values of standard deviations. The default is 10.
    thred : float, optional
        The absolute lower bound for scale space maxima. The default is 0.1.
    ovl : float, optional
        Acceptable overlapping area of the blobs. The default is 0.

    Returns
    -------
    blobs : 2D array of int
         Corrdinates of the detected disk position.

    """
    adjr = r * 0.5
    
    img = samePadding(pattern,kernel)  
    sh,sw = img.shape

    blobs_log = blob_log(img, 
                 min_sigma=adjr,
                 max_sigma=adjr, 
                 num_sigma=n_sigma, 
                 threshold= thred,
                 overlap = ovl)    
    #print(blobs_log)
    rem = []
    f_size = len(kernel)
    for i in range (len(blobs_log)):
        if np.any(blobs_log[i,:2]<f_size+5) or np.any(blobs_log[i,0]>sh-f_size-5) or np.any(blobs_log[i,1]>sw-f_size-5):
            rem.append(i)
    
    blobs_log_out = np.delete(blobs_log, rem, axis =0)
    blobs_log_out -= f_size 
    
    blobs =  blobs_log_out[:,:2].astype(int)
    
    return blobs



def radGradMax(sample, blobs, r, rn=20, ra=2, n_p=40, threshold=3, ctr_ori=None): 
    """
    Radial gradient Maximum process.

    Parameters
    ----------
    sample : 2D array of float or int
        The diffraction pattern.
    blobs : 2D array of int or float
        Blob coordinates.
    r : float
        Radius of the disk
    rn : int, optional
        The total number of rings. The default is 20.
    ra : int, optional
        Half of the window size. The default is 2.
    n_p : int, optional
        The number of sampling points on a ring. The default is 40.
    threshold : float, optional
        A threshold to filter out outliers. The smaller the threshold is, the more outliers are detected. The default is 3.

    Returns
    -------
    ref_ctr : 2D array of float
        Array with three columns, y component, x component and the weight of each detected disk.

    """
    ori_ctr = blobs    
    h,w = sample.shape        
    adjr = r * 1   
    r_scale = np.linspace(adjr*0.8, adjr*1.2, rn)    
    theta = np.linspace(0, 2*np.pi, n_p)     
    ref_ctr = []

    for lp in range (len(ori_ctr)):
        test_ctr = ori_ctr[lp]
        ind_list = []
        for ca in range (-ra,ra):
            for cb in range (-ra,ra):
                cur_row, cur_col = test_ctr[0]+ca, test_ctr[1]+cb
                cacb_rn = np.empty(rn)
                for i in range (rn):
                    row_coor = np.array([cur_row + r_scale[i] * np.sin(theta) + 0.5]).astype(int)
                    col_coor = np.array([cur_col + r_scale[i] * np.cos(theta) + 0.5]).astype(int)
                    
                    row_coor[row_coor>=h]=h-1
                    row_coor[row_coor<0]=0
                    col_coor[col_coor>=w]=w-1
                    col_coor[col_coor<0]=0
                    
                    int_sum = np.sum(sample[row_coor,col_coor])
                    cacb_rn[i] = int_sum
                    
                cacb_rn[:rn//2] *= np.linspace(1,rn//2,rn//2) 
                cacb_diff = np.sum(cacb_rn[:rn//2]) - np.sum(cacb_rn[rn//2:])
                ind_list.append([cur_row, cur_col,cacb_diff])
        
        ind_list = np.array(ind_list) 
        ind_max = np.where(ind_list[:,2]==ind_list[:,2].max())[0][0]
        ref_ctr.append(ind_list[ind_max]) 

    ref_ctr = np.array(ref_ctr)

    # Check Outliers
    z = np.abs(stats.zscore(ref_ctr[:,2]))
    outlier = np.where(z>threshold)
    if len(outlier[0])>0:
        for each in outlier[0]:
            if np.linalg.norm(ref_ctr[each,:2]-[h//2,w//2])> r:
                ref_ctr = np.delete(ref_ctr,outlier[0],axis = 0)
    if ctr_ori is not None:
        ctr=ctr_ori
    else:
        ctr=blobs[0]
    ctr_vec = ref_ctr[:,:2] - ctr
    distance_refine = np.sqrt((ctr_vec[:,0])**2+(ctr_vec[:,1])**2)
    sorted_indices = distance_refine.argsort()
    ref_ctr = ref_ctr[sorted_indices]
        
    return ref_ctr

def cal_weight(data,blobs,r,ra=3):
    weight_all=[]
    height, width = data.shape
    Y, X = np.ogrid[:height, :width]
    for blob in blobs:
        dist_from_blob = np.sqrt((X - blob[1])**2 + (Y - blob[0])**2)
        #annular_mask = (dist_from_blob >= (r - ra)) & (dist_from_blob <= r)
        annular_mask = dist_from_blob <= r
        total_sum = np.sum(data[annular_mask])
        weight_all.append(total_sum)
    return weight_all

def generateInt(data,ref_ctr,ctr,r):
    """
    square area !!!
    
    Parameters
    ----------
    data : 2D-array of float
        The intensity value of the position average electron diffraction.
    ref_ctr : 2D-array
        The central position of a certain order electron diffraction spot from radGradMax-function.
    rank:int
        the order of electron diffraction pattern
    ctr : 1D array of float
        Array of the row and column coordinates of the center.
    r : float
        Radius of the disks.

    Returns
    -------
    arr_int: 1D array of float
        The sum of the intensity of a certain order electron diffraction spot.
    dif_pat_dis: 1D array of float
        The pixel distance between of diffraction spot and center spot.

    """
    ctr_vec = ref_ctr[:,:2] - ctr #+[0,1]
    distance = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
    dis_copy = copy.deepcopy(distance)
    dis_copy.sort()
    dis_copy=np.unique(dis_copy)
    #print(dis_copy) 
    dif_pat_dis = np.sqrt(dis_copy[:])
    arr_int = []
    position_pixel = []
    for i in range(0, len(dis_copy)): 
        dis = []
        intensity = 0
        idx_ctr = np.where(distance == dis_copy[i])[0] 
        #print(idx_ctr)
        if len(idx_ctr) == 1: 
            dis.append(ref_ctr[idx_ctr[0], :2]) 
        else: 
            for each in idx_ctr:   
                dis.append(ref_ctr[each, :2]) 
        dis = np.array(dis, dtype=int) 
        #print(dis)
        for j in range(len(idx_ctr)):
            y = int(dis[j][0]-r-0.5)
            x = int(dis[j][1]-r-0.5)
            for m in range(int(r+0.5)*2+1):
                for n in range(int(r+0.5)*2+1):
                    intensity +=data[y+m,x+n]
                    position_pixel.append((y+m, x+n))
        #print(len(idx_ctr))
        #print(intensity)
        intensity=intensity/len(idx_ctr) #Determine whether the calculated diffractive spot intensities are single or summed per order!!!!
        #print(intensity)
        arr_int.append(intensity)
    return arr_int ,dif_pat_dis

def generateInt_new(data, blobs, r, disk=None):
    """
    Calculate the intensity of blobs in the given data array within a certain radius.

    Parameters
    ----------
    data: The input data array.
    blobs: List of blob coordinates.
    r: Radius.
    disk: List of numbers representing the disk.

    Returns:
    ----------
    intensitys_sum: List of summed intensities for each disk.
    """
    intensitys = []
    intensitys_sum = []
    test=np.zeros_like(data)
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    for blob in blobs:
        x_center = blob[1]
        y_center = blob[0]
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        intensity = np.sum(data * (distances <= r))
        test= test + data * (distances <= r)
        intensitys.append(intensity)
    #visual(test)  #Visualization of the summing area
    total = 0
    if disk is not None:
        for num in disk:
            intensitys_sum.append(np.sum(intensitys[total:total+num]))
            total = total + num
        return np.array(intensitys_sum)
    else:
        return np.array(intensitys)

def approx_pixel_circle_area(x, y, r, i, j, N=5):
    # N: 采样点数（N×N），如N=5则25个点
    dx = np.linspace(-0.5, 0.5, N)
    dy = np.linspace(-0.5, 0.5, N)
    xx, yy = np.meshgrid(i + dx, j + dy)
    mask = (xx - x)**2 + (yy - y)**2 <= r**2
    return np.sum(mask) / (N*N)
def generateInt_new_1(data, blobs, r, disk=None, plot=False, N=5):
    """
    近似采样法加速：用N×N网格采样像素与圆的交集面积比例。
    """
    intensitys = []
    intensitys_sum = []
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    for blob in blobs:
        x_center = blob[1]
        y_center = blob[0]
        x_min = max(int(np.floor(x_center - r - 1)), 0)
        x_max = min(int(np.ceil(x_center + r + 1)), data.shape[1])
        y_min = max(int(np.floor(y_center - r - 1)), 0)
        y_max = min(int(np.ceil(y_center + r + 1)), data.shape[0])

        total_intensity = 0
        area_map = np.zeros_like(data)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                area = approx_pixel_circle_area(x_center, y_center, r, i, j, N)
                area_map[j, i] = area
                if area > 0:
                    total_intensity += area * data[j, i]
        #print('加权总强度:', total_intensity)
        intensitys.append(total_intensity)
        if plot:
            plt.figure(figsize=(10,10))
            plt.imshow(data, cmap='gray', interpolation='none')
            plt.scatter(x[area_map>0], y[area_map>0], color='lime', s=80, edgecolors='black', label='in circle (area>0)')
            plt.scatter([x_center], [y_center], color='red', label='center')
            circle = plt.Circle((x_center, y_center), r, color='blue', fill=False, linewidth=2, label='r=1')
            plt.gca().add_patch(circle)
            plt.legend()
            plt.title('Visualization of data and intensity region (area weighted)')
            plt.xlim(5,15)
            plt.ylim(95,105)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
    total = 0
    if disk is not None:
        for num in disk:
            intensitys_sum.append(np.sum(intensitys[total:total+num]))
            total = total + num
        return np.array(intensitys_sum)
    else:
        return np.array(intensitys)

def generateTilt(data,ctr_select,r):
    ctr_position = np.mean(ctr_select, axis=0)
    test=np.zeros_like(data)
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    for blob in ctr_select:
        x_center = blob[1]
        y_center = blob[0]
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        test= test + data * (distances <= r)
    mass_position = center_of_mass(data, test)
    visual(test)  #Visualization of the summing area
    dy = mass_position[0]-ctr_position[0]
    dx = mass_position[1]-ctr_position[1]
    return dy,dx

def generateNonedisk(data, blobs, r, plot=False):
    mask = np.ones_like(data)
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    for blob in blobs:
        x_center = blob[1]
        y_center = blob[0]
        distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        mask[distances <= r] = 0
    data_nodisk = data * mask
    if plot==True:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.colorbar()
        
        plt.subplot(122)
        plt.imshow(np.sqrt(np.sqrt(data_nodisk)))  # 原始数据与mask相乘
        plt.title('Masked Data')
        plt.colorbar()

        plt.tight_layout()
        plt.show()
    return data_nodisk

def rotate_point(point, ctr, angle_deg):
    point = np.array(point)
    ctr = np.array(ctr)
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])
    # Translate the point to the origin
    translated_point = point - ctr
    # Apply the rotation matrix
    rotated_point = np.dot(rotation_matrix, translated_point)
    # Translate the point back to its original position
    final_point = rotated_point + ctr
    final_point_rounded = np.round(final_point, decimals=1)
    return final_point_rounded

def move_center(data,center=None,square_size=None,plot=True,outdir=None):
    '''
    Moves a square region centered at the specified point to the center of a new image.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data.
    center : tuple(cy,cx)
        Coordinates of the center point.
    square_size : int
        Size of the square region to move.
    plot : bool, optional
        Whether to display the original and new images. Default is True.

    Returns
    -------
    numpy.ndarray
        Cropped square region from the original image.
    tuple
        Adjusted center coordinates in the new image.
    '''
    if center is None:
        com = ndimage.center_of_mass(data)
        center = (int(round(com[0])), int(round(com[1])))
        #print(center)
    center_y, center_x =center
    start_y = int(round(center_y - square_size // 2))
    start_x = int(round(center_x - square_size // 2))

    pad_top = int(round(max(0 - start_y, 0)))
    pad_bottom = int(round(max(start_y + square_size - data.shape[0], 0)))
    pad_left = int(round(max(0 - start_x, 0)))
    pad_right = int(round(max(start_x + square_size - data.shape[1], 0)))
    data_padded = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    start_y_1 = start_y + pad_top
    start_x_1 = start_x + pad_left
    data_new = data_padded[start_y_1:start_y_1+square_size, start_x_1:start_x_1+square_size]

    center_new = (int(round(center_x-start_x)), int(round(center_y-start_y)))
    if plot==True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title('Original Image')
        ax1.imshow(np.sqrt(np.sqrt(data)),cmap='gray')
        ax1.plot(center_x, center_y, 'ro', markersize=2)  # plot the center point
        rect = patches.Rectangle((start_x, start_y), square_size, square_size,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax2.set_title('New Image')
        ax2.imshow(np.sqrt(np.sqrt(data_new)), cmap='gray')
        ax2.plot(center_new[0], center_new[1], 'ro', markersize=2)  # plot the center point
        #ax1.axis('off')
        #ax2.axis('off')
        if outdir is not None:
            fig.savefig(outdir,format='png',dpi=300)
        plt.show()
    return data_new,center_new

def radial_intensity(data, bin_width, plot=False,lim=None):
    '''
    Calculate the radial intensity profile of an image.

    Parameters
    ----------
    data : numpy.ndarray
        Input image data.
    bin_width : float
        Width of each radial bin.

    Returns
    -------
    numpy.ndarray
        Centers of the radial bins.
    numpy.ndarray
        Sum of pixel intensities in each radial bin.
    '''
    center_x, center_y = data.shape[0] // 2, data.shape[1] // 2
    distances = []
    intensities = []
    # Calculate the distance of each pixel to the center and extract its intensity
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            distance = np.hypot(x - center_x, y - center_y)
            distances.append(distance)
            intensities.append(data[x, y])
    # Determine the maximum distance and create a series of bins
    max_distance = np.max(distances)
    bins = np.arange(0, max_distance + bin_width, bin_width)
    # Use numpy.histogram to calculate the sum of pixel intensities in each bin
    intensity_sums, edges = np.histogram(distances, bins=bins, weights=intensities)
    # Calculate the center distance of each bin
    bin_centers = (edges[:-1] + edges[1:]) / 2
    if plot==True:
        plt.figure(figsize=(8, 6))
        if lim is not None:
            mask_x = (bin_centers >= lim[0]) & (bin_centers <= lim[1]) 
            bin_centers = bin_centers[mask_x]
            intensity_sums = intensity_sums[mask_x]
            mask_y = (intensity_sums >= lim[2]) & (intensity_sums <= lim[3]) 
            bin_centers = bin_centers[mask_y]
            intensity_sums = intensity_sums[mask_y]
        plt.plot(bin_centers, intensity_sums, color='black')
        #plt.xlim(0, 180)
        #plt.ylim(0, 0.0002)
        plt.xlabel('Distance from center (pixels)', fontname='Arial', fontweight='bold', fontsize=16)
        plt.ylabel('Integrated Intensity', fontname='Arial', fontweight='bold', fontsize=16)
        plt.title('Radial Intensity Profile', fontname='Arial', fontweight='bold', fontsize=16)
        plt.grid(True)
        plt.xticks(fontname='Arial', fontweight='bold', fontsize=12)
        plt.yticks(fontname='Arial', fontweight='bold', fontsize=12)
        #plt.savefig(f'{folder}/{out}/profile/profile-data.png', dpi=300)
        plt.show()
    return bin_centers, intensity_sums

from scipy.ndimage import gaussian_filter1d
def gauss_smooth(distances, intensities, sigma=3, plot=True):
    """
    Apply Gaussian smoothing to the input data.
    
    Parameters:
    -----------
    distances : array-like
        X-axis data points
    intensities : array-like
        Y-axis data points (values to be smoothed)
    sigma : float, optional
        Standard deviation for Gaussian kernel. Controls smoothing level.
        Default is 3. Higher values create smoother curves.
    
    Returns:
    --------
    tuple
        distances : Original x-axis data
        smoothed_intensities : Smoothed y-axis data
    """
    
    # Apply Gaussian smoothing to the intensity data
    smoothed_intensities = gaussian_filter1d(intensities, sigma)
    
    if plot==True:
        plt.figure(figsize=(10, 5))
        plt.plot(distances, intensities, 'b-', label='Original', alpha=0.5, linewidth=2)
        plt.plot(distances, smoothed_intensities, 'r-', label='Smoothed', linewidth=2)
        plt.xlabel('Distance', fontname='Arial', fontweight='bold', fontsize=16)
        plt.ylabel('Intensity', fontname='Arial', fontweight='bold', fontsize=16)
        plt.legend(loc='best', frameon=False, prop={'family': 'Arial','weight': 'bold','size': 16} )
        plt.grid(True)
        plt.xticks(fontname='Arial', fontweight='bold', fontsize=12)
        plt.yticks(fontname='Arial', fontweight='bold', fontsize=12)
        #plt.savefig(f'{folder}/{out}/profile/profile-data.png', dpi=300)
        plt.show()
    return distances, smoothed_intensities

def count_nonzero_in_range(data_nodisk, start, end, plot=False, ax1='ax1'):
    """
    Count nonzero values within a specific radius range from the center.
    
    Parameters:
    -----------
    data_nodisk : 2D array
        Input array
    start : float
        Start radius
    end : float
        End radius
    
    Returns:
    --------
    int
        Number of nonzero values in the specified range
    """
    center_y, center_x = np.array(data_nodisk.shape) // 2
    y, x = np.ogrid[:data_nodisk.shape[0], :data_nodisk.shape[1]]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    range_mask = (distances >= start) & (distances < end)
    count = np.count_nonzero(data_nodisk[range_mask])
    # Optional: visualize the range (uncomment if needed)
    if plot==True: 
        circle1 = plt.Circle((center_x, center_y), start, fill=False, color='r')
        circle2 = plt.Circle((center_x, center_y), end, fill=False, color='r')
        ax1.add_artist(circle1)
        ax1.add_artist(circle2)
    return count, range_mask


from scipy import integrate
def backmodel_denoising(data_nodisk, distances, intensities, distances_te_1, intensity_te_1, r, plot=False):
    noise = []
    if plot==True:
        fig1, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        fig2, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.plot(distances, intensities, 'b-', label='Background noise Data')
    else:
        ax1=''
    mask_sum = np.zeros(data_nodisk.shape, dtype=bool)
    for distance in distances_te_1:
        start = distance - r
        end = distance + r
        mask = (distances >= start) & (distances < end)
        x_range = distances[mask]
        y_range = intensities[mask]
        if plot==True:
            ax2.fill_between(x_range, y_range, alpha=0.3, color='red', label='Integration area')
            ax2.axvline(x=start+1, color='g', linestyle='--', alpha=0.5)
            ax2.axvline(x=end, color='g', linestyle='--', alpha=0.5)
        # Calculate the integral
        count, range_mask = count_nonzero_in_range(data_nodisk, start, end, plot=plot, ax1=ax1)
        mask_sum = mask_sum + range_mask
        #print(np.sum(data_nodisk[range_mask] == 0), count)
        '''
        integral = 0
        zero_count_sum=0
        count_sum=0
        for sub_start in np.arange(start, end, 2):
            sub_end = sub_start + 2
            range_mask = (distances >= sub_start) & (distances < sub_end)
            sub_x_range = distances[range_mask ]
            sub_y_range = intensities[range_mask ]
            count, range_mask = count_nonzero_in_range(data_nodisk, sub_start, sub_end, plot=False, ax1=ax1)
            zero_count = np.sum(data_nodisk[range_mask] == 0)
            zero_count_sum += zero_count
            count_sum += count
            #pixel_count = np.sum(range_mask)
            sub_integral = integrate.trapezoid(sub_y_range, sub_x_range)
            sub_integral = sub_integral * zero_count/count
            integral = integral + sub_integral
        print('--------',zero_count_sum,'-------',count_sum)
        '''
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x**2 + y**2 <= r**2
        pixel_count = np.sum(mask)
        integral = integrate.trapezoid(y_range, x_range)
        integral = integral*pixel_count/count
        
        noise.append(integral)
    if plot==True:
        im = ax1.imshow(np.sqrt(data_nodisk*mask_sum))
        plt.colorbar(im, ax=ax1)
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), 
                   prop={'family': 'Arial', 'weight': 'bold', 'size': 12})
        plt.figure(fig1.number)
        plt.show()
        plt.figure(fig2.number)
        plt.show()
    #print(intensity_te_1)
    #print(noise)
    intensity_noneNoise = np.array(intensity_te_1) - np.array(noise)
    return intensity_noneNoise

from scipy.optimize import curve_fit
def gauss_fit(arr, num_peaks=2, peak_ranges = [(25, 40), (55, 65)], plot=True, outdir=None, savedatavalue=1):
    '''
    Fit Gaussian curves to the data.

    Parameters
    ----------
    arr : array-like
        Input data array.
    initial_guess : list, optional
        Initial guess for the Gaussian parameters. Default is [700, 35, 10, 400, 58, 20].
    plot : bool, optional
        Whether to plot the data and fits. Default is True.

    Returns
    -------
    None
    '''
    def gauss(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            a = params[i]
            b = params[i+1]
            c = params[i+2]
            y += a * np.exp(-(x - b)**2 / (2 * c**2))
        return y
    def lorentz(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            A = params[i]
            x0 = params[i+1]
            w = params[i+2]
            y += (A / np.pi) * (w / ((x - x0)**2 + w**2))
            return y
    from lmfit.models import PseudoVoigtModel
    def psd_voigt(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            A = params[i]
            x0 = params[i+1]
            w = params[i+2]
            y += A * PseudoVoigtModel().func(x, amplitude=1.0, center=x0, sigma=w)
        return y
    def estimate_initial_params(x, y, num_peaks, peak_ranges):
        initial_params = []
        for i, peak_range in enumerate(peak_ranges):
            mask = (x > peak_range[0]) & (x < peak_range[1])
            x_peak = x[mask]
            y_peak = y[mask]
            
            if len(x_peak) > 0:
                height = np.max(y_peak)
                peak_pos = x_peak[np.argmax(y_peak)]
                width = (peak_range[1] - peak_range[0]) / num_peaks
                initial_params.extend([height, peak_pos, width])
        return initial_params
    
    x = arr[0]
    y = arr[1]
    x_range = peak_ranges[0]
    mask = (x > x_range[0]) & (x < x_range[1])
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    x_range = peak_ranges[1]
    mask1 = (x > x_range[0]) & (x < x_range[1])
    x_filtered1 = x[mask1]
    y_filtered1 = y[mask1]
    
    initial_guess = estimate_initial_params(x, y, num_peaks, peak_ranges)
    
    popt1, pcov = curve_fit(gauss, x_filtered, y_filtered, p0=initial_guess[0:3], maxfev=1000000)
    popt2, pcov = curve_fit(gauss, x_filtered1, y_filtered1, p0=initial_guess[3:6], maxfev=1000000)
    if plot==True:
        fit_y1 = gauss(x[15:80], *popt1)
        fit_y2 = gauss(x[15:80], *popt2)
        plt.figure(figsize=(10, 6))
        plt.plot(x[:80], y[:80], 'bo:', label='Filtered Data', linewidth=2)
        plt.plot(x[15:80], fit_y1, 'r', label='Peak1', linewidth=2)
        plt.plot(x[15:80], fit_y2, 'r', label='Peak2', linewidth=2)
        plt.title('Gauss Fit for Radial Intensity Profile in the Range 0-80')
        plt.xlabel('Distance from center (pixels)')
        plt.ylabel('Integrated Intensity')
        plt.legend()
        if savedatavalue==0:
            directory = os.path.split(outdir)[0]
            if directory:
                if not os.path.isdir(directory):
                    os.makedirs(directory, exist_ok=True)
            plt.savefig(outdir, dpi=300)
            transposed_arr = [list(row) for row in zip(*arr)]
            saveData(f'{directory}/profile-data.dat', transposed_arr, overwrite=True)
            arr_fit = [x[15:80], fit_y1, fit_y2]
            transposed_arr_fit = [list(row) for row in zip(*arr_fit)]
            saveData(f'{directory}/gaussfit.dat',transposed_arr_fit, overwrite=True)
        plt.show()
    popt = np.concatenate((popt1, popt2))
    return popt
    
def detAng(ref_ctr,ctr,r,num=5): # threshold: accepted angle difference
    """
    Detect an angle to rotate the disk coordinates.

    Parameters
    ----------
    ref_ctr : 2D array of float
        Array of disk position coordinates and their corresponding weights
    ctr : 1D array of float
        Center of the zero-order disk.
    r : float
        Radius of the disks.

    Returns
    -------
    wt_ang : float
        The rotation angle.
    ref_ctr : 2D array of float
        Refined disk positions.

    """
    diff = ref_ctr[:,:2]-ctr
    distance = diff[:,0]**2 + diff[:,1]**2
    ctr_idx = np.where(distance==distance.min())[0][0]
    
    dis_copy = copy.deepcopy(distance)
    min_dis = []
    while len(min_dis) <num:
        cur_min = dis_copy.min()
        idx_rem = np.where(dis_copy==cur_min)[0]
        dis_copy = np.delete(dis_copy,idx_rem)
        idx_ctr = np.where(distance==cur_min)[0]
        if len(idx_ctr)==1:
            min_dis.append(ref_ctr[idx_ctr[0],:2])
        else:
            for each in idx_ctr:   
                min_dis.append(ref_ctr[each,:2])
    min_dis_ctr = np.array(min_dis,dtype = int)
    min_dis_ctr = np.delete(min_dis_ctr,0,axis = 0) # delete [0,0]
    vec = min_dis_ctr-ctr
    ang = np.arctan2(vec[:,0],vec[:,1])* 180 / np.pi
    for i in range (len(ang)):
        ang[i] = (360 + ang[i]) if (ang[i]<0) else ang[i]

    sup_pt = min_dis_ctr[np.where(ang==ang.min())[0]] # the point retuning the smallest rotation angle
    ref_diff = ctr-sup_pt
    ini_ang = np.arctan2(ref_diff[:,0],ref_diff[:,1])*180/np.pi
    all_ref = []
    for n in range (len(ini_ang)):
        all_ref.append(np.array([ini_ang[n]]))
    if len(ref_diff)>1:
        ref_diff = ref_diff[0]
    
    for each_ctr in ref_ctr:
        cur_vec = each_ctr[:2] - ref_diff
        cur_diff = ref_ctr[:,:2]-cur_vec
        cur_norm = np.linalg.norm(cur_diff,axis=1)
        if cur_norm.min()<r:
            ref_idx = np.where(cur_norm==cur_norm.min())[0]
            ref_pt = ref_ctr[ref_idx]
            ref_vec = ref_pt - each_ctr
            all_ref.append(np.arctan2(ref_vec[:,0],ref_vec[:,1])* 180 / np.pi)
    for i in range (len(all_ref)):
        if all_ref[i]<0:
            all_ref[i] = 180 + all_ref[i]
        elif all_ref[i] >= 180:
            all_ref[i] = 180 - all_ref[i]
    wt_ang = np.mean(all_ref)
    ref_ctr[ctr_idx,2] = 10**38
    
    return wt_ang, ref_ctr

def rotate_image(image, angle, changedata=True):
    '-------data grey value range-----'
    if changedata==True:
        result = np.log(image + 1)
        result = (result/np.max(result)*255).astype(np.uint8)
    else:
        result = image
    (h, w) = result.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    rotated = cv2.warpAffine(result, M, (w, h))
    return rotated

def rotImg(image, angle, ctr):
    """
    Rotate a pattern.

    Parameters
    ----------
    image : 2D array of int or float
        The input pattern.
    angle : float
        An angel to rotate.
    ctr : 1D array of int or float
        The rotation center.

    Returns
    -------
    result : 2D array of int or float
        The rotated pattern.

    """
    image_center = tuple(np.array([ctr[0],ctr[1]]))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result


def rotCtr(pattern,ref_ctr,angle):
    """
    Rotate disk coordinates.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ref_ctr : 2D array of float
        Array of the detected disk positions.
    angle : float
        Detected angle to rotate.

    Returns
    -------
    ctr_new : 2D array of float
        The transformed disk positions.

    """
    h,w = pattern.shape
    ctr_idx = np.where(ref_ctr[:,2]==ref_ctr[:,2].max())[0][0]
    ctr = ref_ctr[ctr_idx]
    ctr_new = []
    ang_rad = angle*np.pi/180

    for i in range (len(ref_ctr)):
        cur_cd = ref_ctr[i,:2]
        y_new = -(ctr[0] - (cur_cd[0]-ctr[0])*np.cos(ang_rad) + (cur_cd[1]-ctr[1])*np.sin(ang_rad) ) + 2*ctr[0]
        x_new = (ctr[1] + (cur_cd[0]-ctr[0])*np.sin(ang_rad) + (cur_cd[1]-ctr[1])*np.cos(ang_rad) )
        
        if y_new>0 and x_new>0 and y_new<h and x_new<w:
            ctr_new.append([y_new,x_new,ref_ctr[i,2]])
    
    ctr_new = np.array(ctr_new)    

    return ctr_new



def groupY (load_ctr,r):
    """
    Group disks based on their row coordinates.    

    Parameters
    ----------
    load_ctr : 2D array of float
        Array of disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    g_y : a list of arrays of float
        A list with each element as a group of disk positions.

    """
    n = len(load_ctr)
    
    g_y = [[load_ctr[0,:]]]
    for i in range (1,n):        
        gy_mean = []
        for group in g_y:
            cur_mean = 0
            grp_len = len(group)
            for each in group:
                cur_mean += each[0]
            apd_mean = cur_mean/grp_len
            gy_mean.append(apd_mean)
        
        diffy = [np.abs(s-load_ctr[i,0]) for s in gy_mean]
        gy_ind = np.argmin(diffy) 
        min_diffy = np.min(diffy)
        if min_diffy>r:
            g_y.append([load_ctr[i]])
        else:
            g_y[gy_ind].append(load_ctr[i])

    return g_y



def latFit(pattern,rot_ref_ctr,r):  
    """
    Lattice fitting process.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    rot_ref_ctr : 2D array of float
        Array of the disks positionss.
    r : float
        Radius of the disks.

    Returns
    -------
    vec_a : 1D array of float
        The estimated horizontal lattice vector [y component, x component].
    vec_b_ref : 1D array of float
        The estimated non-horizontal lattice vector [y component, x component].
    result_ctr : 2D array of float
        Array of the refined disk positions.
    lat_ctr_arr : 2D array of float
        The array of the positions of disks in the middle row.
    avg_ref_ang : float
        Refined rotation angle.

    """ 
    load_ctr = rot_ref_ctr*1
    g_y = groupY(load_ctr,r)
    
    vec_a = np.array([0,0])
    vec_b_ref = np.array([0,0])
    
    result_ctr = copy.deepcopy(rot_ref_ctr)
    lat_ctr = []
    avg_ref_ang = 0
    
    ########## Sort y values in each group and refine the angle ##########
    ref_ang = []
    for ea_g in g_y:
        if len(ea_g)>1:
            ea_g_arr = np.array(ea_g)
        
            result = np.polyfit(ea_g_arr[:,1], ea_g_arr[:,0], 1)
            ref_ang.append(np.arctan2(result[0],1)* 180 / np.pi)
    
    if len(ref_ang)>0:
        avg_ref_ang =  sum(ref_ang)/len(ref_ang) 
    else:
        avg_ref_ang = 0
        
    rot_ref_ctr2 = rotCtr(pattern,load_ctr,avg_ref_ang)
    
    g_y = groupY(rot_ref_ctr2,r)

    g_y_len = [len(l) for l in g_y]
    
    if max(g_y_len)>1:
        ################ Refine y values #######################            
        n = len(rot_ref_ctr2)
        ref_y = []
        for group in g_y:
            cur_mean = 0
            sum_cur = 0
            for each in group:
                sum_cur += each[2]
            for each in group:
                cur_mean += each[0]*(each[2]/sum_cur)
            ref_y.append(cur_mean) # Weighted mean     
            
        # Change y values to the averaged y in each group    
        result_ctr = copy.deepcopy(rot_ref_ctr2)
        for j in range (n):
            cur_y = rot_ref_ctr2[j,0]
            d_y = [np.abs(s-cur_y) for s in ref_y]
            min_y_ind = np.argmin(d_y)
            result_ctr[j][0] = ref_y[min_y_ind]     
        
        ################ Vec a #######################    
        x_g = []    
        tit_diff_x = []  
        for cur_y in ref_y:
            cur_x_g = result_ctr[np.where(result_ctr[:,0]== cur_y)]
            if len(cur_x_g)>1:
                cur_x_g.sort(axis = 0)
                x_g.append(cur_x_g)
                cur_diff_x = cur_x_g[1:]-cur_x_g[:-1]
                tit_diff_x.append(cur_diff_x)
            else:
                x_g.append(cur_x_g)   
        
        ###################### Calculate average distance ################
        if len(tit_diff_x)>0:
            outl_rem_x = []
            mean_diff_x = []
            
            for i in range (len(tit_diff_x)):
                for x in tit_diff_x[i]:
                    outl_rem_x.append(x[1])
                    
            outl_rem_x = np.array(outl_rem_x)
            q1, q3= np.percentile(outl_rem_x,[25,75])
            lower_bound = 2.5*q1 - 1.5*q3
            upper_bound = 2.5*q3 - 1.5*q1
            
            for each_g in tit_diff_x:
                each_g_mod = each_g*1
                for idx in range (len(each_g)):
                    if each_g[idx,1]<lower_bound or each_g[idx,1]>upper_bound:
                        each_g_mod = np.delete(each_g,idx,axis = 0)               
                
                if len(each_g_mod)>0:
                    cur_mean = np.mean(each_g_mod[:,1],axis=0)
                    mean_diff_x.append([cur_mean,len(each_g_mod)])
                
            mean_diff_x_arr = np.array(mean_diff_x)
            
            if len(mean_diff_x_arr)>0:
                count = 0 
                sum_x = 0
                for i in range (len(mean_diff_x_arr)):
                    sum_x += mean_diff_x_arr[i,0]* mean_diff_x_arr[i,1]
                    count += mean_diff_x_arr[i,1]
                
                vec_a = np.array([0, sum_x/count])
                
                ######### Find vector b #########
                set_ct_ind = np.argmax(result_ctr[:,2])
                set_ct = result_ctr[set_ct_ind]
                
                # Find rough b
                min_nn = 10**38
                nn_vecb_rough = np.array([-1,-1,-1])
                for gn in range (len(x_g)):
                    cur_ct = x_g[gn]
                    if set_ct[0] not in cur_ct[:,0]:
                        dis_xy = cur_ct - set_ct
                        dis_norm = np.linalg.norm(dis_xy[:,:2],axis = 1)
                        xy_min = np.min(dis_norm)
                        if xy_min<=min_nn:  
                            min_nn = xy_min 
                            nn_vecb_rough = cur_ct[np.argmin(dis_norm)]   
                
                # Generate hypothetical lattice
                h,w = pattern.shape 
                lat_ctr = [set_ct[:2]]
                
                ###### Generate pts along vector a (middle row) ######
                # one side    
                cur_h1 = set_ct[0]
                cur_w1 = set_ct[1]
                cur_ct1 = set_ct[:2]*1
                while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                        cur_h1,cur_w1 = cur_ct1-vec_a
                        if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                            cur_ct1 = [cur_h1,cur_w1]
                            lat_ctr.append([cur_h1,cur_w1])
                
                # the other side
                cur_h2 = set_ct[0]
                cur_w2 = set_ct[1]
                cur_ct2 = set_ct[:2]*1.0
                while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                    cur_h2,cur_w2 = cur_ct2+vec_a
                    if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                        cur_ct2 = [cur_h2,cur_w2]
                        lat_ctr.append([cur_h2,cur_w2])                            
                        
                ######### Refine Vector b #########
                vec_b = nn_vecb_rough - set_ct
                if  vec_b[0]<0:
                    vec_b = -vec_b
            
                vec_b_rough = vec_b [:2]
            
                diff_y_ref = []   

                look_y = set_ct[0]-vec_b_rough[0]
                est_ct = lat_ctr - vec_b_rough 
                while look_y>0:
                    for each in est_ct:
                        each_diff_xy = each - result_ctr[:,:2]
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)
                    look_y -= vec_b_rough[0]
                    est_ct -= vec_b_rough
        
                look_y = set_ct[0]+vec_b_rough[0]
                est_ct = lat_ctr + vec_b_rough
                while look_y<h:
                    for each in est_ct:
                        each_diff_xy = result_ctr[:,:2] - each
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
        
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)   
                    look_y += vec_b_rough[0]
                    est_ct += vec_b_rough      
                 
                vec_b_ref = vec_b_rough*1.0
                if len(diff_y_ref)==0:
                    diff_y_ref.append([0,0])
                diff_y_ref = np.array(diff_y_ref)        
                vec_b_ref[1] = vec_b_ref[1] + np.mean(diff_y_ref[:,1])
    
    lat_ctr_arr = np.array(lat_ctr)
    return vec_a, vec_b_ref, result_ctr, lat_ctr_arr, avg_ref_ang



# Generate 2d lattice based on vector a and b
def genLat(pattern, ret_a,ret_b, mid_ctr,r):
    """
    Generate a matrix of hypothetical lattice points.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ret_a : 1D array of float
        The horizontal lattice vector a.
    ret_b : 1D array of float
        The non-horizontal lattice vector b.
    mid_ctr : a list of arrays of float
        a list of disk positions which are in the middle row.
    r : float
        Radius of the disks.

    Returns
    -------
    final_ctr : 2D array of float
        Disk positions in the hypothetical lattice.

    """
    img = pattern
    veca,vecb = ret_a,ret_b
    h,w = img.shape
    veca_ct = mid_ctr[:,:2].copy()
    final_ctr = []
    
    for cur_veca_ct in veca_ct:
        # one side    
        cur_h1 = cur_veca_ct[0]
        cur_w1 = cur_veca_ct[1]
        cur_ct1 = cur_veca_ct*1

        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
            cur_h1,cur_w1 = cur_ct1-vecb
            if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_ct1 = [cur_h1,cur_w1]
                final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_veca_ct[0]
        cur_w2 = cur_veca_ct[1]
        cur_ct2 = cur_veca_ct*1

        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+vecb
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]
                final_ctr.append([cur_h2,cur_w2])  

    ########   Check Again ########
    chk_lat_ctr= final_ctr
    
    for cur_vec2_ct in chk_lat_ctr:
        # one side    
        cur_h1 = cur_vec2_ct[0]
        cur_w1 = cur_vec2_ct[1]
        cur_ct1 = cur_vec2_ct*1
        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_h1,cur_w1 = cur_ct1-veca
                # print(cur_ct1-veca,cur_h1,cur_w1)
                if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                    cur_ct1 = [cur_h1,cur_w1]
                    dif_chk = [(ct[0]-cur_ct1[0])**2+(ct[1]-cur_ct1[1])**2 for ct in chk_lat_ctr]
                    if min(dif_chk)> r**2: 
                        final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_vec2_ct[0]
        cur_w2 = cur_vec2_ct[1]
        cur_ct2 = cur_vec2_ct*1
        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+veca
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]   
                dif_chk2 = [(ct[0]-cur_ct2[0])**2+(ct[1]-cur_ct2[1])**2 for ct in chk_lat_ctr]
                if min(dif_chk2)> r**2:  
                    final_ctr.append([cur_h2,cur_w2])   
                                 
    for pt in mid_ctr:
        final_ctr.append(pt)
    
    final_ctr = np.array(final_ctr)
                    
    return final_ctr



def delArti(gen_lat_pt,ref_ctr,r):
    """
    Delete any artificial lattice points.

    Parameters
    ----------
    gen_lat_pt : 2D array of float
        Array of artificial disk positions.
    ref_ctr : 2D array of float
        Array of detected disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    gen_lat_pt_up : 2D array of float
        A filtered array of disk positions.

    """
    gen_lat_pt_up = []
    for i in range (len(gen_lat_pt)):
        dif_gen_ref = np.array(gen_lat_pt[i] - ref_ctr[:,:2])
        dif_gen_ref_norm = np.linalg.norm(dif_gen_ref,axis = 1)
        if dif_gen_ref_norm.min()< r:
            gen_lat_pt_up.append(gen_lat_pt[i])
    
    gen_lat_pt_up = np.array(gen_lat_pt_up)
    
    return gen_lat_pt_up



def latBack(refe_a,refe_b,angle):
    """
    Transform the lattice vectors to the default coordinate system.

    Parameters
    ----------
    refe_a : 1D array of float
        Array of the vector a.
    refe_b : 1D array of float
        Array of the vector b.
    angle : float
        The rotation angle.

    Returns
    -------
    a_init : 1D array of float
        Transformed array of the vector a.
    b_init : 1D array of float
        Transformed array of the vector b.

    """
    ang_init_back = angle*np.pi/180
    a_init = np.array([refe_a[1]*np.sin(ang_init_back),refe_a[1]*np.cos(ang_init_back)])
    b_init = np.array([refe_b[1]*np.sin(ang_init_back)+refe_b[0]*np.cos(ang_init_back),refe_b[1]*np.cos(ang_init_back)-refe_b[0]*np.sin(ang_init_back)])
    
    return a_init,b_init

def vector_select(a_back,b_back,num=7,plot=False):
    '''
    Selects the top 2 vectors based on the smallest angles with the positive x-axis.

    Parameters
    ----------
    a_back : numpy array
        The first base vector.
    b_back : numpy array
        The second base vector.
    num : int, optional
        Number of vectors to consider. The default is 7.
    plot : bool, optional
        Whether to plot the vectors. The default is False.

    Returns
    -------
    numpy array
        The first selected vector.
    numpy array
        The second selected vector.
    '''
    # Calculate the norms (lengths) of the two base vectors
    mod_a = np.linalg.norm(a_back)
    mod_b = np.linalg.norm(b_back)
    radius = 2 * max(mod_a, mod_b)
    coords = []
    # Determine the scanning range (using integers for example, adjust precision as needed)
    range_factor = int(max(radius // mod_a, radius // mod_b)) + 1
    # Linear combinations to find all possible vectors
    for i in range(-range_factor, range_factor + 1):
        for j in range(-range_factor, range_factor + 1):
            # Linear combination
            vec = i * a_back + j * b_back
            # Check if within the radius
            if np.linalg.norm(vec) <= radius:
                coords.append(vec)
    coords = np.array(coords)
    nearest_6_coords = coords[np.argsort(np.linalg.norm(coords, axis=1))][1:num]
    # Calculate the angles of these 6 coordinates with respect to the positive x-axis and normalize to the range [0, 2Pi]
    angles = np.arctan2(nearest_6_coords[:,0], nearest_6_coords[:,1])
    angles = (angles + 2 * np.pi) % (2 * np.pi) # Adjust angles to be in the range [0, 2Pi]
    sorted_coords_by_angle = nearest_6_coords[np.argsort(angles)]
    top_2_coords = sorted_coords_by_angle[:2]
    if plot==True:
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 1], coords[:, 0], color='blue', marker='o', label='Vectors')
        plt.scatter(top_2_coords[0, 1], top_2_coords[0, 0], color='red',marker='x', label='Top 2 Vectors', s=100)
        plt.scatter(top_2_coords[1, 1], top_2_coords[1, 0], color='green',marker='x', label='Top 2 Vectors', s=100)
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.legend()
        plt.xlim([-radius, radius])
        plt.ylim([radius, -radius])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    return top_2_coords[0], top_2_coords[1]

def drawCircles(ori_pattern,blobs_list,r,lwide=2,outdir=None,title=None,xylim=None,text=False, points=None, sort=False):
    """
    Label the disk positions on the pattern.

    Parameters
    ----------
    ori_pattern : 2D array of int or float
        The pattern to be labeled on.
    blobs_list : 2D array of float
        Array of disk positions.
    r: float
        The radius of the disks.
    outdir: str
        

    Returns
    -------
    None.

    """
    pattern = copy.deepcopy(ori_pattern)
    
    for q in range (len(blobs_list)):
        center = (int(blobs_list[q][0]),int(blobs_list[q][1]))
        pattern[center] = pattern.max()
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(pattern,cmap='gray')
    #ax.imshow(pattern,cmap='viridis')
    for q, blob in enumerate(blobs_list):
        y, x = blob
        c = plt.Circle((x, y),r, color='red', linewidth=lwide, fill=False)
        ax.add_patch(c)
        if text==True:
            #ax.text(x, y, f'({y:.2f}, {x:.2f})', color='green', fontsize=4, ha='center', va='center', fontweight='bold')
            ax.text(x, y, f'({y:.0f}, {x:.0f})', color='black', fontsize=4, ha='center', va='center', fontweight='bold')
        if sort==True:
            ax.text(x, y-10, f'{q}', color='red', fontsize=8, ha='center', va='center', fontweight='bold')
        plt.title(title)
        plt.axis('off')
    if xylim is not None:
        plt.xlim(xylim[0], xylim[1])
        plt.ylim(xylim[2], xylim[3])
    if points is not None:
        point1 = points[0] + points[2]
        point2 = points[1] + points[2]
        ax.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-')
        ax.annotate('', xy=(point1[1], point1[0]), xytext=(points[2][1], points[2][0]),
                    arrowprops=dict(arrowstyle="->", color='g', lw=2))
        ax.annotate('', xy=(point2[1], point2[0]), xytext=(points[2][1], points[2][0]),
                    arrowprops=dict(arrowstyle="->", color='r', lw=2))
    plt.show()
    if outdir is not None:
        directory = os.path.split(outdir)[0]
        if directory:
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
        fig.savefig(outdir,format='tif',dpi=300)
    pass



def latDist(lat_par,refe_a,refe_b,err=0.2):
    """
    This function filters out the outliers of the lattice parameters based on the references.

    Parameters
    ----------
    lat_par : 2D array of arrays of float
        2D array with each element as two arrays of lattice vectors.
    refe_a : 1D array of float
        The reference lattice vector a.
    refe_b : 1D array of float
        The reference lattice vector b.
    err : float, optional
        Acceptable error percentage. The default is 0.2 (20%).

    Returns
    -------
    store_whole : 3D array of float
        Array containing 3 columns, y coordinate, x coordinate, and 4 lattice vector elements
        (y of vector a, x of vector a, y of vector b, x of vector b).

    """
    arr_vec = lat_par
    
    sm_y,sm_x = lat_par.shape[:2]
    std_ax = refe_a[1] # vec_a[0,std_2x]
    std_ay = refe_a[0]
    std_bx = refe_b[1] # vec_b[std_1y,std_1x]
    std_by = refe_b[0]
    
    acc_ax_min = std_ax*(1-err) if std_ax>0 else std_ax*(1+err)
    acc_ax_max = std_ax*(1+err) if std_ax>0 else std_ax*(1-err)
    acc_ay_min = std_ay*(1-err) if std_ay>0 else std_ay*(1+err)
    acc_ay_max = std_ay*(1+err) if std_ay>0 else std_ay*(1-err)
    acc_bx_min = std_bx*(1-err) if std_bx>0 else std_bx*(1+err)
    acc_bx_max = std_bx*(1+err) if std_bx>0 else std_bx*(1-err)
    acc_by_min = std_by*(1-err) if std_by>0 else std_by*(1+err)
    acc_by_max = std_by*(1+err) if std_by>0 else std_by*(1-err)
    
    store_whole = np.zeros((sm_y,sm_x,4),dtype = float)

    # Delete paramater outliers
    ct = 0
    for row in range (sm_y):
        for col in range (sm_x):
            
            each = arr_vec[row,col]
        
            gax = float(each[0,1])
            gay = float(each[0,0])
            gbx = float(each[1,1])  
            gby = float(each[1,0])
            
            if gax>acc_ax_max or gax<acc_ax_min or gay>acc_ay_max or gay<acc_ay_min or gbx>acc_bx_max or gbx<acc_bx_min or gby>acc_by_max or gby<acc_by_min:
                ct += 1
    
            else:
                store_whole[row,col][0] = gay        
                store_whole[row,col][1] = gax
                store_whole[row,col][2] = gby
                store_whole[row,col][3] = gbx                

    return store_whole       



def calcStrain(lat_fil, refe_a,refe_b):
    """
    Compute strain maps.
    
    Parameters
    ----------
    lat_fil : 2D array of arrays of float
        2D array with each element as two lattice vectors.
    refe_a : 1D array of float 
        The reference vector a.
    refe_b : 1D array of float
        The reference vector b.

    Returns
    -------
    st_xx : 2D array of float
        Estimated strain along the x direction.
    st_yy : 2D array of float
        Estimated strain along the y direction.
    st_xy : 2D array of float
        Shear strain.
    st_yx : 2D array of float
        Shear strain.
    tha_ang : 2D array of float
        Angle of lattice rotation in deg.

    """
    sm_y,sm_x = lat_fil.shape[:2]
    
    st_xx = np.zeros((sm_y,sm_x),dtype=float)
    st_yx = np.zeros((sm_y,sm_x),dtype=float)
    st_xy = np.zeros((sm_y,sm_x),dtype=float)
    st_yy = np.zeros((sm_y,sm_x),dtype=float)
    tha_ang = np.zeros((sm_y,sm_x),dtype=float)
    
    G0_T = np.array([[refe_a[1],refe_a[0]],[refe_b[1],refe_b[0]]])
    
    for row in range (sm_y):
        for col in range (sm_x):
            if any(lat_fil[row,col]!=0):
                gay,gax,gby,gbx = lat_fil[row,col]
    
                G = np.array([[gax,gbx],[gay,gby]])
                G_T = np.transpose(G)
                G_T_n1 = np.linalg.inv(G_T)
                
                D = G_T_n1.dot(G0_T)
                theta = np.arctan2((D[1,0]-D[0,1]),(D[0,0]+D[1,1]))
                
                M = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
                
                F = M.dot(D)
                I = np.array([[1,0],[0,1]])
                
                eps = F-I
                
                st_xx[row,col] = eps[0,0]
                st_yy[row,col] = eps[1,1] 
                st_xy[row,col] = eps[0,1]
                st_yx[row,col] = eps[1,0] 
                tha_ang[row,col] = theta/np.pi*180

    return st_xx,st_yy,st_xy,st_yx,tha_ang


def detect_ellipse(data, gauss_kernel=(11,11), low_threshold=10, high_threshold=30, 
                   kernel_size=7, dilation_iterations=1, erosion_iterations=1, 
                   ellipses_size=20, plot=True):
    '''
    Detects ellipses in an image.

    Parameters:
    ----------
    data : numpy.ndarray
        Input image data.
    gauss_kernel : tuple, optional
        Gaussian blur kernel size. Default is (11,11).
    low_threshold : int, optional
        Low threshold for edge detection. Default is 10.
    high_threshold : int, optional
        High threshold for edge detection. Default is 30.
    kernel_size : int, optional
        Kernel size for morphological operations. Default is 7.
    dilation_iterations : int, optional
        Number of dilation iterations. Default is 1.
    erosion_iterations : int, optional
        Number of erosion iterations. Default is 1.
    ellipses_size : int, optional
        Minimum size of contours to fit ellipses. Default is 20.
    plot : bool, optional
        Whether to plot intermediate results. Default is True.

    Returns:
    ----------
    ellipses : list
        List of detected ellipses.
    '''
    # 1. Gaussian Blur
    data_uint8 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #gauss_kernel = (11,11)  # Controls the degree of Gaussian blur, odd numbers, larger for more blur
    blur = cv2.GaussianBlur(data_uint8, gauss_kernel, 0)
    if plot==True:
        plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)) # Convert color channels from BGR to RGB
        plt.title('Gauss Blur Result')
        plt.axis('off') # Turn off axis display
        plt.show()
    # 2. Edge Detection
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    if plot==True:
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)) # Convert color channels from BGR to RGB
        plt.title('Edge Detection Result')
        plt.axis('off')# Turn off axis display
        plt.show()
        #cv2.imwrite(f'{folder}/analysis/edge_detection_result.png', edges)  # Save edge detection result to file
    # 3. Morphological Operations
    #kernel_size = 7  # Define kernel size, initially set to 3x3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    eroded = cv2.erode(dilated, kernel, iterations=erosion_iterations)
    if plot==True:
        plt.imshow(cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB)) # Convert color channels from BGR to RGB
        plt.title('Morphologically Processed Edges')
        plt.axis('off')
        plt.show()
        #cv2.imwrite(f'{folder}/analysis/Morphologically_Processed_Edges.png', eroded)  # Save morphological operation result to file
    # 4. Find Contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    ellipses = []   
    for cnt in contours:
        if len(cnt) > ellipses_size:  # Ensure contours are large enough
            ellipse = cv2.fitEllipse(cnt)  # Fit ellipses to the found contours
            ellipses.append(ellipse)
    if not ellipses:
        ellipses = [[(0, 0), (0, 0), 0]]
    if plot==True:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(np.sqrt(data), cmap='gray')
        (x_center, y_center), (minor_axis, major_axis), angle = ellipses[-1]
        e = Ellipse(xy=(x_center, y_center), width=minor_axis, height=major_axis, angle=angle,
                    edgecolor='red', lw=1, facecolor='none')
        ax.add_patch(e)
        plt.title('Detected Ellipses')
        plt.axis('off')
        plt.show()
        #cv2.imwrite(f'{folder}/analysis/Detected_Ellipses.png', data)  # Save contour detection result to file
    return ellipses[-1]

def generateMask(ellipses, outer=5, inter=20, plot=True):
    '''
    Generates an elliptical mask image.

    Parameters
    ----------
    ellipses : tuple
        Tuple containing the center coordinates, axes lengths, and angle of the ellipse.
    outer : int, optional
        Thickness of the outer ellipse boundary. Default is 5.
    inter : int, optional
        Thickness of the inner ellipse boundary. Default is 20.
    plot : bool, optional
        Whether to display the generated mask image. Default is True.

    Returns
    -------
    numpy.ndarray
        The generated elliptical mask image.
    '''
    (center_float, axes_float, angle) = ellipses
    image = np.zeros((256, 256), dtype=np.uint8)
    center = (int(round(center_float[0])), int(round(center_float[1])))
    axes = (int(round(axes_float[0] / 2)), int(round(axes_float[1] / 2)))
    outer_axes = (int(round(axes[0] + outer)), int(round(axes[1] + outer)))
    inner_axes = (int(round(axes[0] - inter)), int(round(axes[1] - inter)))
    cv2.ellipse(image, center, outer_axes, angle, 0, 360, 255, -1)
    if inner_axes[0] > 0 and inner_axes[1] > 0:
        cv2.ellipse(image, center, inner_axes, angle, 0, 360, 0, -1)
    else:
        raise ValueError("Inner ellipse axes must be positive")
    if plot==True:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('Mask')
        plt.show()
    return image

from scipy import ndimage
def generate_r_mass(data, center, angle, plot=True):
    '''
    Calculates the distance between the center of mass and a specified center for a given angle.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    center : tuple
        Tuple containing the coordinates of the center.
    angle : float
        Angle in degrees.
    plot : bool, optional
        Whether to display the visualization. Default is True.

    Returns
    -------
    float
        Distance between the center of mass and the specified center.
    '''
    y, x = np.indices((data.shape))
    y = center[1] - y
    x = x - center[0]
    angles = np.rad2deg(np.arctan2(y, x)) % 360
    distances = []
    com = []
    angle_threshold = 1.5
    for angle_num in [angle,angle+90]:
        angle1 = angle_num % 360
        angle2 = (angle_num + 180) % 360
        mask1 = (angles >= (angle1 - angle_threshold)) & (angles <= (angle1 + angle_threshold))
        mask2 = (angles >= (angle2 - angle_threshold)) & (angles <= (angle2 + angle_threshold))
        com1 = ndimage.center_of_mass(data, labels=mask1, index=1)
        com2 = ndimage.center_of_mass(data, labels=mask2, index=1)
        distance = np.sqrt((com1[0]-com2[0])**2+(com1[1]-com2[1])**2)
        com.extend([com1, com2])
        distances.append(distance)
    if plot==True:
        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap='gray')
        #plt.contour(mask1, [0.5], colors='r')
        plt.plot(center[0], center[1], 'yo')  # plot the center point
        for point in com:
            plt.plot(point[1], point[0], 'bo')  # Plot each point as a red circle
        plt.title(f'Center of Mass for Angle {angle} degrees')
        #plt.axis('off')
        plt.show()
    return distances

def array_average(data, mask):
    '''
    Calculate the average of non-zero elements in the data array after applying the mask.

    Parameters
    ----------
    data : numpy array
        Input data array.
    mask : numpy array
        Mask array to apply to the data.

    Returns
    -------
    numpy array, float
        Normalized data array and the average of non-zero elements.
    '''
    new_data = data*mask
    average_x_nonzero = np.mean(new_data[new_data != 0])
    #variance_x_nonzero = np.var(new_data[new_data != 0])
    sample_std_dev = np.std(new_data[new_data != 0], ddof=1)
    new_data[new_data == 0] = average_x_nonzero
    new_data = (new_data-average_x_nonzero)/average_x_nonzero
    return new_data, average_x_nonzero, sample_std_dev

def histogram_analysis(arr1, bin_width=0.01, lim=[0,0.5,0,80,200], plot=True, plot_fit=True, out_dir=None):
    arr1_flattened = arr1.flatten()
    arr1_flattened = arr1_flattened[arr1_flattened != 0]  # 排除0
    min_val = arr1_flattened.min()
    max_val = arr1_flattened.max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    hist, edges = np.histogram(arr1_flattened, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    if out_dir is not None:
        directory = os.path.split(out_dir)[0]
        if directory:
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
    with open(f'{out_dir}.txt', mode='w') as file:
        file.write('Bin Center\tCounts\n')
        for center, frequency in zip(bin_centers, hist):
            file.write(f"{center:.3f}\t{frequency}\n")
    #print(f"data has saved to {out_dir}")
    
    '-----plot line fit-----'
    def poly_func(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    mask = bin_centers!= 0
    bin_centers = bin_centers[mask]
    hist = hist[mask]
    if plot_fit==True:
        params, covariance = curve_fit(poly_func, bin_centers, hist, maxfev=5000)
    else:
        params = [1, 1, 1]
    min_fit = lim[0]
    max_fit = lim[1]
    x_fit = np.linspace(min_fit, max_fit, 1000)
    y_fit = poly_func(x_fit, *params)
    
    if plot==True:
        plt.figure(figsize=(8, 6))
        plt.hist(arr1_flattened, bins=bins, color=(6/255, 48/255, 98/255), edgecolor='black', alpha=0.8)
        if plot_fit==True:
            #plt.scatter(bin_centers, hist, color='red', label='Data points', alpha=0.7)
            plt.plot(x_fit, y_fit, color='red', label='Fitted curve', linewidth=3)
            plt.legend(prop={'family': 'Arial', 'weight': 'bold', 'size': 16}, frameon=False)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        plt.xticks(fontname='Arial', fontweight='bold', fontsize=14)
        plt.yticks(fontname='Arial', fontweight='bold', fontsize=14)
        plt.title(f'Histogram of {lim[4]}', fontname='Arial', fontweight='bold', fontsize=16)
        plt.xlabel('Value', fontname='Arial', fontweight='bold', fontsize=16)
        plt.ylabel('Counts', fontname='Arial', fontweight='bold', fontsize=16)
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        #plt.grid(True)
        if plot_fit==True:
            plt.savefig(f'{out_dir}_fit.png', dpi=300)
        else:
            plt.savefig(f'{out_dir}.png', dpi=300)
        plt.show()
    return np.array(bin_centers), np.array(hist), np.array(x_fit), np.array(y_fit)

def generate_vectors(center, point1, point2, num_vectors=5, plot=False):
    vector1 = np.array(point1) - np.array(center)
    vector2 = np.array(point2) - np.array(center)
    vectors = []
    for alpha in range(-num_vectors,num_vectors,1):
        for beta in range(-num_vectors,num_vectors,1):
            new_point = center + alpha * vector1 + beta * vector2
            vectors.append(new_point)
    vectors = np.array(vectors)
    vectors = vectors[np.argsort(np.linalg.norm(vectors - center, axis=1))]
    if plot==True:
        plt.figure(figsize=(8, 8))
        plt.xlim(0, 256)
        plt.ylim(0, 256)
        plt.grid()
        plt.plot(center[1], center[0], 'ro', label='Center (128, 128)')
        plt.plot(point1[1], point1[0], 'bo', label='Point 1 (128, 138)')
        plt.plot(point2[1], point2[0], 'go', label='Point 2 (138, 128)')
        plt.scatter(vectors[:, 1], vectors[:, 0], color='purple', label='Generated Vectors')
        plt.title('Generated Vector Points')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()
    return vectors

def cal_copy(value, X, Y, num_copy):
    matrix = np.arange(X * Y).reshape((Y, X))
    #print(matrix)
    y = value // X
    x = value % X
    num = num_copy//2
    if num_copy % 2 == 1:
        value_set = matrix[y-num:y+num+1,x-num:x+num+1]
    return value_set

from scipy.ndimage import binary_opening, binary_closing
def mask_arrange(mask_img, min_size = 15, kernel_size = 5, clean=False, close=False):
    # 方法1: 使用连通区域分析去除小区域
    # 标记连通区域
    labeled_mask, num_features = label(mask_img)
    print(f"找到 {num_features} 个连通区域")
    # 计算每个区域的大小
    region_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        region_sizes.append((i, size))
    # 创建清理后的掩码
    mask_img_cleaned = np.zeros_like(mask_img)
    for region_id, size in region_sizes:
        if size >= min_size:
            mask_img_cleaned[labeled_mask == region_id] = 1
    # 方法2: 使用形态学操作进一步清理
    # 先进行开运算去除小的噪声点
    #mask_img_cleaned = binary_opening(mask_img_cleaned, structure=np.ones((kernel_size, kernel_size)))
    # 再进行闭运算填充小的空洞
    mask_img_final = binary_closing(mask_img_cleaned, structure=np.ones((kernel_size, kernel_size)))
    # 最终清理后的掩码
    mask_img_final = mask_img_final.astype(np.float32)
    # 可视化对比
    #visual(mask_img_cleaned, title='After removing small objects')
    #visual(mask_img_final, title='After morphological operations')
    # 统计信息
    original_pixels = np.sum(mask_img)
    cleaned_pixels = np.sum(mask_img_final)
    removed_pixels = original_pixels - cleaned_pixels
    print(f"原始掩码像素数: {original_pixels}")
    print(f"清理后像素数: {cleaned_pixels}")
    print(f"移除的像素数: {removed_pixels}")
    print(f"移除比例: {removed_pixels/original_pixels*100:.2f}%")
    # 使用清理后的掩码
    mask_img = mask_img_final
    return mask_img