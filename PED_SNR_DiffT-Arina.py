%load_ext autoreload
%autoreload 2
import importlib
import AutoDisk as AD
from AutoDisk.autodisk import *
import numpy as np
from PIL import Image
import sys
import py4DSTEM
print(py4DSTEM.__version__)
from py4DSTEM.visualize import show
importlib.reload(AD)

folder = r'E:\Graphene nano-temperature mapping\arina-20260401\chips-arina'
params=[(200,37,115),(400,26,117),(600,30,108),(800,43,100)]  #[(25,63,122)]
subfolder='SNR-analysis' 
test_position=[]
for t,x0,y0 in params:
    T=f'{t}'
    out = f'{subfolder}/{T}'
    filename = f'{folder}/1mrad-temperature/7-{T}_master.h5'
    X,Y = 256,256
    dataset = py4DSTEM.import_file(filename, scan_width=X)
    data_ori = np.array(dataset.data, dtype = "float32")   #scan-hang scan-lie ED-hang ED-lie
    print('-------------',f'{T}:',data_ori.shape,'-------------')
    print(np.sum(data_ori[0,0]))
    '--------(1)extraction data---------'
    width = 40
    test_position=data_ori[y0:y0+width, x0:x0+width]  #scan-hang scan-lie ED-hang ED-lie
    test_position = AD.autodisk.data4D_cal_copy(test_position, region_size=3)
    avg_pat_test = AD.autodisk.generateAvg(test_position, normalization=True)
    AD.autodisk.saveData(f'{folder}/{out}/avg_pat.dat',avg_pat_test,dataformat='np.float32', overwrite=True)
    AD.autodisk.visual(np.sqrt(np.sqrt(avg_pat_test)),title='diffraction-pattern'
                       ,outdir=f'{folder}/{out}/diffraction-pattern.png')
    data = avg_pat_test
    '--------(2)Autodisk-----------'
    #(1)Find center
    ctr_ori = AD.autodisk.find_center_of_mass(data,threshold_value = 0.001)
    #(2)Find radius
    r = AD.autodisk.calculate_radius(data, ctr_ori, thre=0.10, plot=True)
    print('Center of the zero-order disk is {}. The radius of the disk is {}.'.format(ctr_ori,r))
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(data,cmap='gray')
    y,x = ctr_ori
    c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
    ax.add_patch(c)
    plt.show()
    #(2)Build a ring kernal for cross-correlation.
    kernel = AD.autodisk.generateKernel(data,ctr_ori,r,0.7,2)
    kernel_out = AD.autodisk.visual(kernel)
    kernel[kernel < kernel.mean()] = 0
    kernel[kernel !=0] = 1
    kernel_out = AD.autodisk.visual(kernel)
    #(3)Generate the cross-correlated pattern.By changing the thred-parameters of the ctrDet-function!
    cros_map = AD.autodisk.crossCorr(data,kernel)
    show_cros_map = AD.autodisk.visual(cros_map)
    thre=0.000004
    blobs = AD.autodisk.ctrDet(cros_map, r, kernel, 1, thre)  # 001-SiC:0.18      120-Si3N4:0.53 
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(data)),blobs,r)
    #(4)Marking the position of the diffraction spot!
    blobs_new = np.array(blobs)
    ctr = np.array(ctr_ori)
    sorted_blobs = blobs_new[np.argsort(np.linalg.norm(blobs_new - ctr, axis=1))]
    sorted_blobs=sorted_blobs+[1, 1]
    #print(sorted_blobs)
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),sorted_blobs,r,lwide=1,text=True,outdir=f'{folder}/{out}/disk.tif')
    
    #2024.11.09
    pattern = data
    disk_num = 7
    ctr = sorted_blobs[:disk_num]
    #print(ctr)
    ref_ctr = np.empty((len(ctr), 3))
    ref_ctr[:,:2] = ctr
    ref_ctr[:,2] = cal_weight(pattern,ctr,r,ra=3)
    angle, refined_ctr = detAng(ref_ctr,ctr[0],r,num=disk_num)
    rot_ref_ctr = rotCtr(pattern,ref_ctr,angle)
    ret_a,ret_b,ref_ctr2, mid_ctr,ref_ang = latFit(pattern,rot_ref_ctr,r)
    print(angle, ref_ang, angle+ref_ang)
    if any(ret_a!=0) and any(ret_b!=0):
        a_back,b_back = latBack(ret_a, ret_b, angle+ref_ang)       
        '-----vector_select-----'
        a_select,b_select = AD.autodisk.vector_select(a_back,b_back,num=disk_num,plot=True)
    point1=(ctr_ori[0]+a_select[0], ctr_ori[1]+a_select[1])
    point2=(ctr_ori[0]+b_select[0], ctr_ori[1]+b_select[1])
    sorted_blobs = AD.autodisk.generate_vectors(ctr_ori, point1, point2, num_vectors=5)

    r=r+1
    ctr_ori0 = AD.autodisk.find_center_of_mass(data,threshold_value = 0.001)
    refine=[ctr_ori0[0]-sorted_blobs[0][0],ctr_ori0[1]-sorted_blobs[0][1]]
    print(refine)
    blobs = np.array(sorted_blobs) + np.array(refine)
    blobs = blobs[:48]
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),np.round(blobs, 1),r,lwide=1.5,xylim=None,text=False,sort=True
                            ,color_groups = [1,6,6,6,12,6,6])
    selected_indices = list(range(0, 43))# + [45,46]
    #selected_indices = list(range(0, 41)) + [43,44]
    blobs = blobs[selected_indices]
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),np.round(blobs, 1),r,lwide=1.5,xylim=None,text=False,sort=True
                            ,outdir=f'{folder}/{out}/disk_select.tif', color_groups = [1,6,6,6,12,6,6])
    blobs = AD.autodisk.find_center_of_mass(data, blobs, r=7, plot=False)
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),np.round(blobs, 1),r,lwide=1.5,xylim=None,text=False,sort=False
                            ,outdir=f'{folder}/{out}/disk_select2.tif', color_groups = [1,6,6,6,12,6,6])
    print(r)
    '-------angle adjust------'
    num_disk_all = [6,6,6,12,6,6]
    center = blobs[0]
    num_order = 1
    for num_disk in num_disk_all:
        ctr_select = blobs[num_order: num_order+num_disk]
        angles = np.arctan2(ctr_select[:, 0] - center[0], ctr_select[:, 1] - center[1])
        blobs[num_order: num_order+num_disk] = ctr_select[np.argsort(angles)]
        num_order = num_order+num_disk
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),np.round(blobs, 1),r,lwide=1.5,xylim=None,text=False,sort=True
                            ,outdir=f'{folder}/{out}/disk_select3.tif', color_groups = [1,6,6,6,12,6,6])

    blobs= np.round(blobs[[2,8,14,22,32,38]])
    #sys.exit()
    n_n=0
    for blob in blobs:
        profile_all, num_mean_std_SNR_all = AD.autodisk.SNR_analysis(test_position, blob=blob, width=11, plot=False)
        #np.savetxt(f'{folder}/{out}/profile-{blob[0]}_{blob[1]}.txt', profile_all, fmt='%i', delimiter=' ')
        np.savetxt(f'{folder}/{out}/num_mean_std_SNR_all-{n_n}.txt', num_mean_std_SNR_all, fmt='%.2f', delimiter=' ')
        n_n +=1