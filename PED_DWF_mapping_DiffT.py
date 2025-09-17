%load_ext autoreload
%autoreload 2
import importlib
import AutoDisk as AD
from AutoDisk.autodisk import *
import numpy as np
from PIL import Image
importlib.reload(AD)

folder=r'E:\Graphene nano-temperature mapping\Graphene-2024-11-09\temperature'
#params=[(200,108,91)]#,(250,104,93),(300,108,97),(350,107,94),(400,115,90),(450,101,88),(500,105,96),(550,97,100),(600,108,89),(650,107,86),(700,109,85),(750,107,89),(800,109,92),(850,114,86),(900,107,99),(950,103,92),(1000,92,93)]
params=[(200,108,91),(250,105,94),(300,110,97),(350,107,94),(400,115,90),(450,101,88),(500,107,96),(550,101,99),(600,108,89),(650,108,87),(700,111,85),(750,108,88),(800,110,91),(850,114,86),(900,107,97),(950,106,89)]
#params=[(200,81,27),(250,78,30),(300,84,33),(350,81,31),(400,88,27),(450,75,24),(500,80,33),(550,74,37),(600,80,28),(650,81,26),(700,83,22),(750,82,25),(800,84,29),(850,87,23),(900,82,34),(950,80,25)]
subfolder = 'DWF_mapping_area1_5x5average_40_r2-2_new_autodisk-diffT-r=4-morerepeat-all-15-newL'
num_copy = 15
plot = True
for T,x0,y0 in params:
    out = f'{subfolder}/DWF_Mapping-{T}'
    data_name = f'{folder}/data/{T}.dat'
    image = Image.open(f'{folder}/mrc/{T}.tif')
    X, Y = image.size
    data_ori = np.memmap(data_name, dtype=np.float32, mode='r', shape=(X*Y, 256, 256))
    print(data_ori.shape)

    '--------(1)extraction data---------'
    test_position=[]
    #x0, y0 = 108,91
    x1, y1 = x0,y0-1+50
    position = AD.autodisk.bresenham_area(X, Y, x0, y0, x1, y1, y1-y0+1)
    #print(position.shape)
    for row in position:
        for value in row:
            test_position.append(data_ori[value])
    test_position = np.array(test_position).reshape(position.shape[0],position.shape[1],256,256)
    avg_pat_test = AD.autodisk.generateAvg(test_position, normalization=True)
    #print(np.sum(avg_pat_test))
    AD.autodisk.saveData(f'{folder}/{out}/avg_pat.dat',avg_pat_test,dataformat='np.float32', overwrite=True)
    AD.autodisk.visual(np.sqrt(np.sqrt(np.sqrt(avg_pat_test))),title='diffraction-pattern'
                       ,outdir=f'{folder}/{out}/diffraction-pattern.png')
    data = avg_pat_test

    '--------(2)Autodisk-----------'
    #(1)Find center
    ctr_ori = AD.autodisk.find_center_of_mass(data, threshold_value = 0.001) 
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
    thre=0.000001
    blobs = AD.autodisk.ctrDet(cros_map, r, kernel, 1, thre)  # 001-SiC:0.18      120-Si3N4:0.53 
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(data)),blobs,r)
    #(4)Marking the position of the diffraction spot!
    blobs_new = np.array(blobs)
    ctr = np.array(ctr_ori)
    sorted_blobs = blobs_new[np.argsort(np.linalg.norm(blobs_new - ctr, axis=1))]
    sorted_blobs=sorted_blobs+[1, 1]
    #print(sorted_blobs)
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),sorted_blobs,r,lwide=1,text=True,outdir=f'{folder}/{out}/disk.tif')
    #(5)Rejust from the ideal lattice
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
    print(ctr_ori0)
    refine=[ctr_ori0[0]-sorted_blobs[0][0],ctr_ori0[1]-sorted_blobs[0][1]]
    blobs = np.array(sorted_blobs) + np.array(refine)
    blobs = blobs[:43]
    AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),np.round(blobs, 1),r,lwide=1.5,xylim=None,text=True
                            ,outdir=f'{folder}/{out}/disk_select.tif')
    print(r)
    '-------(3)Debye Waller Factor mapping--------'
    arr1=[];arr2=[];arr3=[];intensity_all=[];lnq_all=[];lnq_all_unfixed=[];r_squared_all=[];intensity_single_all=[]
    intensity_0=[];intensity_1=[];intensity_2=[];intensity_3=[];intensity_4=[];intensity_5=[]
    '''
    for row in position:
        for value in row:
            data = data_ori[value]/np.sum(data_ori[value])
    '''
    '--copy caculation--'
    #num_copy = 5
    num_copy_sub = num_copy//2
    print('The Number of data enhancement:', num_copy)
    if num_copy_sub==0:
        position = position
    else:
        position = position[num_copy_sub:-num_copy_sub, num_copy_sub:-num_copy_sub]
    print(position.shape)
    for row in position:
        for value in row:
            HC_sub = []
            value_set = AD.autodisk.cal_copy(value,X,Y,num_copy)
            for value_x in range(num_copy):
                for value_y in range(num_copy):
                    HC_sub.append(data_ori[value_set[value_x, value_y]])
            HC_sub = np.array(HC_sub).reshape(num_copy,num_copy,256,256)
            data = AD.autodisk.generateAvg(HC_sub, normalization=True)
            #print(np.sum(data))
            scan_row = value//X-y0+1
            scan_col = value%X-x0+1
            #AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),blobs,r,lwide=1.5,xylim=None,text=True )
            '-----adjust to starin-----'
            r = 8
            ctr = AD.autodisk.find_center_of_mass(data, blobs, r, plot=False)
            ctr = AD.autodisk.find_center_of_mass(data, ctr, r, plot=False)
            #AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),ctr,r,lwide=0.5,xylim=None,text=True)
                                    #outdir=f'{folder}/{out}/ED_autodisk_intensity/ED-{scan_row}x{scan_col}_{value}.tif')
            '-----adjust to tilt----'
            center = ctr[0]
            ctr_select = ctr[7:13]
            '''
            if plot is True:
                fig, ax = plt.subplots(figsize = (5,5))
                ax.imshow(np.sqrt(np.sqrt(np.sqrt(data))), cmap='gray')
                #print(ctr_select)
                for idx, blob in enumerate(ctr_select, start=1):
                    y,x = blob
                    c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
                    ax.add_patch(c)
                    ax.text(x, y, str(idx), color='red', fontsize=12, ha='center', va='center')      
                plt.axis('off')
            '''
            '-------angle adjust------'
            angles = np.arctan2(ctr_select[:, 0] - center[0], ctr_select[:, 1] - center[1])
            ctr_select = ctr_select[np.argsort(angles)]
            if scan_row==2 and scan_col==2:
                fig, ax = plt.subplots(figsize = (5,5))
                ax.imshow(np.sqrt(np.sqrt(np.sqrt(data))), cmap='gray')
                #print(ctr_select)
                for idx, blob in enumerate(ctr_select, start=1):
                    y, x = blob
                    c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
                    ax.add_patch(c)
                    ax.text(x, y, str(idx-1), color='yellow', fontsize=12, ha='center', va='center')
                    #ax.text(x, y-10, f'{x:.2f},{y:.2f}', color='red', fontsize=8, ha='center', va='center')
                plt.axis('off')
                out_path_ED = f'{folder}/{out}/Intensity'
                if not os.path.isdir(out_path_ED):
                    os.makedirs(out_path_ED, exist_ok=True)
                plt.savefig(f'{out_path_ED}/pattern_tilt.png',dpi=300)
            r=4
            intensity = AD.autodisk.generateInt_new(data,ctr_select,r)#,disk=[1,6,6,6,12,6,6])    #disk=[2,2,4,2,4,2]
            intensity_0.append(intensity[0])
            intensity_1.append(intensity[1])
            intensity_2.append(intensity[2])
            intensity_3.append(intensity[3])
            intensity_4.append(intensity[4])
            intensity_5.append(intensity[5])
            '''
            for i_num in range(30,38,2):
                AD.autodisk.drawCircles(np.sqrt(np.sqrt(np.sqrt(data))),ctr,r,lwide=0.5,xylim=[ctr[i_num][1]-10,ctr[i_num][1]+10,ctr[i_num][0]+10,ctr[i_num][0]-10]
                                        ,text=True, outdir=f'{folder}/{subfolder}/highorder-disk/{T}-disk-refine-{i_num}.tif')
            '''
            intensity = AD.autodisk.generateInt_new(data,ctr[:43],r)#,disk=[1,6,6,6,12,6,6])    #disk=[2,2,4,2,4,2]
            #ctr,intensity = AD.autodisk.find_center_of_mass(data, blobs, r, ctr_ori, plot=False, show_intensity=True)
            intensity_all.append(sum(intensity[1:]))
            distances = np.linalg.norm(ctr - ctr[0], axis=1)
            distances_te = distances[1:]#*0.02/(0.0197*59.83948847466714)  
            intensity_te = intensity[1:]
            intensity_te_0 = np.array([np.sum(intensity_te[0:6]),np.sum(intensity_te[6:12]),np.sum(intensity_te[12:18])
                                     ,np.sum(intensity_te[18:30]/2),np.sum(intensity_te[30:36]),np.sum(intensity_te[36:42])
                                     ])
            intensity_single_all.append(intensity_te_0.T)
            if not os.path.exists(f'{folder}/{out}/imgdec_all_intensity'):
                os.makedirs(f'{folder}/{out}/imgdec_all_intensity')
            AD.autodisk.saveData(f'{folder}/{out}/imgdec_all_intensity/intensity_single-{scan_row}x{scan_col}_{value}.dat', intensity_te_0, overwrite=True)
            '-----adjust to cell structure----'
            #corrcet_factor = [1,4,1,1,4,4]
            corrcet_factor = [1, 4.04009349, 1.01105515, 1.01208224, 4.04923262, 4.04988759]
            #corrcet_factor =[1.,4.36229485,1.11066878,1.09673051,4.39575932,4.39016575]
            intensity_te[0:6] /=corrcet_factor[0]
            intensity_te[6:12] /=corrcet_factor[1]
            intensity_te[12:18] /=corrcet_factor[2]
            intensity_te[18:30] /=corrcet_factor[3]
            intensity_te[30:36] /=corrcet_factor[4]
            intensity_te[36:42] /=corrcet_factor[5]

            '-----order sum-------'
            distances_te_1 = np.array([np.mean(distances_te[0:6]),np.mean(distances_te[6:12]),np.mean(distances_te[12:18])
                                     ,np.mean(distances_te[18:30]),np.mean(distances_te[30:36]),np.mean(distances_te[36:42])
                                     #,np.mean(distances_te[42:54]),np.mean(distances_te[54:60])
                                     ])
            intensity_te_1 = np.array([np.sum(intensity_te[0:6]),np.sum(intensity_te[6:12]),np.sum(intensity_te[12:18])
                                     ,np.sum(intensity_te[18:30]/2),np.sum(intensity_te[30:36]),np.sum(intensity_te[36:42])
                                     #,np.sum(intensity_te[42:54]/2),np.sum(intensity_te[54:60])
                                     ])
            
            AD.autodisk.saveData(f'{folder}/{out}/imgdec_all_intensity/intensity_single.dat', intensity_te_1, overwrite=True)
            '-------scale bar-----'
            #distances_te_2 = distances_te_1*0.02/(0.0197*59.21942201152531) 
            distances_te_2 = np.array([0.4674900965098184, 0.8097165991902834, 0.9349801930196368, 1.2368625357505638, 1.4024702895294552, 1.6194331983805668])
            '-----background denoising-------'
            back_denosing = False
            if back_denosing==True:
                data_nodisk = AD.autodisk.generateNonedisk(data,ctr,r,plot=True)
                ctr = np.array(ctr, dtype=int)
                data_nodisk, data_nodisk_center = AD.autodisk.move_center(data_nodisk , center=ctr[0], square_size=256, plot=True)
                profile_distances, intensities = AD.autodisk.radial_intensity(data_nodisk, bin_width=1, plot=True, lim=[r+5,130,0,0.003])
                profile_distances, intensities = AD.autodisk.gauss_smooth(profile_distances, intensities, sigma=6,plot=True)
                intensity_te_2 = AD.autodisk.backmodel_denoising(data_nodisk, profile_distances, intensities, 
                                                                 distances_te_1, intensity_te_1, r,
                                                                 plot=True)
                '-------PM_slope-----'
                k,s,q,r_squared=AD.autodisk.PM_slope(intensity_te_2, distances_te_2, 'C')
            else:
                '-------PM_slope-----'
                k,s,q,r_squared=AD.autodisk.PM_slope(intensity_te_1, distances_te_2, 'C')
            
            s = np.array(s)
            q = np.array(q)
            lnq_all.append(q)
            k_unfixed,s_unfixed,q_unfixed,r_squared_unfixed=AD.autodisk.PM_slope(intensity_te_0, distances_te_2, 'C')
            lnq_all_unfixed.append(q_unfixed)
            '------select disk fitting------'
            mask = [1,4,5]
            s_ = s[mask]
            q_ = q[mask]
            m, b = np.polyfit(s, q, 1)
            if 0 in intensity_te:
                print(f"Stopping loop at index {value} due to zero intensity.")
                m=0
            arr1.append(m/2)
            r_squared_all.append(r_squared)
            if plot is True:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.imshow(np.sqrt(np.sqrt(np.sqrt(data))),cmap='gray')
                for blob in ctr:
                    y, x = blob
                    c = plt.Circle((x, y),r, color='red', linewidth=1.5, fill=False)
                    ax1.add_patch(c)
                ax1.set_title(f'ED-{scan_row}x{scan_col}_{value}', fontname='Arial', fontweight='bold', fontsize=12)
                ax2.scatter(s, q, s=25, c='blue')
                ax2.tick_params(axis='both', which='major', labelsize=12, labelrotation=0)
                for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
                    tick.set_fontname('Arial')
                    tick.set_fontweight('bold')
                for ax in [ax1, ax2]:
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                ax2.plot(s, m*s + b, color='red')
                ax2.text(0.05, 0.9, f'y = {m:.2f}x + {b:.2f}', transform=plt.gca().transAxes, fontname='Arial', fontweight='bold', fontsize=12)
                plt.ylim([3, 12])
                ax2.set_xlabel('s^2', fontname='Arial', fontweight='bold', fontsize=12)
                ax2.set_ylabel('lnq', fontname='Arial', fontweight='bold', fontsize=12)
                #plt.text(0.05, 5, f'No.{value}', ha='left', va='top',fontname='Arial', fontweight='bold', fontsize=12)
                ax2.text(0.05, 0.8, f'No.{value}', transform=plt.gca().transAxes, fontname='Arial', fontweight='bold', fontsize=12)
                if not os.path.exists(f'{folder}/{out}/imgdec_all_DWF'):
                    os.makedirs(f'{folder}/{out}/imgdec_all_DWF')
                plt.savefig(f'{folder}/{out}/imgdec_all_DWF/intensity_distance-{scan_row}x{scan_col}.png',dpi=300)
                plt.show()
            plot = False

    intensity_0 = np.array(intensity_0).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr1.dat', intensity_0, overwrite=True)
    intensity_1 = np.array(intensity_1).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr2.dat', intensity_1, overwrite=True)
    intensity_2 = np.array(intensity_2).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr3.dat', intensity_2, overwrite=True)
    intensity_3 = np.array(intensity_3).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr4.dat', intensity_3, overwrite=True)
    intensity_4 = np.array(intensity_4).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr5.dat', intensity_4, overwrite=True)
    intensity_5 = np.array(intensity_5).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/Iarr6.dat', intensity_5, overwrite=True)
    ratio_min=0.9
    ratio_max=1.1
    ratio1 = intensity_0/intensity_3
    mask1 = np.where((ratio1 <= ratio_max) & (ratio1 >= ratio_min), 1, 0)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio14.dat', ratio1, overwrite=True)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio14_mask.dat', ratio1*mask1, overwrite=True)
    ratio2 = intensity_1/intensity_4
    mask2 = np.where((ratio2 <= ratio_max) & (ratio2 >= ratio_min), 1, 0)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio25.dat', ratio2, overwrite=True)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio25_mask.dat', ratio2*mask2, overwrite=True)
    ratio3 = intensity_2/intensity_5
    mask3 = np.where((ratio3 <= ratio_max) & (ratio3 >= ratio_min), 1, 0)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio36.dat', ratio3, overwrite=True)
    AD.autodisk.saveData(f'{folder}/{out}/Intensity/ratio36_mask.dat', ratio3*mask3, overwrite=True)
    
    arr1 = np.array(arr1).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/arr1.dat', arr1, overwrite=True)
    
    lnq_all = np.array(lnq_all)
    AD.autodisk.saveData(f'{folder}/{out}/lnq_all.dat', lnq_all, overwrite=True)
    average_lnq_all = np.mean(lnq_all, axis=0)
    combined_list = list(zip(s, average_lnq_all))
    AD.autodisk.saveData(f'{folder}/{out}/lnq_s2_average.dat', combined_list, overwrite=True)
    lnq_all_unfixed = np.array(lnq_all_unfixed)
    AD.autodisk.saveData(f'{folder}/{out}/lnq_all_unfixed.dat', lnq_all_unfixed, overwrite=True)
    average_lnq_all_unfixed = np.mean(lnq_all_unfixed, axis=0)
    combined_list_unfixed = list(zip(s, average_lnq_all_unfixed))
    AD.autodisk.saveData(f'{folder}/{out}/lnq_s2_average_unfixed.dat', combined_list_unfixed, overwrite=True)
    
    average_intensity_all = np.mean(intensity_single_all, axis=0)
    AD.autodisk.saveData(f'{folder}/{out}/average_intensity.dat', average_intensity_all, overwrite=True)

    intensity_all = np.array(intensity_all).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/intensity_all.dat', intensity_all, overwrite=True)
    I_max_se=max(map(max, intensity_all))/max(map(max, intensity_all))
    I_min_se=0/max(map(max, intensity_all))
    print('IntensityMax:',max(map(max, intensity_all)), I_max_se, I_min_se)
    mask = np.where((intensity_all <= np.max(intensity_all)*I_max_se) & (intensity_all >= np.max(intensity_all)*I_min_se), 1, 0)
    #mask=np.where(intensity_all >= np.max(intensity_all)*0.25, 1, 0)
    AD.autodisk.saveData(f'{folder}/{out}/intensity_all_1.dat', intensity_all*mask, overwrite=True)
    
    r_squared_all = np.array(r_squared_all).reshape(len(position),len(position[0]))
    AD.autodisk.saveData(f'{folder}/{out}/r_squared.dat', r_squared_all, overwrite=True)
    
    arr1_1,x,vx = AD.autodisk.array_average(arr1, mask)
    print('Average Value:',x,'\n sample_std_dev:',vx)
    output_text = f'{x}\n{vx}'
    with open(f'{folder}/{out}/output.txt', 'w') as f:
        f.write(output_text)
    AD.autodisk.saveData(f'{folder}/{out}/arr1_1.dat', arr1_1, overwrite=True)
    AD.autodisk.saveData(f'{folder}/{out}/arr1_2.dat', arr1*mask, overwrite=True)
    AD.autodisk.saveData(f'{folder}/{out}/arr1_3.dat', arr1*mask1*mask2*mask3, overwrite=True)
E=[]
i_number=0
for T in range(200,951,50):
    out = f'{subfolder}/DWF_Mapping-{T}'
    with open(f'{folder}/{out}/output.txt', 'r') as file:
        data = [float(num) for line in file for num in line.strip().split()]
    E_son = [T, data[0], data[1]]
    E.extend(E_son)
    print(E_son)
    i_number +=1
E = np.array(E).reshape(i_number,3)
print(E)
AD.autodisk.saveData(f'{folder}/{subfolder}/T-DWF-error.dat', E, overwrite=True)