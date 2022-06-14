import multiprocessing
from os import listxattr, setuid
from numpy.core.defchararray import index, multiply, upper
from osgeo import gdal
import cv2
import tifffile as tiff
import open3d as o3d
import matplotlib
import numpy as np
import math
import csv
from pyproj import Proj, transform
from pyproj import transform
from pyproj import *
from scipy.optimize import lsq_linear

UTM2latlon_transformer = Transformer.from_crs(2152, 4326)
latlon2UTM_transformer = Transformer.from_crs(4326, 2152)

def utm_to_latlon(easting, northing):

    # inProj = Proj('epsg:2152') # UTM zone 12
    # outProj = Proj('epsg:4326') # latlon 
    
    
    lat,lon = transform(2152,4326,easting,northing)
    # lat = round(lat,7)
    # lon = round(lon,7)

    # lat,lon = transform(inProj,outProj,easting,northing)

    # SE_utm = utm.from_latlon(33.07451869, -111.97477775)
    # utm_zone = SE_utm[2]
    # utm_num  = SE_utm[3]

    # lat, lon = utm.to_latlon(easting, northing, utm_zone, utm_num)

    return lon,lat

def latlon_to_utm(lon,lat):

    # inProj = Proj('epsg:4326') # latlon 
    # outProj = Proj('epsg:2152') # UTM zone 12
    
    # easting,northing = transform(inProj,outProj,lat,lon)
    # lat = round(lat,7)
    # lon = round(lon,7)
    easting,northing = transform(4326,2152,lat,lon)

    # SE_utm = utm.from_latlon(lat,lon)
    # easting = SE_utm[0]
    # northing = SE_utm[1]

    return easting,northing

def get_coord_from_tiff(path):

    ds = gdal.Open(path)
    meta = gdal.Info(ds)

    lines = meta.splitlines()

    for line in lines:
        if 'Upper Left' in line:
            u_l = line.split()[2:4]
            u_l[0] = u_l[0].replace('(','').replace(',','')
            u_l[1] = u_l[1].replace(')','')

        if 'Lower Left' in line:
            l_l = line.split()[2:4]
            l_l[0] = l_l[0].replace('(','').replace(',','')
            l_l[1] = l_l[1].replace(')','')

        if 'Upper Right' in line: 
            u_r = line.split()[2:4]
            u_r[0] = u_r[0].replace('(','').replace(',','')
            u_r[1] = u_r[1].replace(')','')

        if 'Lower Right' in line:
            l_r = line.split()[2:4]
            l_r[0] = l_r[0].replace('(','').replace(',','')
            l_r[1] = l_r[1].replace(')','')

        if 'Center' in line:
            c = line.split()[1:3]
            c[0] = c[0].replace('(','').replace(',','')
            c[1] = c[1].replace(')','')

    # upper_left = latlon_to_utm(float(u_l[0]),float(u_l[1]))
    # lower_left = latlon_to_utm(float(l_l[0]),float(l_l[1]))
    # upper_right = latlon_to_utm(float(u_r[0]),float(u_r[1]))
    # lower_right = latlon_to_utm(float(l_r[0]),float(l_r[1]))
    # center = latlon_to_utm(float(c[0]),float(c[1]))

    upper_left = (float(u_l[0]),float(u_l[1]))
    lower_left = (float(l_l[0]),float(l_l[1]))
    upper_right = (float(u_r[0]),float(u_r[1]))
    lower_right = (float(l_r[0]),float(l_r[1]))
    center = (float(c[0]),float(c[1]))
    
    coord = {'UL':upper_left,'LL':lower_left,'UR':upper_right,'LR':lower_right,'C':center}

    return coord

def load_full_ortho(path):

    main_image = tiff.imread(path)
    main_image = cv2.cvtColor(main_image,cv2.COLOR_RGB2BGR)

    return main_image

def mark_plants_on_ortho(ortho,ortho_coord_org,plants):
    
    # ortho_coord = {}
    # ortho_coord['UL'] = utm_to_latlon(ortho_coord_org['UL'][0],ortho_coord_org['UL'][1])
    # ortho_coord['UR'] = utm_to_latlon(ortho_coord_org['UR'][0],ortho_coord_org['UR'][1])
    # ortho_coord['LL'] = utm_to_latlon(ortho_coord_org['LL'][0],ortho_coord_org['LL'][1])
    # ortho_coord['LR'] = utm_to_latlon(ortho_coord_org['LR'][0],ortho_coord_org['LR'][1])
    
    ortho_coord = ortho_coord_org
    plants_ortho_bounding_boxes = []

    for p in plants:
        # lon,lat = utm_to_latlon(p[0],p[1])
        # lon = p[0]
        # lat = p[1]

        if p['UL']['lon']>=ortho_coord['UL'][0] and p['UL']['lon']<= ortho_coord['UR'][0] and \
            p['UL']['lat']>=ortho_coord['LL'][1] and p['UL']['lat']<= ortho_coord['UR'][1] and \
            p['LR']['lon']>=ortho_coord['UL'][0] and p['LR']['lon']<= ortho_coord['UR'][0] and \
            p['LR']['lat']>=ortho_coord['LL'][1] and p['LR']['lat']<= ortho_coord['UR'][1]:

            # lon,lat = latlon_to_utm(lon,lat)

            # c = (int((lon - ortho_coord_org['UL'][0])/(ortho_coord_org['UR'][0]-ortho_coord_org['UL'][0])*ortho.shape[1]),\
            #     int((ortho_coord_org['UL'][1]-lat)/(ortho_coord_org['UL'][1]-ortho_coord_org['LL'][1])*ortho.shape[0]))

            # cv2.circle(ortho,c,10,(0,0,255),-1)

            p1 = (int((p['UL']['lon'] - ortho_coord_org['UL'][0])/(ortho_coord_org['UR'][0]-ortho_coord_org['UL'][0])*ortho.shape[1]),\
                int((ortho_coord_org['UL'][1]-p['UL']['lat'])/(ortho_coord_org['UL'][1]-ortho_coord_org['LL'][1])*ortho.shape[0]))

            p2 = (int((p['LR']['lon'] - ortho_coord_org['UL'][0])/(ortho_coord_org['UR'][0]-ortho_coord_org['UL'][0])*ortho.shape[1]),\
                int((ortho_coord_org['UL'][1]-p['LR']['lat'])/(ortho_coord_org['UL'][1]-ortho_coord_org['LL'][1])*ortho.shape[0]))

            cv2.rectangle(ortho,p1,p2,(0,0,255),4)
            plants_ortho_bounding_boxes.append({'UL':p1,'UR':(p2[0],p1[1]),'LL':(p1[0],p2[1]),'LR':p2})

    return ortho,plants_ortho_bounding_boxes

def clip_ortho(ortho,ortho_coord,coord):

    lon_distance = ortho_coord['UR'][0]-ortho_coord['UL'][0]
    lat_distance = ortho_coord['UL'][1]-ortho_coord['LL'][1]

    x1 = int((ortho_coord['UL'][1]-coord['UL'][1])/(lat_distance)*ortho.shape[0])
    # y1 = int((coord['UL'][0]-ortho_coord['UL'][0])/(lon_distance)*ortho.shape[1])
    y1 = 0

    x2 = int((ortho_coord['UL'][1]-coord['LR'][1])/(lat_distance)*ortho.shape[0])
    # y2 = int((coord['LR'][0]-ortho_coord['UL'][0])/(lon_distance)*ortho.shape[1])
    y2 = ortho.shape[1]

    clipped_coords = {'UL':(ortho_coord['UL'][0],coord['UL'][1]),\
        'UR':(ortho_coord['UR'][0],coord['UL'][1]),\
        'LL':(ortho_coord['UL'][0],coord['LR'][1]),\
        'LR':(ortho_coord['UR'][0],coord['LR'][1])}
    
    return ortho[x1:x2,y1:y2], clipped_coords

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path,format="ply")
    # mins = np.min(np.array(pcd.points),axis=0)
    # maxs = np.max(np.array(pcd.points),axis=0)
    # # print(mins)
    # # print(maxs)

    # center = ((mins[0]+maxs[0])/2,(mins[1]+maxs[1])/2,(mins[2]+maxs[2])/2)
    # # R = pcd.get_rotation_matrix_from_xyz((np.pi, 0,0))
    # # pcd = pcd.rotate(R, center=((mins[0]+maxs[0])/2,(mins[1]+maxs[1])/2,(mins[2]+maxs[2])/2))
    # pcd = pcd.translate((0,0,0), relative=False)
    # T = np.eye(4)
    # T[1,1] = -1
    # pcd = pcd.transform(T)
    # pcd = pcd.translate(center, relative=False)
    # # print("----")
    # mins = np.min(np.array(pcd.points),axis=0)
    # maxs = np.max(np.array(pcd.points),axis=0)
    # # print(mins)
    # # print(maxs)

    return pcd

def down_sample_pcd(pcd,voxel_size):
    return pcd.voxel_down_sample(voxel_size)

def generate_height_image_from_pcd(pcd,ortho,ortho_coord,win_size=5):
    
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    
    lon_distance = ortho_coord['UR'][0]-ortho_coord['UL'][0]
    lat_distance = ortho_coord['UL'][1]-ortho_coord['LL'][1]

    mins = np.min(np.array(pcd.points),axis=0)
    maxs = np.max(np.array(pcd.points),axis=0)
    
    min_latlon = utm_to_latlon(mins[0],mins[1])
    # mins = (min_latlon[0],min_latlon[1],mins[2])

    max_latlon = utm_to_latlon(maxs[0],maxs[1])
    # maxs = (max_latlon[0],max_latlon[1],maxs[2])

    image = np.zeros((int((max_latlon[1]-min_latlon[1])/(lat_distance)*ortho.shape[0]),\
        int((max_latlon[0]-min_latlon[0])/(lon_distance)*ortho.shape[1]),3))

    for pt in pcd.points:
        # tmp = utm_to_latlon(pt[0],pt[1])
        # p = (tmp[0],tmp[1],pt[2])
        p = pt

        offset = (int(math.floor((p[0]-mins[0])/(maxs[0]-mins[0])*(image.shape[1]))),\
            int(math.floor((p[1]-mins[1])/(maxs[1]-mins[1])*(image.shape[0]))))
        
        color_ratio = (p[2]-mins[2])/(maxs[2]-mins[2])
        mapped_color = cmap(color_ratio)
        
        if offset[1]-win_size > 0 and offset[1]+win_size < image.shape[0] and\
        offset[0]-win_size > 0 and offset[0]+win_size < image.shape[1]:
            image[offset[1]-win_size:offset[1]+win_size,\
            offset[0]-win_size:offset[0]+win_size,0] += int(mapped_color[2]*255)

            image[offset[1]-win_size:offset[1]+win_size,\
            offset[0]-win_size:offset[0]+win_size,1] += int(mapped_color[1]*255)

            image[offset[1]-win_size:offset[1]+win_size,\
            offset[0]-win_size:offset[0]+win_size,2] += int(mapped_color[0]*255)
        elif offset[1] >= 0 and offset[1] < image.shape[0] and offset[0] >= 0 and offset[0] < image.shape[1]:
            image[offset[1],offset[0],:] = [int(mapped_color[2]*255),int(mapped_color[1]*255),int(mapped_color[0]*255)]

    image = cv2.normalize(image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    image = image.astype('uint8')

    image = cv2.flip(image,0)

    cv2.putText(image,"SW",(100,image.shape[0]-100),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255,255), 3, cv2.LINE_AA)
    cv2.putText(image,"NW",(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255,255), 3, cv2.LINE_AA)
    cv2.putText(image,"SE",(image.shape[1]-100,image.shape[0]-100),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255,255), 3, cv2.LINE_AA)
    cv2.putText(image,"NE",(image.shape[1]-100,100),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255,255), 3, cv2.LINE_AA)

    bounds_utm = {'UL':(mins[0],maxs[1]),'LR':(maxs[0],mins[1]),'UR':(maxs[0],maxs[1]),'LL':(mins[0],mins[1])}

    bounds = {'UL':utm_to_latlon(mins[0],maxs[1]),\
        'LR':utm_to_latlon(maxs[0],mins[1]),\
        'UR':utm_to_latlon(maxs[0],maxs[1]),\
        'LL':utm_to_latlon(mins[0],mins[1])}
    
    return image,bounds,bounds_utm

def apply_T_to_point(point,T):

    if len(point) != 3:
        point = [point[0],point[1],1]

    new_point = np.matmul(T,point)

    if T.shape[0] == 3:
        new_point /= new_point[2]

    return new_point[0],new_point[1]

def apply_T_to_point_3d(point,T):

    if len(point) != 4:
        point = [point[0],point[1],point[2],1]

    new_point = np.matmul(T,point)

    if T.shape[0] == 4:
        new_point /= new_point[3]

    return new_point[0],new_point[1],new_point[2]

def get_size_from_T(image_apply,image_other,T):
    list_x = []
    list_y = []

    transformed = apply_T_to_point((0,0),T)
    list_x.append(transformed[0])
    list_y.append(transformed[1])

    transformed = apply_T_to_point((image_apply.shape[1],0),T)
    list_x.append(transformed[0])
    list_y.append(transformed[1])

    transformed = apply_T_to_point((image_apply.shape[1],image_apply.shape[0]),T)
    list_x.append(transformed[0])
    list_y.append(transformed[1])

    transformed = apply_T_to_point((0,image_apply.shape[0]),T)
    list_x.append(transformed[0])
    list_y.append(transformed[1])

    list_x.append(0)
    list_x.append(image_other.shape[1])

    list_y.append(0)
    list_y.append(image_other.shape[0])

    return (int(math.ceil(np.max(list_x)-np.min(list_x))),int(math.ceil(np.max(list_y)-np.min(list_y)))),abs(int(np.min(list_x))),abs(int(np.min(list_y)))

def merge_two_images_with_T(image_apply,image_other,T,c1,c2):

    size,min_x,min_y = get_size_from_T(image_apply,image_other,T)
    result = cv2.warpPerspective(image_apply, T,size)
    
    result[0+min_y:image_other.shape[0]+min_y, 0+min_x:image_other.shape[1]+min_x,:] = \
        cv2.addWeighted(result[0+min_y:image_other.shape[0]+min_y, 0+min_x:image_other.shape[1]+min_x,:], c1, image_other, c2, 0.0)

    return result

def get_rotation_around_point(theta,p):
    T = np.eye(3)
    T[0,0] = math.cos(math.radians(theta))
    T[0,1] = -math.sin(math.radians(theta))
    T[1,0] = math.sin(math.radians(theta))
    T[1,1] = math.cos(math.radians(theta))

    T_tr_neg = np.eye(3)
    T_tr_neg[0,2] = -p[0]
    T_tr_neg[1,2] = -p[1]

    T_tr_pos = np.eye(3)
    T_tr_pos[0,2] = p[0]
    T_tr_pos[1,2] = p[1]

    return np.matmul(T_tr_pos,np.matmul(T,T_tr_neg))

def compose_T(param_dict):

    T = np.eye(3)
    T[0,2] = param_dict['tr_x']
    T[1,2] = param_dict['tr_y']

    T[0,0] = param_dict['s_x']*math.cos(math.radians(param_dict['theta']))
    T[0,1] = -math.sin(math.radians(param_dict['theta']))
    T[1,0] = math.sin(math.radians(param_dict['theta']))
    T[1,1] = param_dict['s_y']*math.cos(math.radians(param_dict['theta']))

    return T

def decompose_affine(T):
    scale = math.sqrt(T[0,0]**2+T[1,0]**2)
    theta = math.atan(T[1,0]/T[0,0])
    shear = (T[0,1]+T[1,0])/(T[0,0])
    t_x = T[0,2]
    t_y = T[1,2]

    return scale,theta,shear,t_x,t_y

def generate_and_append_banner(merged_image,params):
    banner = np.zeros((300,merged_image.shape[1],3)).astype('uint8')
    banner[:,:,0] = 54
    banner[:,:,1] = 52
    banner[:,:,2] = 45

    cv2.putText(banner,"Translate: ",(50,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 4, cv2.LINE_AA)
    cv2.putText(banner,"I, J, K, L",(400,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Rotate: ",(900,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"Q, E keys",(1200,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Scale x: ",(1700,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"A, D keys",(2050,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Scale y: ",(2500,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"W, S keys",(2850,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Adjust Brightness: ",(3300,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"G, H keys",(3980,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Exit: ",(4440,90),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"Esc",(4640,90),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.line(banner,(40,150),(7594,150),(233, 230, 223),3,cv2.LINE_AA)

    cv2.putText(banner,"Current Parameters: ",(50,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (85, 112, 225), 3, cv2.LINE_AA)

    cv2.putText(banner,"Translation (x,y): ",(880,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"{0}, {1}".format(params['tr_x'],params['tr_y']),(1540,240),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Rotate (theta): ",(1870,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"{0:.1f}".format(params['theta']),(2490,240),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Scale (x,y): ",(2750,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"{0:.2f}, {1:.2f}".format(params['s_x'],params['s_y']),(3290,240),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Opacity: ",(3790,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,"{0:.2f}".format(params['ortho_opacity']),(4170,240),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.putText(banner,"Increment: ",(4490,240),cv2.FONT_HERSHEY_SIMPLEX, 2.2, (110, 203, 253), 3, cv2.LINE_AA)
    cv2.putText(banner,("Large" if params['tr_interval']==10 else "Small"),(5000,240),cv2.FONT_HERSHEY_SIMPLEX, 1.8, (233, 230, 223), 3, cv2.LINE_AA)

    cv2.rectangle(merged_image,(2,2),(merged_image.shape[1]-2,merged_image.shape[0]-2),(0,0,150),5)
    res = cv2.vconcat([banner,merged_image])

    return res
    
def get_full_transformation(src,dst):
    
    A = []
    b = []

    for i,p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0],p1[1],1,0,0,0,0,0,0])
        b.append(p2[0])

        A.append([0,0,0,p1[0],p1[1],1,0,0,0])
        b.append(p2[1])

        A.append([0,0,0,0,0,0,p1[0],p1[1],1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    T = X.reshape((3,3))

    return T

def get_full_transformation_3d(src,dst):
    
    A = []
    b = []

    for i,p1 in enumerate(src):

        p2 = dst[i]

        A.append([p1[0],p1[1],p1[2],1,0,0,0,0,0,0,0,0,0,0,0,0])
        b.append(p2[0])

        A.append([0,0,0,0,p1[0],p1[1],p1[2],1,0,0,0,0,0,0,0,0])
        b.append(p2[1])

        A.append([0,0,0,0,0,0,0,0,p1[0],p1[1],p1[2],1,0,0,0,0])
        b.append(p2[2])

        A.append([0,0,0,0,0,0,0,0,0,0,0,0,p1[0],p1[1],p1[2],1])
        b.append(1)

    A = np.array(A)
    b = np.array(b)

    res = lsq_linear(A, b)
    X = res.x

    print(np.matmul(A,X)-b)
    T = X.reshape((4,4))

    return T

def get_pcd_corners_in_ortho_GPS(pcd_image,ortho_image,clipped_coords,T):
    
    UL = [0,0]
    UR = [pcd_image.shape[1],0]
    LL = [0,pcd_image.shape[0]]
    LR = [pcd_image.shape[1],pcd_image.shape[0]]

    # ortho image to GPS

    OI_UL = [0,0]
    OI_UR = [ortho_image.shape[1],0]
    OI_LL = [0,ortho_image.shape[0]]
    OI_LR = [ortho_image.shape[1],ortho_image.shape[0]]

    src = np.array([OI_UL,OI_UR,OI_LL,OI_LR])
    dst = np.array([clipped_coords['UL'],clipped_coords['UR'],clipped_coords['LL'],clipped_coords['LR']])

    # H,_ = cv2.findHomography(src,dst)
    H = get_full_transformation(src,dst)

    # pcd image to ortho coords

    UL_OC = apply_T_to_point(UL,T)
    UR_OC = apply_T_to_point(UR,T)
    LL_OC = apply_T_to_point(LL,T)
    LR_OC = apply_T_to_point(LR,T)

    # ortho coords to GPS

    UL_GPS = apply_T_to_point(UL_OC,H)
    UR_GPS = apply_T_to_point(UR_OC,H)
    LL_GPS = apply_T_to_point(LL_OC,H)
    LR_GPS = apply_T_to_point(LR_OC,H)

    corrected_coords = {}
    # corrected_coords["UL"] = latlon_to_utm(UL_GPS[0],UL_GPS[1])
    # corrected_coords["UR"] = latlon_to_utm(UR_GPS[0],UR_GPS[1])
    # corrected_coords["LL"] = latlon_to_utm(LL_GPS[0],LL_GPS[1])
    # corrected_coords["LR"] = latlon_to_utm(LR_GPS[0],LR_GPS[1])
    corrected_coords["UL"] = UL_GPS
    corrected_coords["UR"] = UR_GPS
    corrected_coords["LL"] = LL_GPS
    corrected_coords["LR"] = LR_GPS
    
    return corrected_coords

def transform_pcd_after_manual_correction(pcd,original_coords,original_coords_utm,corrected_coords):
   
    corrected_coords['UL'] = latlon_to_utm(corrected_coords['UL'][0],corrected_coords['UL'][1])
    corrected_coords['UR'] = latlon_to_utm(corrected_coords['UR'][0],corrected_coords['UR'][1])
    corrected_coords['LL'] = latlon_to_utm(corrected_coords['LL'][0],corrected_coords['LL'][1])
    corrected_coords['LR'] = latlon_to_utm(corrected_coords['LR'][0],corrected_coords['LR'][1])
    
    # src = np.array([list(original_coords_utm['UL'])+[1],\
    #     list(original_coords_utm['UR'])+[1],\
    #     list(original_coords_utm['LL'])+[1],\
    #     list(original_coords_utm['LR'])+[1]])

    # dst = np.array([list(corrected_coords['UL'])+[1],\
    #     list(corrected_coords['UR'])+[1],\
    #     list(corrected_coords['LL'])+[1],\
    #     list(corrected_coords['LR'])+[1]])

    center_orig = ((original_coords_utm['UL'][0]+original_coords_utm['UR'][0])/2,\
        (original_coords_utm['UL'][1]+original_coords_utm['LL'][1])/2)

    center_corr = ((corrected_coords['UL'][0]+corrected_coords['UR'][0])/2,\
        (corrected_coords['UL'][1]+corrected_coords['LL'][1])/2)

    src = np.array([original_coords_utm['UL'],\
        original_coords_utm['UR'],\
        original_coords_utm['LL'],\
        original_coords_utm['LR'],\
        center_orig]).astype('float32')

    dst = np.array([corrected_coords['UL'],\
        corrected_coords['UR'],\
        corrected_coords['LL'],\
        corrected_coords['LR'],\
        center_corr]).astype('float32')

    # print(src)
    # print(dst)

    # H,_ = cv2.estimateAffinePartial2D(src,dst)
    H,_ = cv2.estimateAffine2D(src,dst)
    # H,_ = cv2.findHomography(src,dst)
    # H = cv2.getAffineTransform(src[:3,:],dst[:3,:])
    # H = get_full_transformation_3d(src,dst)
    # H = get_full_transformation(src,dst)
    
    # UL_TEST = apply_T_to_point_3d(list(original_coords_utm['UL'])+[1],H)
    # UR_TEST = apply_T_to_point_3d(list(original_coords_utm['UR'])+[1],H)
    # LL_TEST = apply_T_to_point_3d(list(original_coords_utm['LL'])+[1],H)
    # LR_TEST = apply_T_to_point_3d(list(original_coords_utm['LR'])+[1],H)

    UL_TEST = apply_T_to_point(original_coords_utm['UL'],H)
    UR_TEST = apply_T_to_point(original_coords_utm['UR'],H)
    LL_TEST = apply_T_to_point(original_coords_utm['LL'],H)
    LR_TEST = apply_T_to_point(original_coords_utm['LR'],H)

    # print(UL_TEST,corrected_coords['UL'])
    # print(UR_TEST,corrected_coords['UR'])
    # print(LL_TEST,corrected_coords['LL'])
    # print(LR_TEST,corrected_coords['LR'])

    tr_x_final = [corrected_coords['UL'][0]-UL_TEST[0],\
        corrected_coords['UR'][0]-UR_TEST[0],\
        corrected_coords['LL'][0]-LL_TEST[0],\
        corrected_coords['LR'][0]-LR_TEST[0]]

    tr_y_final = [corrected_coords['UL'][1]-UL_TEST[1],\
        corrected_coords['UR'][1]-UR_TEST[1],\
        corrected_coords['LL'][1]-LL_TEST[1],\
        corrected_coords['LR'][1]-LR_TEST[1]]
    
    tr_x_final = np.mean(tr_x_final)
    tr_y_final = np.mean(tr_y_final)

    # print(tr_x_final)
    # print(tr_y_final)
    # H = get_full_transformation(src,dst)
    
    # sc,th,sh,tx,ty = decompose_affine(H)
    # print(sc,th,sh,tx,ty)
    # # H,_ = cv2.findHomography(src,dst)

    # new_lon_d = corrected_coords['UR'][0] - corrected_coords['UL'][0]
    # old_lon_d = original_coords_utm['UR'][0] - corrected_coords['UL'][0]

    # new_lat_d = corrected_coords['UL'][1] - corrected_coords['LL'][1]
    # old_lat_d = original_coords_utm['UL'][1] - corrected_coords['LL'][1]
    
    # # s_lon = new_lon_d/old_lon_d
    # # s_lat = new_lat_d/old_lat_d

    # # s_alt = (s_lon+s_lat)/2
    
    T = np.eye(4)

    # # T[0,0] = s_lon
    # # T[1,1] = s_lat
    # # T[2,2] = s_alt

    T[0:2,0:2] = H[0:2,0:2]

    T[0,3] = H[0,2]
    T[1,3] = H[1,2]
    
    # T[3,3] = H[2,2]
    # T[3,0:2] = H[2,0:2]
    
    # print(T)
    # print(H)
    

    pcd = pcd.transform(T)
    pcd = pcd.translate((tr_x_final,tr_y_final,0))

    # center = ((corrected_coords['UL'][0]+corrected_coords['UR'][0])/2,\
    #     (corrected_coords['UL'][1]+corrected_coords['LL'][1])/2,0)

    # pcd = pcd.translate(center, relative=False)

    # mins = np.min(np.array(pcd.points),axis=0)
    # maxs = np.max(np.array(pcd.points),axis=0)

    # print(mins)
    # print(maxs)
    # print(np.array(pcd.points))

    # -----------------

    # mins = np.min(np.array(pcd.points),axis=0)
    # maxs = np.max(np.array(pcd.points),axis=0)

    # print(mins)
    # print(maxs)

    # lon_d_utm = pcd_bounds['UR'][0]-pcd_bounds['UL'][0]
    # lat_d_utm = pcd_bounds['UL'][1]-pcd_bounds['LL'][1]

    # tr_x_utm = params['tr_x']*lon_d_utm/(pcd_image.shape[1])
    # tr_y_utm = params['tr_y']*lat_d_utm/(pcd_image.shape[0])

    # print(tr_x_utm,tr_y_utm,params['s_x'])

    # center = ((mins[0]+maxs[0])/2,(mins[1]+maxs[1])/2,(mins[2]+maxs[2])/2)
    # pcd = pcd.translate((0,0,0), relative=False)

    # T = np.eye(4)
    # T[0,0] = params['s_x']
    # # T[0,3] = tr_x_utm
    # # T[1,3] = tr_y_utm

    # pcd = pcd.transform(T)
    # pcd = pcd.translate(center, relative=False)

    # pcd = pcd.translate((-tr_x_utm*params['s_x'],tr_y_utm,0), relative=True)

    # mins = np.min(np.array(pcd.points),axis=0)
    # maxs = np.max(np.array(pcd.points),axis=0)

    # print(mins)
    # print(maxs)

    return pcd

def paint_plants(pcd,plants):
    
    pcd.paint_uniform_color([1,1,1])

    mins = np.min(np.array(pcd.points),axis=0)
    maxs = np.max(np.array(pcd.points),axis=0)

    ll_mins = utm_to_latlon(mins[0],mins[1])
    ll_maxs = utm_to_latlon(maxs[0],maxs[1])
    # ll_mins = mins
    # ll_maxs = maxs

    tree = o3d.geometry.KDTreeFlann(pcd)

    for p in plants:

        if p[0]>=ll_mins[0] and p[0]<=ll_maxs[0] and p[1]>=ll_mins[1] and p[1]<=ll_maxs[1]:
            tmp = latlon_to_utm(p[0],p[1])
            # tmp = p
            p = np.array([tmp[0],tmp[1],maxs[2]])
            
            [k, idx, _] = tree.search_knn_vector_3d(p, 100)
            np.asarray(pcd.colors)[idx[1:], :] = (0,1,0)
    
    return pcd

def read_plants(path):
    plants = []
    
    with open(path, mode='r',encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[0] != "date" and rows[0] != "EMPTY":
                
                
                bb = {'UL':{'lon':float(rows[11]),'lat':float(rows[10])},\
                    'UR':{'lon':float(rows[13]),'lat':float(rows[10])},\
                    'LL':{'lon':float(rows[11]),'lat':float(rows[12])},\
                    'LR':{'lon':float(rows[13]),'lat':float(rows[12])}}

                plants.append(bb)
                
    
    plants = np.array(plants)

    return plants

def save_pcd(pcd,path):
    o3d.io.write_point_cloud(path, pcd)

def get_plant_loc_on_heatmap(plants,T):

    plant_loc_in_heatmap = []

    for p in plants:
        heatmap_loc = {
        'UL' : apply_T_to_point(p['UL'],np.linalg.inv(T)),\
        'UR' : apply_T_to_point(p['UR'],np.linalg.inv(T)),\
        'LL' : apply_T_to_point(p['LL'],np.linalg.inv(T)),\
        'LR' : apply_T_to_point(p['LR'],np.linalg.inv(T)),\
    }
        plant_loc_in_heatmap.append(heatmap_loc)

    return plant_loc_in_heatmap

def mark_plant_on_heatmap(plants,heatmap_main):
    
    heatmap = heatmap_main.copy()

    for p in plants:

        p1 = (int(p['UL'][0]),int(p['UL'][1]))
        p2 = (int(p['LR'][0]),int(p['LR'][1]))

        cv2.rectangle(heatmap,p1,p2,(0,0,255),4)

    return heatmap

def paint_plant_bboxes(pcd,heatmap,plant_bb_heatmap,pcd_coords):

    OI_UL = [0,0]
    OI_UR = [heatmap.shape[1],0]
    OI_LL = [0,heatmap.shape[0]]
    OI_LR = [heatmap.shape[1],heatmap.shape[0]]

    src = np.array([OI_UL,OI_UR,OI_LL,OI_LR])
    dst = np.array([pcd_coords['UL'],pcd_coords['UR'],pcd_coords['LL'],pcd_coords['LR']])

    # H,_ = cv2.findHomography(src,dst)
    H = get_full_transformation(src,dst)

    # UL_Test = apply_T_to_point(OI_UL,H)
    # UR_Test = apply_T_to_point(OI_UR,H)
    # LL_Test = apply_T_to_point(OI_LL,H)
    # LR_Test = apply_T_to_point(OI_LR,H)

    # print(pcd_coords['UL'],UL_Test)
    # print(pcd_coords['UR'],UR_Test)
    # print(pcd_coords['LL'],LL_Test)
    # print(pcd_coords['LR'],LR_Test)

    points = np.array(pcd.points)
    # print(points[:10,:2])

    pcd.paint_uniform_color([1,1,1])
    for p in plant_bb_heatmap:

        UL = apply_T_to_point(p['UL'],H)
        LR = apply_T_to_point(p['LR'],H)
        cond = (points[:,0]>=UL[0]) & (points[:,0]<=LR[0]) & (points[:,1]>=LR[1]) & (points[:,1]<=UL[1])
        
        cond1 = list(np.where(points[:,0]>=UL[0])[0])
        cond2 = list(np.where(points[:,0]<=LR[0])[0])
        cond3 = list(np.where(points[:,1]>=LR[1])[0])
        cond4 = list(np.where(points[:,1]<=UL[1])[0])
        
        final = list(set(cond1)&set(cond2)&set(cond3)&set(cond4))
        print(final)

        np.asarray(pcd.colors)[final, :] = (0,1,0)

    return pcd
        
