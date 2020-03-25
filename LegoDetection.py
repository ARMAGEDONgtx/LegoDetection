import cv2 as cv
import numpy as np
import os
import json
import sys

def callb(x):
    pass

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def how_many(file_name, json_name):
    img_name = file_name.split('.',1)[0]
    with open(json_name) as json_file:
        data = json.load(json_file)
    suma = 0
    for p in data[img_name]:
        suma = suma + 1
    return suma

def how_many_by_color(file_name, color, json_name):
    img_name = file_name.split('.',1)[0]
    with open(json_name) as json_file:
        data = json.load(json_file)
    suma = 0
    for p in data[img_name]:
        suma = int(p[color]) + suma
    return suma

def get_boundry(img, quant):
    # Find Canny edges
    boxes = []
    edged = cv.Canny(img, 500, 150)
    contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntssort = sorted(contours, key=lambda x: cv.contourArea(x))
    # count as many as json file says
    cpy = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    cntssort = cntssort[-quant:]
    for cnt in cntssort:
        if cv.contourArea(cnt) > 50:
            # compute the center of the contour
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.drawContours(cpy, [cnt], -1, (0, 255, 0), 3)
            cv.circle(cpy, (cX, cY), 7, (0, 0, 255), -1)
            cv.putText(cpy, str(cv.contourArea(cnt)), (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # get the min area rect
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            boxes.append(box)
            # draw a red 'nghien' rectangle
            cv.drawContours(cpy, [box], 0, (0, 0, 255))
            for k in box:
                cv.putText(cpy, str(k[0])+ ', '+str(k[1]), (k[0], k[1]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    return boxes, cpy

def get_middles(img,quant):
    edged = cv.Canny(img, 500, 150)
    contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntssort = sorted(contours, key=lambda x: cv.contourArea(x))
    centers = []
    # count as many as json file says
    cpy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cntssort = cntssort[-quant:]
    for cnt in cntssort:
        if cv.contourArea(cnt) > cv.getTrackbarPos('area', 'window'):
            # compute the center of the contour
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX,cY))
    return centers

def give_reds(img, circles=False):
    if circles == True:
        th = 110
        morp = 3
    else:
        th = 120
        morp = 8
    hsv = cv.cvtColor(img.copy(),cv.COLOR_BGR2HSV)
    cpy_blur = cv.medianBlur(hsv, 1)
    mask_red = cv.inRange(cpy_blur, (0, 40, 40), (12, 255, 255))
    target = cv.bitwise_and(img, img, mask=mask_red)
    # covert img with mask to gray scale
    target_gray = cv.cvtColor(target, cv.COLOR_HSV2BGR)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)
    # threshold
    ret, th1 = cv.threshold(target_gray, th, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    out =  cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel, iterations=morp)
    if circles == False:
        out = cv.erode(out, kernel, iterations=1)
        out = cv.dilate(out, kernel, iterations=2)
    return out

def give_yellows(img, circles = False):
    if circles == True:
        slow = 100
        vlow = 100
        morp = 3
    else:
        slow = 125
        vlow = 0
        morp = 8
    hsv = cv.cvtColor(img.copy(),cv.COLOR_BGR2HSV)
    cpy_blur = cv.medianBlur(hsv, 1)
    mask_yellow = cv.inRange(cpy_blur, (15, slow, vlow), (30, 255, 255))
    target = cv.bitwise_and(img, img, mask=mask_yellow)
    # covert img with mask to gray scale
    target_gray = cv.cvtColor(target, cv.COLOR_HSV2BGR)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)
    # threshold
    ret, th1 = cv.threshold(target_gray, 132, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    out =  cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel, iterations=morp)
    if circles == False:
        out = cv.erode(out, kernel, iterations=1)
        out = cv.dilate(out, kernel, iterations=2)
    return out

def give_blues(img, circles = False):
    if circles == True:
        morp = 3
    else:
        morp = 7
    hsv = cv.cvtColor(img.copy(),cv.COLOR_BGR2HSV)
    cpy_blur = cv.medianBlur(hsv, 1)
    mask_blue = cv.inRange(cpy_blur, (90, 80, 80), (130, 255, 255))
    target = cv.bitwise_and(img, img, mask=mask_blue)
    # covert img with mask to gray scale
    target_gray = cv.cvtColor(target, cv.COLOR_HSV2BGR)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)
    # threshold
    ret, th1 = cv.threshold(target_gray, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    out =  cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel, iterations=morp)
    if circles == False:
        out = cv.erode(out, kernel, iterations=1)
    return out

def give_whites(img,th,e,d):
    hsv = cv.cvtColor(img.copy(),cv.COLOR_BGR2HSV)
    cpy_blur = cv.medianBlur(hsv, 1)
    mask_blue = cv.inRange(cpy_blur, (35, 8, 180), (179, 40, 255))
    target = cv.bitwise_and(img, img, mask=mask_blue)
    cv.imshow('reds',target)
    # covert img with mask to gray scale
    target_gray = cv.cvtColor(target, cv.COLOR_HSV2BGR)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)
    # threshold
    ret, th1 = cv.threshold(target_gray, th, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    out = cv.erode(th1, kernel, iterations=1)
    out =  cv.morphologyEx(out, cv.MOR, kernel, iterations=d)
    out = cv.erode(out, kernel, iterations=e)
    cv.imshow('out', out)
    return out

def give_grays(img):
    hsv = cv.cvtColor(img.copy(),cv.COLOR_BGR2HSV)
    cpy_blur = cv.medianBlur(hsv, 9)
    mask_gray = cv.inRange(cpy_blur, (40, 18, 0), (100, 50, 170))
    target = cv.bitwise_and(img, img, mask=mask_gray)
    # covert img with mask to gray scale
    target_gray = cv.cvtColor(target, cv.COLOR_HSV2BGR)
    target_gray = cv.cvtColor(target_gray, cv.COLOR_BGR2GRAY)
    # threshold
    ret, th1 = cv.threshold(target_gray, 0, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    out =  cv.morphologyEx(th1, cv.MORPH_CLOSE, kernel, iterations=9)
    out = cv.erode(out, kernel, iterations=2)
    out = cv.dilate(out, kernel, iterations=2)
    #cv.imshow('out', out)
    return out

def esitmate_whites(img):
    # list to store radius of all detected coins
    out = np.zeros(img.shape, dtype=np.uint8)
    out.fill(255)
    inp = img.copy()
    coins_r = []
    # convert to gray for circle detection
    img_small = cv.medianBlur(inp, 5)
    gray_img = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)
    # detect circles
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, 6, param1=20, param2=12, minRadius=3, maxRadius=7)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.rectangle(out, (i[0] - 12, i[1] - 12), (i[0] + 12, i[1] + 12), (0, 0, 0), -1)
    out = cv.cvtColor(out,cv.COLOR_BGR2GRAY)
    return out

def my_add4(img1,img2,img3,img4):
    s = img1.shape
    out = img1.copy()
    for y in range(0, s[0]):
        for x in range(0, s[1]):
            if img1[y][x] == 255 or img2[y][x] == 255 or img3[y][x] == 255 or img4[y][x] == 255:
                out[y][x] = 255
            else:
                out[y][x] = 0
    return out

def my_add4_2(img1,img2,img3,img4):
    out = img1.copy()
    out[img2 == 255] = 255
    out[img3 == 255] = 255
    out[img4 == 255] = 255
    return out

def my_add3(img1,img2,img3):
    s = img1.shape
    out = img1.copy()
    for y in range(0, s[0]):
        for x in range(0, s[1]):
            if img1[y][x] == 255 or img2[y][x] == 255 or img3[y][x] == 255:
                out[y][x] = 255
            else:
                out[y][x] = 0
    return out

def my_add3_2(img1,img2,img3):
    out = img1.copy()
    out[img2 == 255] = 255
    out[img3 == 255] = 255
    return out

def erease_from_main(img1,er):
    s = er.shape
    out = img1.copy()
    for y in range(0, s[0]):
        for x in range(0, s[1]):
            if er[y][x] == 255:
                out[y][x] = (0,0,0)
    return out

def erease_from_main2(img1,er):
    out = img1.copy()
    out[er == 255] = (0,0,0)
    return out

def guess_all(img_without):
    # list to store radius of all detected coins
    out = img_without.copy()
    coins_r = []
    # convert to gray for circle detection
    img_small = cv.medianBlur(out, 5)
    gray_img = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)
    #detect circles
    circles = cv.HoughCircles(gray_img, cv.HOUGH_GRADIENT, 1, 6, param1=20, param2=12, minRadius=3, maxRadius=7)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.rectangle(gray_img, (i[0] - 12, i[1] - 12), (i[0] + 12, i[1] + 12), (0, 0, 0), -1)
    kernel = np.ones((3, 3), np.uint8)
    out = cv.morphologyEx(gray_img, cv.MORPH_OPEN, kernel, iterations=12)
    out = cv.erode(out, kernel, iterations=1)
    ret, th1 = cv.threshold(out, 10, 255, cv.THRESH_BINARY)
    return th1

def ray_tracing(pnt,poly):
    x = pnt[0]
    y = pnt[1]
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

class lego_block:

    def __init__(self, box):
        self.has_red = 0
        self.has_blue = 0
        self.has_yellow = 0
        self.has_grey = 0
        self.has_white = 0
        self.red_circles = 0
        self.yellow_circles = 0
        self.blue_circles = 0
        self.rest_circles = 0
        self.box = box
        self.no = None

    def check_colors(self, centers, color):
        if color == 'red':
            for c in centers:
                h = ray_tracing(c,self.box)
                if h == True:
                    self.has_red = self.has_red + 1
        elif color == 'blue':
            for c in centers:
                h = ray_tracing(c, self.box)
                if h == True:
                    self.has_blue = self.has_blue + 1
        elif color == 'yellow':
            for c in centers:
                h = ray_tracing(c, self.box)
                if h == True:
                    self.has_yellow = self.has_yellow + 1
        elif color == 'grey':
            for c in centers:
                h = ray_tracing(c, self.box)
                if h == True:
                    self.has_grey = self.has_grey + 1
        elif color == 'white':
            for c in centers:
                h = ray_tracing(c, self.box)
                if h == True:
                    self.has_white = self.has_white + 1

    def show(self):
        print('Its lego no ' +str(self.no))
        print('My box - ' + str(self.box))
        print('Has red - ' + str(self.has_red))
        print('Has blue - ' + str(self.has_blue))
        print('Has yellow - ' + str(self.has_yellow))
        print('Has grey - ' + str(self.has_grey))
        print('Has white - ' + str(self.has_white))

    def copy(self):
        lego = lego_block(self.box)
        lego.has_red = self.has_red
        lego.has_blue = self.has_blue
        lego.has_white = self.has_white
        lego.has_grey = self.has_grey
        lego.has_yellow = self.has_yellow
        lego.no = self.no
        return lego

    def count_circles(self, org_image, color):
        copy_img = np.copy(org_image)
        # detect all circles
        if color == 'red':
            minDst = 10
            param1 = 20
            param2 = 10
            minR = 2
            maxR = 8
            blur = 5
        elif color == 'blue':
            minDst = 10
            param1 = 10
            param2 = 10
            minR = 2
            maxR = 8
            blur = 5
        elif color == 'yellow':
            minDst = 10
            param1 = 6
            param2 = 9
            minR = 2
            maxR = 8
            blur = 5
        else:
            minDst = 8
            param1 = 35
            param2 = 13
            minR = 4
            maxR = 9
            blur = 1
        cpy_blur = cv.medianBlur(copy_img, blur)
        circl = cv.HoughCircles(cpy_blur, cv.HOUGH_GRADIENT, 1, minDst, param1=param1, param2=param2, minRadius=minR, maxRadius=maxR)
        cpy_blur = cv.cvtColor(cpy_blur, cv.COLOR_GRAY2BGR)
        if circl is not None:
            circles = np.uint16(np.around(circl))
            for i in circles[0, :]:
                if ray_tracing((i[0], i[1]), self.box):
                    if color == 'red' and self.has_red > 0:
                        self.red_circles = self.red_circles + 1
                    elif color == 'blue' and self.has_blue > 0:
                        self.blue_circles = self.blue_circles + 1
                    elif color == 'yellow' and self.has_yellow > 0:
                        self.yellow_circles = self.yellow_circles + 1
                    elif self.has_white > 0 or self.has_grey >0:
                        self.rest_circles = self.rest_circles + 1
                    #cv.circle(org_image, (i[0], i[1]), i[2], (0, 0, 255), 2)



    def draw_block(self,img):
        # draw a red 'nghien' rectangle
        cv.drawContours(img, [self.box], 0, (0, 0, 255))
        str_to_draw = 'R:{0}/{5}, B:{1}/{6}, Y:{2}/{7}, G:{3}/{8}, W:{4}/{9}'.format(self.has_red, self.has_blue, self.has_yellow, self.has_grey, self.has_white,self.red_circles,self.blue_circles,self.yellow_circles,self.rest_circles,self.rest_circles)
        cv.putText(img, str_to_draw, (self.box[0][0], self.box[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return img

def create_my_legos(boxes,rc,bc,yc,gc,wc):
    legos = []
    no = 0
    for b in boxes:
        lego =  lego_block(b)
        lego.check_colors(rc,'red')
        lego.check_colors(bc,'blue')
        lego.check_colors(yc,'yellow')
        lego.check_colors(gc,'grey')
        lego.check_colors(wc, 'white')
        lego.no = no
        no = no + 1
        #lego.show()
        legos.append(lego)
    return legos

def create_org_legos(file_name, json_name):
    legos = []
    img_name = file_name.split('.', 1)[0]
    with open(json_name) as json_file:
        data = json.load(json_file)
    no = 0
    for p in data[img_name]:
        lego = lego_block(None)
        lego.has_red = int(p['red'])
        lego.has_blue = int(p['blue'])
        lego.has_yellow = int(p['yellow'])
        lego.has_grey = int(p['grey'])
        lego.has_white = int(p['white'])
        lego.no = no
        no = no + 1
        #lego.show()
        legos.append(lego)
    return legos

def connect_legos(my_legos, org_legos):
    def calculete_likelychood(lego1,lego2):
        val = abs(lego1.has_red - lego2.has_red) + abs(lego1.has_blue - lego2.has_blue) + abs(lego1.has_grey - lego2.has_grey) + abs(lego1.has_yellow - lego2.has_yellow) + abs(lego1.has_white - lego2.has_white)
        return val

    def compare_box(box1,box2):
        if box1[0][0] == box2[0][0] and  box1[0][1] == box2[0][1] and box1[1][0] == box2[1][0] and  box1[1][1] == box2[1][1] and box1[2][0] == box2[2][0] and  box1[2][1] == box2[2][1] and box1[3][0] == box2[3][0] and  box1[3][1] == box2[3][1]:
            return True
        else:
            return False

    def match_by_color(copy_my_legos, org_lego):
        tmp = copy_my_legos.copy()
        for ml_cpy in copy_my_legos:
            if (org_lego.has_red > 0 and ml_cpy.has_red == 0) or (org_lego.has_red == 0 and ml_cpy.has_red > 0):
                tmp.remove(ml_cpy)
            elif (org_lego.has_blue > 0 and ml_cpy.has_blue == 0) or (org_lego.has_blue == 0 and ml_cpy.has_blue > 0):
                tmp.remove(ml_cpy)
            elif (org_lego.has_yellow > 0 and ml_cpy.has_yellow == 0) or (org_lego.has_yellow == 0 and ml_cpy.has_yellow > 0):
                tmp.remove(ml_cpy)
            elif (org_lego.has_grey > 0 and ml_cpy.has_grey == 0) or (org_lego.has_grey == 0 and ml_cpy.has_grey > 0):
                tmp.remove(ml_cpy)
            elif (org_lego.has_white > 0 and ml_cpy.has_white == 0) or (org_lego.has_white == 0 and ml_cpy.has_white > 0):
                tmp.remove(ml_cpy)
        return tmp

    my_legos_left = []
    # my legos left to match
    for m_l in my_legos:
        my_legos_left.append(m_l.copy())
    # first level try, connect just by colors
    for org_lego in org_legos:
        #copy legos
        copy_my_legos = []
        for m_l in my_legos_left:
            copy_my_legos.append(m_l.copy())
        copy_my_legos = match_by_color(copy_my_legos,org_lego)
        if len(copy_my_legos) == 1:          # first level try worked
            org_lego.box = copy_my_legos[0].box
            # remove fro my legos left block that has been taken, do it by boundry box comparing
            for mll in my_legos_left:
                if compare_box(copy_my_legos[0].box, mll.box):
                    my_legos_left.remove(mll)
                    break

    # second level try, choose most likely
    for org_lego in org_legos:
        if org_lego.box is None:
            # copy legos
            copy_my_legos = []
            for m_l in my_legos_left:
                copy_my_legos.append(m_l.copy())
            copy_my_legos = match_by_color(copy_my_legos, org_lego)
            if len(copy_my_legos) > 1:
                tmp_dict = {}
                # calculate likelychood for each lego left
                for cml in copy_my_legos:
                    tmp_dict[copy_my_legos.index(cml)] = calculete_likelychood(org_lego,cml)
                best_match_index = min(tmp_dict, key=tmp_dict.get)
                best_match = copy_my_legos[best_match_index]
                org_lego.box = best_match.box
                # remove fro my legos left block that has been taken, do it by boundry box comparing
                for mll in my_legos_left:
                    if compare_box(best_match.box, mll.box):
                        my_legos_left.remove(mll)
                        break

    # third level try, from beggining, just most likely connect
    for org_lego in org_legos:
        if org_lego.box is None:
            # copy once again
            copy_my_legos = []
            for m_l in my_legos_left:
                copy_my_legos.append(m_l.copy())
            tmp_dict = {}
            # calculate likelychood for each lego left
            for cml in copy_my_legos:
                tmp_dict[copy_my_legos.index(cml)] = calculete_likelychood(org_lego, cml)
            best_match_index = min(tmp_dict, key=tmp_dict.get)
            best_match = copy_my_legos[best_match_index]
            org_lego.box = best_match.box
            # remove fro my legos left block that has been taken, do it by boundry box comparing
            for mll in my_legos_left:
                if compare_box(best_match.box, mll.box):
                    my_legos_left.remove(mll)
                    break

def get_json_from_img(my_json,file_name,legos):
    img_name = file_name.split('.', 1)[0]
    circ_list = []
    for l in legos:
        circ_list.append(l.red_circles+l.blue_circles+l.yellow_circles+l.rest_circles)
    my_json[img_name] = circ_list

def write_final_json(my_json, json_name):
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(my_json, f, ensure_ascii=False, indent=4)

# ---------- PROGRAM BEGIN ---------------------------------------------------------------------------------------------
# get arguments
if len(sys.argv) >= 3:
    photo_path = sys.argv[1]
    json_in_path = sys.argv[2]
    json_out_path = sys.argv[3]
else:
    photo_path = './images_directory'
    json_in_path = './input_file/states.json'
    json_out_path = './output_directory/results.json'

my_json = {}
saved_counter = 1
for r,d,f in walklevel(photo_path,level=0):
    for file in f:
        img = cv.imread(photo_path + "/" + file)
        img_small = cv.resize(img, None, fx=0.25, fy=0.25)

        hsv = cv.cvtColor(img_small,cv.COLOR_BGR2HSV)
        r = give_reds(img_small)
        red_centers = get_middles(r,how_many_by_color(file,'red',json_in_path))
        b = give_blues(img_small)
        blue_centers = get_middles(b, how_many_by_color(file, 'blue',json_in_path))
        y = give_yellows(img_small)
        yellow_centers = get_middles(y, how_many_by_color(file, 'yellow',json_in_path))
        g = give_grays(img_small)
        grey_centers = get_middles(g, how_many_by_color(file, 'grey',json_in_path))

        rbyg = my_add4_2(r,b,y,g)
        without = erease_from_main2(img_small,rbyg)
        #image with black elements as lego blocks connected
        end_img = guess_all(without)

        #get quantity from file and extract boudires
        s = how_many(file,json_in_path)
        boxes, bound = get_boundry(end_img,s)
        #cv.imshow('test',without)
        #cv.waitKey()
        white = esitmate_whites(without)
        white_centers = get_middles(white, how_many_by_color(file, 'white',json_in_path))

        #create legos class instances
        my_legos = create_my_legos(boxes,red_centers,blue_centers,yellow_centers,grey_centers,white_centers)
        #create legos based on json
        org_legos = create_org_legos(file, json_in_path)

        #connect legos
        connect_legos(my_legos,org_legos)

        #circle calculation
        rby = my_add3_2(r,b,y)
        without_gw = erease_from_main2(img_small,rby)
        without_gw = cv.cvtColor(without_gw, cv.COLOR_BGR2GRAY)
        r_circ = give_reds(img_small,True)
        b_circ = give_blues(img_small, True)
        y_circ = give_yellows(img_small, True)

        drawn = img_small.copy()
        for ol in org_legos:
            #ol.show()
            ol.count_circles(r_circ, 'red')
            ol.count_circles(b_circ, 'blue')
            ol.count_circles(y_circ, 'yellow')
            ol.count_circles(without_gw, 'rest')
            drawn = ol.draw_block(drawn)

        cv.imshow('test', drawn)

        get_json_from_img(my_json, file,org_legos)
        key = cv.waitKey()

write_final_json(my_json,json_out_path)
cv.destroyAllWindows()