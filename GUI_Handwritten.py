import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import PIL
import tkinter as tk
from tkinter.ttk import *
from PIL import Image, ImageTk, ImageDraw
import tkinter.filedialog as fd
import cv2
import skimage.feature as skf
import numpy as np
import joblib

#Loadmodel
model = joblib.load("D://DetectNumber//model_cut_image.sav")

#Predict
def predict(img_pre):
    img = np.array(img_pre,dtype=np.uint8) 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY )
    # Find contours in the image
    ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in ctrs:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x,y, x+w,y+h])
    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    
    roi = im_th[top-1:bottom+1,left-1:right+1]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi = roi.reshape(-1)
    # Pridict digit
    predictions = model.predict([roi]) 
    
    return predictions

# Bắt sự kiện cho nút mở file, mở thư mục và chọn ảnh để load
def openFolderImage():
    # global cho main_image
    global main_image
    name = fd.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes = [("Image Files", "	*.apng;*.jpg;*.jpeg;*.png;*.gif;*.svg;*.webp"),("All Files", "*.*")])
    img = Image.open(name).convert('RGB') 
    main_image = img 
    
    # Dự đoán chữ số
    pred = predict(main_image)
    lbl_pred.configure(text=pred)
    
    # Load ảnh lên label hiển thị ảnh trong giao diện chính
    img = ImageTk.PhotoImage(img.resize((250, 250)))
    lbl.configure(image=img)
    lbl.image = img 

# Bắt sự kiện cho nút mở vẽ, mở một window mới và bắt đầu vẽ
def Window_Paint():
    # Function cho phép vẽ 
    def paint(event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        canvas.create_oval((x1, y1, x2, y2), fill='white', width=15)
        #  --- PIL
        draw.line((x1, y1, x2, y2), fill='white', width=15)
    
    # Tạo một cữa sổ mới tk.Toplevel()
    window_paint = tk.Toplevel()
    window_paint.title("Paint Application")
    window_paint.geometry("%dx%d+%d+%d" % (400,425,550, 150))
    
    # Nhãn lời chào trong Window Paint
    label_greeting_paint = tk.Label(window_paint,text="Paint Appication", font=('calibri',19, 'bold'))
    label_greeting_paint.place(x=18, y=10,width=380,height=30)
    
    # Tạo và setup khung để vẽ
    canvas = tk.Canvas(window_paint, height=310, width=310,borderwidth=2, relief="groove")
    canvas.bind('<B1-Motion>', paint)
    canvas.place(x=40, y=50)

    # Lấy ảnh đã vẽ
    pil_image = PIL.Image.new('RGB', (300, 300), 'black')
    draw = ImageDraw.Draw(pil_image)
    
    # Bắt sự kiện khi nhấn nút close
    def clearAll():
        canvas.delete("all")
        lbl.config(image="")
        lbl_pred.config(text="")
        window_paint.destroy()

    # Bắt sự kiện khi nhấn nút ok
    def ok():
        global main_image
        #Lấy ảnh đã vẽ
        img = pil_image
        main_image = img
        #Dự đoán chữ số
        pred = predict(main_image)
        lbl_pred.configure(text=pred)
        # Load ảnh lên label hiển thị ảnh trong giao diện chính
        img = ImageTk.PhotoImage(img.resize((250, 250)))
        lbl.configure(image=img)
        lbl.image = img
        
        window_paint.destroy()

    # Tạo Frame chứ 2 nút OK và Clode
    frm1 = Frame(window_paint)
    frm1.place(x=120, y=380,width=200,height=30)
    btn_ok = Button(frm1, text="OK", command=lambda: ok())
    btn_ok.place(x=0, y=0,width=80,height=30)
    btn_clear = Button(frm1, text="Close", command=lambda: clearAll())
    btn_clear.place(x=100, y=0,width=80,height=30)

# Bắt sự kiện mở một window mới chứa đặc trưng của ảnh
def Window_Img_Feature():
    def get_image_hog(lbl_image_hog,pixel):
        # Chuyển đổi ảnh PIL thành ảnh OpenCV
        img = np.array(main_image) 
        # Chuyển đổi ảnh từ không gian màu này sang không gian màu khác
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Tính toán đặc trưng HOG 
        hog_features, hog_image = skf.hog(img, orientations=4, pixels_per_cell=(pixel,pixel),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",visualize=True) 
        # Load ảnh lên label hiển thị ảnh trong giao diện chính
        img = Image.fromarray((hog_image*255).astype(np.uint8))
        img = ImageTk.PhotoImage(img.resize((120, 120)))
        lbl_image_hog.configure(image=img)
        lbl_image_hog.image = img
        
    def get_image_sift(lbl_image_sift,flag):
        # Chuyển đổi ảnh PIL thành ảnh OpenCV
        img = np.array(main_image) 
        #COLOR_HSV2RGB
        gray = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        # Trích xuất đặc trưng bằng SIFT
        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(img,kp,gray,flags=flag)
        # Load ảnh lên label hiển thị ảnh trong giao diện chính
        img = Image.fromarray((img*255).astype(np.uint8))
        img = ImageTk.PhotoImage(img.resize((120, 120)))
        lbl_image_sift.configure(image=img)
        lbl_image_sift.image = img
    
    # Matches 2 ảnh lại vs nhau , trích xuất đặc trưng
    def get_image_sift_double():
        # Chuyển đổi ảnh PIL thành ảnh OpenCV
        img1 = np.array(main_image) 
        # Tạo ảnh 2 tương tự ảnh 1
        img2 = img1.copy()
        img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # Trích xuất đặc trưng bằng SIFT
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        matches = bf.match(descriptors1,descriptors2)
        matches = sorted(matches,key=lambda x:x.distance)
        img3 = cv2.drawMatches(img1, keypoints1, img2,keypoints2,matches[0:100],img2, flags=cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
        # Load ảnh lên label hiển thị ảnh trong giao diện
        img3 = Image.fromarray((img3*255).astype(np.uint8))
        img3 = ImageTk.PhotoImage(img3.resize((250, 120)))
        lbl_image_sift3.configure(image=img3)
        lbl_image_sift3.image = img3
        
    def get_iamge_canny(lbl_image_canny, sigma):
        # Chuyển đổi ảnh PIL thành ảnh OpenCV
        img = np.array(main_image) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Chuẩn hóa hình ảnh thang độ xám
        normed_gray = (img - img[:].mean())/img[:].std()
        # Phát hiện các cạnh bằng thuật toán Canny
        edges = skf.canny(normed_gray, low_threshold=0.05, sigma=sigma)
        # Load ảnh lên label hiển thị ảnh trong giao diện
        img = Image.fromarray((edges*255).astype(np.uint8))
        img = ImageTk.PhotoImage(img.resize((120, 120)))
        lbl_image_canny.configure(image=img)
        lbl_image_canny.image = img
        
    # Tạo một cữa sổ mới tk.Toplevel()
    window_feature = tk.Toplevel()
    window_feature.title("Img Feature")
    window_feature.geometry("%dx%d+%d+%d" % (550,550,400, 70))
    
    # Lời chào trong Windown feature
    label_greeting_feature = tk.Label(window_feature,text="Image Feature Extraction", font=('calibri',19, 'bold'))
    label_greeting_feature.place(x=0, y=10,width=600,height=30)

    # Frame HOG
    frm_hog = Frame(window_feature)
    frm_hog.place(x=20, y=60,width=530,height=155)
    # Nhãn HOG
    label_hog = Label(frm_hog, text='HOG (Histogram of Oriented Gradients)',font=('Arial', 12))
    label_hog.place(x=0, y=0,width=580,height=20)
    # Label hiên thị ảnh HOG1
    lbl_image_hog1 = Label(frm_hog,borderwidth=2, relief="groove")
    lbl_image_hog1.place(x=0, y=30,width=120,height=120)
    get_image_hog(lbl_image_hog1,3)
    # Label hiên thị ảnh HOG2
    lbl_image_hog2 = Label(frm_hog,borderwidth=2, relief="groove")
    lbl_image_hog2.place(x=130, y=30,width=120,height=120)
    get_image_hog(lbl_image_hog2,5)
    # Label hiên thị ảnh HOG3
    lbl_image_hog3 = Label(frm_hog,borderwidth=2, relief="groove")
    lbl_image_hog3.place(x=260, y=30,width=120,height=120)
    get_image_hog(lbl_image_hog3,7)
    # Label hiên thị ảnh HOG4
    lbl_image_hog4 = Label(frm_hog,borderwidth=2, relief="groove")
    lbl_image_hog4.place(x=390, y=30,width=120,height=120)
    get_image_hog(lbl_image_hog4,9)
    
    # Frame SIFT
    frm_sift = Frame(window_feature)
    frm_sift.place(x=20, y=220,width=580,height=155)
    # Nhãn SIFT
    label_sift = Label(frm_sift, text='SIFT (Scale-Invariant Feature Transform)',font=('Arial', 12))
    label_sift.place(x=0, y=0,width=580,height=20)
    
    lbl_image_sift1 = Label(frm_sift,borderwidth=2, relief="groove")
    lbl_image_sift1.place(x=0, y=30,width=120,height=120)
    get_image_sift(lbl_image_sift1,cv2.INTER_CUBIC)
    
    lbl_image_sift2 = Label(frm_sift,borderwidth=2, relief="groove")
    lbl_image_sift2.place(x=130, y=30,width=120,height=120)
    get_image_sift(lbl_image_sift2,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    lbl_image_sift3 = Label(frm_sift,borderwidth=2, relief="groove")
    lbl_image_sift3.place(x=260, y=30,width=250,height=120)
    get_image_sift_double()
    
    # Frame Canny
    frm_canny = Frame(window_feature)
    frm_canny.place(x=20, y=380,width=580,height=155)
    
    label_crap_cut = Label(frm_canny, text='CANNY (Canny Edge Detector)',font=('Arial', 12))
    label_crap_cut.place(x=0, y=0,width=580,height=20)
    
    lbl_image_canny1 = Label(frm_canny,borderwidth=2, relief="groove")
    lbl_image_canny1.place(x=0, y=30,width=120,height=120)
    get_iamge_canny(lbl_image_canny1,0.5)
    
    lbl_image_canny2 = Label(frm_canny,borderwidth=2, relief="groove")
    lbl_image_canny2.place(x=130, y=30,width=120,height=120)
    get_iamge_canny(lbl_image_canny2,0.8)
    
    lbl_image_canny3 = Label(frm_canny,borderwidth=2, relief="groove")
    lbl_image_canny3.place(x=260, y=30,width=120,height=120)
    get_iamge_canny(lbl_image_canny3,1.1)
    
    lbl_image_canny4 = Label(frm_canny,borderwidth=2, relief="groove")
    lbl_image_canny4.place(x=390, y=30,width=120,height=120)
    get_iamge_canny(lbl_image_canny4,1.4)

#Thiết kế giao diện chính
window = tk.Tk()
window.title('name')
# kích thước 400x425 ở vị trí (500,150)
window.geometry('400x425+550+150')

# Nhãn chào trong giao diện chính
label_greeting = tk.Label(window,text="Handwritten Digit Recognition", font=('calibri',19, 'bold'))
label_greeting.place(x=18, y=12,width=380,height=30)

# Nhãn hiển thị ảnh load từ máy tính hay vẽ
lbl = Label(window,borderwidth=2, relief="groove")
lbl.place(x=70, y=60,width=250,height=250)
# Ma trận kích thước (250,250,3) lưu trữ ảnh load lên từ folder hoặc vẽ , dùng để trích xuất đặc trưng ảnh
main_image = np.zeros((250, 250, 3))

# Frame dự đoán chữ số và % độ chính xác
frm_pred = Frame(window)
frm_pred.place(x=70, y=320,width=300,height=50)
lbl2 = Label(frm_pred, text='Predict is: ',font=('Arial', 14))
lbl2.place(x=60, y=0,width=150,height=50)
lbl_pred = Label(frm_pred,text='9',font=('Arial', 14))
lbl_pred.place(x=150, y=0,width=150,height=50)

# Frame có 4 nút, mở file, mở vẽ, đặc trưng ảnh, thoát
frm = Frame(window)
frm.place(x=20, y=380,width=400,height=30)
btn1 = Button(frm, text="Open File", command=lambda: openFolderImage())
btn1.place(x=0, y=0,width=80,height=30)
btn2 = Button(frm, text="Open Paint", command=lambda: Window_Paint())
btn2.place(x=100, y=0,width=80,height=30)
btn3 = Button(frm, text="Img Feature",command=lambda: Window_Img_Feature())
btn3.place(x=200, y=0,width=80,height=30)
btn4 = Button(frm, text="Exit", command=lambda: exit())
btn4.place(x=300, y=0,width=60,height=30)

window.mainloop()