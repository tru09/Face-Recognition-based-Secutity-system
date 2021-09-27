import cv2
import os

def face_save(name):
    vid = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    path = os.getcwd()
    directory = os.path.join(path,'training',name)
    try:
        os.makedirs(directory)
    except FileExistsError:
        print("Error")
    os.chdir(directory)
    count = 0

    while True:
        ret,img = vid.read()
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=4)
        profiles = profile_cascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=4)

        # cv2.imshow('img',img)
        if faces is None:
            continue
        else:
            for (x, y, w, h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
                crop = img[y:y+h,x:x+w]
                resize_img = cv2.resize(crop,(256,256))
                s = '{}-{}.png'.format(name,count)
                # cv2.imwrite(s,resize_img)
                count+=1

                cv2.imshow('cam_frontal',frame)

        if profiles is None:
            continue
        else:
            for (x,y,w,h) in profiles:
                p_frame = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
                cv2.imshow('cam_profile',p_frame)


        if cv2.waitKey(10) == 27:
            break

    vid.release()
    cv2.destroyAllWindows()

# i = int(input("Do you want to register a user?...press 1"))

# if i==1:
    # name = input("Enter the name")
face_save("Shashank")

# face_save('s2')
