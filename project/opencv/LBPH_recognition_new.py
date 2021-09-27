#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
import time
# from gtts import gTTS
import os
import shutil
import motor
import em
import receive

# In[25]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")  # Object of face detector
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

roi_gray = []

#Removes parts of the sides of the face
#This is done so that the algorithm has to work with only the relevant/ most important part of the image
def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:                                        #Trims parts of the face
        w_rm = int(0.2 * w / 2)
        faces.append(image[y : y + h, x + w_rm :  x + w - w_rm])

    return faces                                                            #Returns co-ordinates of the face

def save_faces(frame,faces,folder,counter):
    cut_face = cut_faces(frame, faces)

    face_bw = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)

    face_bw_eq = cv2.equalizeHist(face_bw)
    face_bw_eq = cv2.resize(face_bw_eq, (256, 256), interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Face Recogniser', face_bw_eq)


    cv2.imwrite(folder + '/' + str(counter) + '.png',
                face_bw_eq)
    print('Images Saved:' + str(counter))
    cv2.imshow('Saved Face', face_bw_eq)
# In[26]:


#Adds a new person to the dataset and creates a separate folder for them
def add_person2(e9):
    person_name = e9.get()     #Get the name of the new person

    folder = 'people_folder' +'/'+ person_name

    if not os.path.exists(folder):                                          #Find the if the data for the given person already exists

        os.makedirs(folder)                                                    # Makes the new folder for saving the photos

        video = cv2.VideoCapture(0)

        counter = 1
        timer = 0

        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)

        while counter < 21:
            _, frame = video.read()


            if counter == 1:
                time.sleep(6)
            else:
                time.sleep(1)

            faces = face_cascade.detectMultiScale(frame,1.3,4)
            profile_faces = profile_cascade.detectMultiScale(frame,1.3,4)

            if len(faces):
                save_faces(frame,faces,folder,counter)
                counter += 1
            elif len(profile_faces):
                save_faces(frame,profile_faces,folder,counter)
                counter += 1

            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)

    else:
        print("This name already exists.")                                  # If the person already exists


# In[27]:


#Does the face recognition in real time
#Pressing ESC closes the live recognition
def live():

    cv2.namedWindow('Predicting for')
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people_folder")]
    threshold = 37


    for i, person in enumerate(people):
        print(person)
        labels_dic[i] = person

        for image in os.listdir("people_folder/" + person):
            print(image)
            images.append(cv2.imread('people_folder/'+person+'/'+ image, 0))
            labels.append(i)

    labels = np.array(labels)

    #rec_eig = cv2.face.EigenFaceRecognizer_create()
    rec_lbhp = cv2.face.LBPHFaceRecognizer_create()
    rec_lbhp.train(images, labels)

    cv2.namedWindow('face')
    webcam = cv2.VideoCapture(0)
    k_run = 0
    u_run = 0
    while True:
        msg = receive.r()
        if msg[:-3].upper()=='Y':
            motor.run_motor()
            
        _, frame = webcam.read()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        if faces is None:
            faces = profile_cascade.detectMultiscale(frame,1.3,5)

        if len(faces):
            cut_face = cut_faces(frame, faces)

            face = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, (256, 256), interpolation = cv2.INTER_CUBIC)

            cv2.imshow('face', face)

            collector = cv2.face.StandardCollector_create()
            rec_lbhp.predict_collect(face, collector)
            conf = collector.getMinDist()

            print('Confidence ', conf)
            pred = collector.getMinLabel()
            txt = ''

            if conf < threshold:
                u_run = 0
                txt = labels_dic[pred].upper()
                k_run += 1
                if k_run>=10:
                    motor.run_motor()
            else:
                k_run = 0
                txt = 'Uknown'
                u_run += 1
                if u_run>=10:
                    em.SendMail(face)

            cv2.putText(frame, txt,
                        (faces[0][0], faces[0][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

            print(faces)
            cv2.rectangle(frame, (faces[0][0], faces[0][1]),(faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]), (255, 255, 0), 8)#Makes rectangle around face

            cv2.putText(frame,"ESC to exit", (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)

        cv2.imshow("Live", frame)

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break


# In[28]:


def delete2(e10):
    person_name = e10.get()
    folder = 'people_folder' +'/'+ person_name
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        print("User Does not exists")


# In[30]:


from tkinter import *
root=Tk()
root.geometry("700x700")
root.resizable(0,0)

def pr(e9):
    s9 = e9.get()
    print(s9)


def delete():
    f4=Frame(root,bg="#091e42")
    f4.place(x=0,y=0,width=700,height=700)
    l10=Label(f4,text="Enter name",font=("arial",25))
    l10.grid(row=0,column=0,pady=25,sticky=W)
    e10=Entry(f4,font=("arial",25))
    e10.grid(row=0,column=1,pady=25)

    b10=Button(f4,text="start",font=("arial",25),command=lambda :delete2(e10))
    b10.grid(row=2,column=0,pady=25,columnspan=2)

    b11=Button(f4,text="back",font=("arial",25),command=show2)
    b11.grid(row=3,column=0,pady=25,columnspan=2)






def add_person():
    f3=Frame(root,bg="#091e42")
    f3.place(x=0,y=0,width=700,height=700)
    l9=Label(f3,text="whats your name",font=("arial",25))
    l9.grid(row=0,column=0,pady=25,sticky=W)
    e9=Entry(f3,font=("arial",25))
    e9.grid(row=0,column=1,pady=25)


    b9=Button(f3,text="start",font=("arial",25),command=lambda :add_person2(e9))
    b9.grid(row=2,column=0,pady=25,columnspan=2)

    b8=Button(f3,text="back",font=("arial",25),command=show2)
    b8.grid(row=3,column=0,pady=25,columnspan=2)

def exit():
    root.destroy()






def show2():

    f2=Frame(bg="#091e42",)
    f2.place(x=0,y=0,width=700,height=700)
    l3=Label(f2,text="welcome",font=("arial",25))
    l3.grid(row=0,column=0,pady=25,sticky=W)
    b2=Button(f2,text="add a face",font=("arial",25),command=add_person)
    b2.place(x=350,y=120,anchor=CENTER)
    b3=Button(f2,text="delete face",font=("arial",25),command=delete)
    b3.place(x=350,y=200,anchor=CENTER)
    b4=Button(f2,text="settings",font=("arial",25))
    b4.place(x=350,y=280,anchor=CENTER)
    b5=Button(f2,text="go live",font=("arial",25),command=live)
    b5.place(x=350,y=360,anchor=CENTER)
    b6=Button(f2,text="exit",font=("arial",25),command=exit)
    b6.place(x=350,y=440,anchor=CENTER)

def show(e1,e2,f1):
    s1=e1.get()
    s2=e2.get()
    print("i m show")
    if (s1=="face" and s2=="f"):
        show2()
    else:

        l4=Label(f1,text="incorrect password",font=("arial",22))
        l4.grid(row=4,column=0,pady=25,sticky=W)
        b7=Button(f1,text="retry",font=("arial",25),command=house)
        b7.grid(row=5,column=0,pady=25,columnspan=2,)










def house():
    s1="face"
    s2="f"
    f1=Frame(bg="#091e42")
    f1.place(x=0,y=0,width=700,height=700)

    l1=Label(f1,text="enter user name",font=("arial",25))
    l1.grid(row=0,column=0,pady=25,sticky=W)
    e1=Entry(f1,font=("arial",25))
    e1.grid(row=0,column=1,pady=25)


    l2=Label(f1,text="enter Password",font=("arial",25))
    l2.grid(row=1,column=0,pady=25)
    e2=Entry(f1,show='*',font=("arial",25))
    e2.grid(row=1,column=1,pady=25)


    b1=Button(f1,text="login",font=("arial",25),command=lambda: show(e1,e2,f1))
    b1.grid(row=2,column=0,pady=25,columnspan=2)










    print("i m fool")

    root.mainloop()


house()



cv2.destroyAllWindows()


# In[ ]:

#deleted msg/name doesnt exist
#



# In[ ]:





# In[ ]:
