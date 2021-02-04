from django.shortcuts import render

from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required
# from .models import *
from django.contrib.auth.models import User
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from visualapp.forms import *
from visualapp.models import *

import csv
import xlwt

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorboard.plugins import projector
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def index(request):
    return render(request, 'index.html')




@login_required(login_url='/signin/')
def dashboard(request):
    if request.method == 'POST':
        print('amitaaaaaaaaaaaaa')
        excel_file = request.FILES["fileToUpload"]
        print(type(excel_file),"excel_fileexcel_fileexcel_fileexcel_file")
        print(excel_file,"excel_fileexcel_fileexcel_fileexcel_fileexcel_file")
        excel_test=str(excel_file)
        if excel_test[-1]=="s" and excel_test[-2]=="l" and excel_test[-3]=="x":
            df=pd.read_excel(excel_file)
        if excel_test[-1]=="v" and excel_test[-2]=="s" and excel_test[-3]=="c":
            df=pd.read_csv(excel_file)
        if excel_test[-1]=="x" and excel_test[-2]=="s" and excel_test[-3]=="l" and excel_test[-4]=="x":
            df=pd.read_excel(excel_file)

                ## Get working directory
        PATH = os.getcwd()
        ## Path to save the embedding and checkpoints generated
        # os.mkdir('logs')
        LOG_DIR = os.path.join(PATH, 'logs_sampleData4')
        # # Read data from excel to become dataframe

        ## Load data
        # df = pd.read_excel(os.path.join(PATH, "sampleData3.xlsx"))

        # # Some samples in excel

        df.head()

        # # Find keys of columns

        df.columns

        # # Get vector data and fill nan value by 0

        df_train = df[df.columns[2:]].fillna(0)
        df_train.head()

        # # Create metadata file which contain index of samples

        ## Load the metadata file. Metadata consists your labels. This is optional. Metadata helps us visualize(color) different clusters that form t-SNE
        metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
        metadata = df[df.columns[0]]
        metadata

        # # Write them to hard disk( because input metadata of tensorboad is file)

        with open(metadata_path,'w+') as f:
            f.write('{}\t{}\n'.format('id', 'F'))
            for index, label in enumerate(metadata):
                f.write('{}\t{}\n'.format(label, df_train['l'][index]))

        # # Create PCA with n_components is size of sample vectors

        pca = PCA( n_components=len(df_train.columns)-2,
                 random_state = 123,
                 svd_solver='auto')

        # # Fit data to PCA and create variable from data to tensorflow

        df_pca = pd.DataFrame(pca.fit_transform(df_train))
        df_pca = df_pca.values
        ## TensorFlow Variable from data
        tf_data = tf.Variable(df_pca)

        # # Open session in tensorflow and add data and metadata to tensorboard

        ## Running TensorFlow Session
        with tf.compat.v1.Session() as sess:
            # config saving checkpoint of pca
            saver = tf.compat.v1.train.Saver([tf_data])
            # run graph by Session.run
            sess.run(tf_data.initializer)
            # save checckpoint to hard disk
            saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
            # config projector which we will use to visualize to tensorboard
            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            # add name of tensorboard
            embedding.tensor_name = tf_data.name
            # Link this tensor to its metadata(Labels) file
            embedding.metadata_path = metadata_path
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

        return render(request,'sucess.html')
    return render(request,'userdashboard.html')

@login_required(login_url='/signin/')

def updateboard(request):
    os.system('fuser -k 6006/tcp')
    os.system('nohup tensorboard --logdir=/home/ubuntu/visual/logs_sampleData4/ --host 0.0.0.0 --port 6006 &')
    os.system('\r\n')
    # return HttpResponse("tensorboard is updated succesfully")
    return render(request,'final.html')



def signup_page(request):
    if request.method=="POST":
        form=signupform(request.POST)
        if form.is_valid():
            name=request.POST["Name"]
            email=request.POST["Email"]
            password=request.POST["Password"]
            Firstname=request.POST["Firstname"]
            lastname=request.POST["lastname"]
            user = User.objects.create_user(username=name,email=email,password=password,first_name=Firstname,last_name=lastname)
            user.save()
            return redirect('/signin/')
    else:
        form = signupform()
        print("notdshksfdhjsdfhsdfahlsafd")
    return render(request,'signup.html',{"form":form})





def login_user(request):
    if request.user.is_authenticated:
        print("Logged in")
        return redirect("/userdash/")
    else:
        print("Not logged in")

    if request.method == 'POST':
        form = loginform(request.POST)
        if form.is_valid():
            username = request.POST.get('Username')
            print(username)
            password = request.POST.get('Password')
            print(password)
            user = authenticate(username=username, password=password)
            if user:
                print("yesssssssssssssssss")
                login(request,user)
                return redirect("/userdash/")

    else:
        form = loginform()
        print("not")
    return render(request, 'signin.html', {"form": form})

@login_required(login_url='/signin/')
def user_logout(request):
    logout(request)
    return redirect('/signin/')


