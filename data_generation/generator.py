#usr/bin/python3
import os, random, shutil,math
import numpy as np
import matplotlib.pyplot as plt
from .sim_funcs import *
from multiprocessing import Process
import logging

#NOTE: path_airfoil_database is the path of the airfoil database, which contains the .dat files of airfoils. If you want to use a relative path for that, please give the path relative to the OpenFOAM case folder.

def log_sample(v_dense,v_sparse,num):
    num=int(num)
    ori=np.exp(np.random.uniform(math.log(1), math.log(11),num))
    if v_dense<v_sparse:
        return v_dense+(v_sparse-v_dense)/10*(ori-1)
    else:
        ori=1+(11-ori)
        return v_sparse+(v_dense-v_sparse)/10*(ori-1)

def generate_log_random_parameters(sample_times,name_airfoil=None,path_airfoil_database  = "./airfoil_database/",min_velocity=10,max_velocity=100,range_AoA=22.5,ununiform_ratio=0.8):
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    if name_airfoil is None:
        files = os.listdir(path_airfoil_database)
        if len(files)==0:
            print("error - no airfoils found in {}".format(path_airfoil_database))
            exit(1)

    unun_num=int(sample_times*ununiform_ratio)
    un_num=sample_times-unun_num
    left_AoA=log_sample(-1*range_AoA,0,int(unun_num/2))
    right_AoA=log_sample(range_AoA,0,unun_num-int(unun_num/2))
    AoA=np.hstack((left_AoA,right_AoA,np.random.uniform(-1*range_AoA,range_AoA,un_num)))
    ve=np.hstack((log_sample(max_velocity,min_velocity,unun_num),np.random.uniform(min_velocity,max_velocity,un_num)))
    results=[]
    for i in range(sample_times):
        if name_airfoil is None:
            name = os.path.splitext(os.path.basename(files[np.random.randint(0, len(files))]))[0]
        else:
            name = name_airfoil
        results.append([name,ve[i],AoA[i]])
    return results

def generate_uniform_random_parameters(sample_times,name_airfoil=None,path_airfoil_database  = "./airfoil_database/",min_velocity=10,max_velocity=100,min_AoA=-22.5,max_AoA=22.5):
    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    if name_airfoil is None:
        files = os.listdir(path_airfoil_database)
        if len(files)==0:
            print("error - no airfoils found in {}".format(path_airfoil_database))
            exit(1)
    if len(files)==0:
        print("error - no airfoils found in {}".format(path_airfoil_database))
        exit(1)
    results=[]
    for i in range(sample_times):
        if name_airfoil is None:
            name = os.path.splitext(os.path.basename(files[np.random.randint(0, len(files))]))[0]
        else:
            name = name_airfoil
        results.append([name,np.random.uniform(min_velocity, max_velocity),np.random.uniform(min_AoA, max_AoA) ])
    return results

def generate_given_parameters(airfoil_names,velocities,AoAs):
    if not isinstance(airfoil_names,list):
        airfoil_names=[airfoil_names]
    if not isinstance(velocities,list):
        velocities=[velocities]
    if not isinstance(AoAs,list):
        AoAs=[AoAs]
    results=[]
    for name in airfoil_names:
        for ve in velocities:
            for AoA in AoAs:
                results.append([name,ve,AoA])
    return results

def plot_ve_AoA(paras,log=False):
    plt.scatter([para[2] for para in paras],[para[1] for para in paras],s=5)
    plt.xlabel("AoA")
    plt.ylabel("velocity")
    if log:
        plt.axes(yscale = "log",xscale="symlog") 
    plt.show()

def create_logger(name_logger):
    fm = logging.Formatter(fmt="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(filename="./{}.log".format(name_logger), mode='a')
    fh.setFormatter(fm)
    logger = logging.getLogger(name_logger)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(fh)
    return logger

def generator_process(parameters,path_OF_case,process_name,path_airfoil_database,logger,save_steps=[2510+i*10 for i in range(100)],save_image=False,copy_residuals=False):
    length=len(parameters)
    for i,((name,vel,AoA),save_dir) in enumerate(parameters):
        logger.info("{}: Running for {} airfoil, velocity={:.2f}, AoA={:.2f}, Total progress:{}/{}".format(process_name,name,vel,AoA,i+1,length))
        generate_a_data(airfoil_file=path_airfoil_database+name+".dat", velocity=vel, AoA=AoA, save_dir=save_dir, case_path=path_OF_case, save_steps=save_steps, save_image=save_image,copy_residuals=copy_residuals)
    shutil.rmtree(path_OF_case)
    print("{}: All work done.".format(process_name))

def parallel_generator(paras,process_num,path_OF_case="./OpenFOAM/",path_airfoil_database= "../airfoil_database/",save_dir="./generated_data/",save_steps=[2510+i*10 for i in range(100)],save_image=False,copy_residuals=False,folder_case_num=100):
    num_case=len(paras)
    if num_case%folder_case_num==0:
        num_folders=int(num_case/folder_case_num)
    else:
        num_folders=num_case//folder_case_num+1    
    if num_folders ==1:
        paras_all=[[c,save_dir] for c in paras]
    else:
        paras_all=[]
        for i,c in enumerate(paras):
            paras_all.append([c,save_dir+"part_{}/".format(i//folder_case_num+1)])
   
    indexes_group=[(i*int(num_case/process_num)) for i in range(process_num)]
    indexes_group.append(num_case)
    def process_instance(parameters,path_OF_case,process_name,logger):
        generator_process(parameters=parameters,path_OF_case=path_OF_case,process_name=process_name,path_airfoil_database=path_airfoil_database,logger=logger,save_steps=save_steps,save_image=save_image,copy_residuals=copy_residuals)
    for i in range(process_num):
        path_OF_case_now="./working_case_{}/".format(i+1)
        shutil.copytree(path_OF_case,path_OF_case_now)
        p=Process(target=process_instance, args=(paras_all[indexes_group[i]:indexes_group[i+1]],path_OF_case_now,"process_{}".format(i+1),create_logger("process_{}".format(i+1))))
        p.start()
