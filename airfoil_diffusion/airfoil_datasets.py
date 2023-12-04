#usr/bin/python3
import os
import torch
import numpy as np
from tqdm import tqdm
from .network_modules.helpers import default

DIMLESS_MAG=[100,38.5,1.0,4.3,2.15,2.35]

def find_max_min_mag(datafiles):
    dataset=AirfoilDataset(datafiles,load_in_memory=False,model="dimless")
    data0=dataset.read_datai(0)
    max_list=[torch.max(data0[j]).item() for j in range(6)]
    min_list=[torch.min(data0[j]).item() for j in range(6)]
    for i in range(len(dataset)-1):
        datai=dataset.read_datai(i+1)
        for j in range(6):
            if j != 2:
                maxv=torch.max(datai[j])
                minv=torch.min(datai[j])
                if  maxv> max_list[j]:
                    max_list[j] = maxv.item()
                if minv < min_list[j]:
                    min_list[j] = minv.item()
    return max_list,min_list,[max([abs(max_list[i]),abs(min_list[i])]) for i in range(6)]

def save_case_list_to_filedataFiles(case_list,save_name):
    os.makedirs(os.path.dirname(os.path.abspath(save_name)),exist_ok=True)
    with open(save_name,"w") as f:
        for case in case_list[:-1]:
            f.write(os.path.abspath(case["path"]+case["file_name"])+os.linesep)
        f.write(os.path.abspath(case_list[-1]["path"]+case_list[-1]["file_name"]))

def sort_case_list(case_list,key="velocity",reverse=False):
    aim_list=[case[key] for case in case_list]
    indexes=sorted(range(len(aim_list)), key=lambda k: aim_list[k],reverse=reverse)
    newcase_list=[]
    for i in indexes:
        c_now=case_list[i]
        newcase_list.append(
            {"airfoil":c_now["airfoil"],"velocity":c_now["velocity"],"AoA":c_now["AoA"],"file_name":c_now["file_name"],"path":c_now["path"],"time_tag":c_now["time_tag"]}
        )
    return newcase_list

def normalized2dimless(p_u_fields,mag_list=None):
    with torch.no_grad():
        if mag_list is None:
            mag_list = DIMLESS_MAG
        recover=mag_list[3:6]
        return p_u_fields*torch.tensor(recover,device=p_u_fields.device).unsqueeze(-1).unsqueeze(-1)
    
def normalized2real(p_u_fields,velocity,mag_list=None):
    with torch.no_grad():
        dimless=normalized2dimless(p_u_fields=p_u_fields,mag_list=mag_list)
        return dimless*torch.tensor([velocity*velocity,velocity,velocity],device=dimless.device).unsqueeze(-1).unsqueeze(-1)

def real2dimless(fields,velocity):
    with torch.no_grad():
        fields[-3] /= (velocity*velocity)
        fields[-2] /= velocity
        fields[-1] /= velocity            
        return fields

def real2normalized(fields,max_mag_list,velocity):
    with torch.no_grad():
        datai=real2dimless(fields=fields,velocity=velocity)
        for j in range(datai.shape[-3]):
            datai[-1-j] /= max_mag_list[-1-j]
        return datai

def read_single_raw_file(file_path):
    with torch.no_grad():
        return torch.tensor(np.load(file_path)["a"]).float()

def read_single_file(file_path,scale_factor=None,offset_pressure=True,model="normalized",velocity=None,mag_list=None):
    with torch.no_grad():
        datai=torch.tensor(np.load(file_path)["a"]).float()
        if scale_factor is not None:
            if scale_factor !=1:
                datai=torch.nn.functional.interpolate(input=datai.unsqueeze(0),scale_factor=scale_factor, mode='bilinear')
                datai=datai.squeeze(0)
        if offset_pressure:
            cmask=1-datai[2]
            datai[3]-= torch.sum(datai[3]*cmask)/torch.sum(cmask)
            datai[3]*= cmask
        if model=="normalized":
            velocity=default(velocity,float(os.path.basename(file_path).split("_")[-3])/100)
            return real2normalized(fields=datai,max_mag_list=default(mag_list,DIMLESS_MAG),velocity=velocity)
        elif model=="dimless":
            velocity=default(velocity,float(os.path.basename(file_path).split("_")[-3])/100)
            return real2dimless(fields=datai,velocity=velocity)
        elif model=="real":
            return datai


class DataFiles():

    def __init__(self,case_list):
        self.case_list=case_list
    
    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        return self.case_list[idx]

    def get_case_information(self,filename,filepath):
        if filename[-4:]==".npz":
            words=filename[0:-4]
        else:
            words=filename
        words=words.split("_")
        return {"airfoil":"_".join(words[0:-3]),"velocity":float(words[-3])/100,"AoA":float(words[-2])/100,"time_tag":float(words[-1]),"file_name":filename,"path":filepath}

    def get_case_path(self,idx):   
        return self.case_list[idx]["path"]+self.case_list[idx]["file_name"]
    
    def sort(self,key="velocity",reverse=False):
        self.case_list=sort_case_list(self.case_list,key=key,reverse=reverse)

    def get_airfoil_names(self):
        names=[]
        for case in self.case_list:
            if case["airfoil"] not in names:
                names.append(case["airfoil"])
        return names

    def get_simulation_cases(self):
        result=[]
        for case in self.case_list:
            found=False
            for r in result:
                if case["airfoil"]==r["airfoil"] and case["velocity"]-r["velocity"]==0 and case["AoA"]-r["AoA"]==0:
                    found=True
            if not found:
                result.append({"airfoil":case["airfoil"],"velocity":case["velocity"],"AoA":case["AoA"]})
        return result
    
    def select_simulation_cases(self,simulation_cases):
        result=[]
        for r in simulation_cases:
            for case in self.case_list:
                if case["airfoil"]==r["airfoil"] and case["velocity"]-r["velocity"]==0 and case["AoA"]-r["AoA"]==0:
                    result.append(case)
        return result
                
class FolderDataFiles(DataFiles):
    def __init__(self,folder_path):
        files=os.listdir(folder_path)
        case_list=[]
        if files[0][0:5]=="part_" or files[1][0:5]=="part_":
            for folder_now in files:
                case_list+=[self.get_case_information(name,folder_path+folder_now+os.sep) for name in os.listdir(folder_path+folder_now+os.sep) if os.path.splitext(name)[-1] == ".npz"]
        else:
            case_list=[self.get_case_information(name,folder_path) for name in files if os.path.splitext(name)[-1] == ".npz"]
        super().__init__(case_list)

class FileDataFiles(DataFiles):
    def __init__(self,file_path,base_path=""):
        with open(file_path) as f:
            names=f.readlines()
        super().__init__([self.get_case_information(os.path.basename(base_path+name.strip()),os.path.dirname(base_path+name.strip())+os.sep) for name in names])

class AirfoilDataset_Base():
    def __init__(self,datafiles,mag_list,data_size=128,model="normalized",max_list=None,min_list=None):
        self.datafiles=datafiles
        self.mag_list=mag_list
        self.scale_factor=data_size/128
        if self.scale_factor-1 != 0:
            self.interpolate=True
        else:
            self.interpolate=False
        if model=="normalized":
            self.read_datai=self.read_datai_normalized
        elif model=="dimless":
            self.read_datai=self.read_datai_dimless
        elif model=="real":
            self.read_datai=self.read_datai_real
        else:
            raise Exception('parameter "model" need to me one of ["normalized","dimless","real"], but got {}'.format(model)) 
        
    def __len__(self):
        return len(self.datafiles)

    def read_datai_real(self,idx):
        datai=read_single_raw_file(self.datafiles.get_case_path(idx))
        if self.interpolate:
            datai=torch.nn.functional.interpolate(input=datai.unsqueeze(0),scale_factor=self.scale_factor, mode='bilinear')
            datai=datai.squeeze(0)
        cmask=1-datai[2]
        datai[3]-= torch.sum(datai[3]*cmask)/torch.sum(cmask)
        datai[3]*= cmask
        return datai

    def read_datai_dimless(self,idx):
        return real2dimless(self.read_datai_real(idx),self.datafiles[idx]["velocity"])   
  
    def read_datai_normalized(self,idx):
        return real2normalized(fields=self.read_datai_real(idx),max_mag_list=self.mag_list,velocity=self.datafiles[idx]["velocity"])   

class AirfoilDataset_memory(AirfoilDataset_Base):
    def __init__(self, datafiles, mag_list, data_size=128, model="normalized", max_list=None, min_list=None):
        super().__init__(datafiles, mag_list, data_size, model, max_list, min_list)
        self.datas=[]
        for i in tqdm(range(len(self)),desc="Loading datas"):
            self.datas.append(self.read_datai(i))            
        
    def __getitem__(self, idx):
        return self.datas[idx][0:3], self.datas[idx][3:], self.datafiles[idx]

class AirfoilDataset_disk(AirfoilDataset_Base):
    def __init__(self, datafiles, mag_list, data_size=128, model="normalized", max_list=None, min_list=None):
        super().__init__(datafiles, mag_list, data_size, model, max_list, min_list)
    
    def __getitem__(self, idx):
        datai=self.read_datai(idx)
        return datai[0:3], datai[3:], self.datafiles[idx]  

def AirfoilDataset(datafiles,data_size=128,load_in_memory=True,mag_list=None,model="normalized"):
    if mag_list is None:
        mag_list=DIMLESS_MAG
    if load_in_memory:
        return AirfoilDataset_memory(datafiles,mag_list,max_list=None,min_list=None,data_size=data_size,model=model)
    else:
        return AirfoilDataset_disk(datafiles,mag_list,max_list=None,min_list=None,data_size=data_size,model=model)