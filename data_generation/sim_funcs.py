#usr/bin/python3
# Adapted from https://github.com/thunil/Deep-Flow-Prediction

import os
import math
import numpy as np
import os
from PIL import Image
from matplotlib import cm


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


def imageOut(filename, outputs_param, targets_param, saveTargets=False):
    outputs = np.copy(outputs_param)
    targets = np.copy(targets_param)

    for i in range(3):
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        outputs[i] -= min_value
        targets[i] -= min_value
        max_value -= min_value
        outputs[i] /= max_value
        targets[i] /= max_value

        suffix = ""
        if i == 0:
            suffix = "_pressure"
        elif i == 1:
            suffix = "_velX"
        else:
            suffix = "_velY"

        im = Image.fromarray(cm.magma(outputs[i], bytes=True))
        im = im.resize((512, 512))
        im.save(filename + suffix + "_pred.png")

        if saveTargets:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            im = im.resize((512, 512))
            im.save(filename + suffix + "_target.png")


def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im.save(filename)


def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)])) < 1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(
            pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX",
                                    "{}".format(pointIndex-1))
                outFile.write(line)

    if os.system("gmsh airfoil.geo -3 -o airfoil.msh -format msh2 > /dev/null") != 0:
        print("error during mesh creation!")
        return(-1)

    if os.system("gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return(-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp", "constant/polyMesh/boundary")

    return(0)


def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    os.system("./Allclean && simpleFoam $1>/dev/null 2>error.log")


def outputProcessing(airfoil_name, velocity, AoA, save_dir,sample_file, tag_name="", save_image=True, res=128):
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    angle = AoA/180*math.pi 
    freestreamX = math.cos(angle) * velocity
    freestreamY = math.sin(angle) * velocity
    npOutput = np.zeros((6, res, res))
    os.makedirs(save_dir, exist_ok=True)
    if save_image:
        os.makedirs(save_dir+"data_pictures/", exist_ok=True)

    ar = np.loadtxt(sample_file)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf) < 1e-4 and abs(ar[curIndex][1] - yf) < 1e-4:
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
                npOutput[3][x][y] = ar[curIndex][3] #p
                npOutput[4][x][y] = ar[curIndex][4] #ux
                npOutput[5][x][y] = ar[curIndex][5] #uy
                curIndex += 1
                # fill input as well
            else:
                # fill mask
                npOutput[2][x][y] = 1.0
                

    #fileName = dataDir + str(uuid.uuid4()) # randomized name
    tag = "{}_{:.0f}_{:.0f}_{}".format(
        airfoil_name, velocity*100, AoA*100, tag_name)
    fileName = save_dir + tag
    if save_image:
        saveAsImage(
            save_dir+'data_pictures/pressure_{}.png'.format(tag), npOutput[3])
        saveAsImage(
            save_dir+'data_pictures/velX_{}.png'.format(tag), npOutput[4])
        saveAsImage(
            save_dir+'data_pictures/velY_{}.png'.format(tag), npOutput[5])
        saveAsImage(
            save_dir+'data_pictures/inputX_{}.png'.format(tag), npOutput[0])
        saveAsImage(
            save_dir+'data_pictures/inputY_{}.png'.format(tag), npOutput[1])

    #print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)


def generate_a_data(airfoil_file: str, velocity: float, AoA: float, save_dir: str, case_path="./OpenFOAM/", save_steps=[2600, 2700, 2800, 2900, 3000], save_image=True,copy_residuals=False):
    """_summary_

    Args:
        airfoil_file (str): The full path of the airfoil shape file. Note that if you want to use a relative path, please give the path relative to the OpenFOAM case folder.
        velocity (float):  The total input freestream velocity
        angle (float): Angle of attack
        save_dir (str): The folder path where the new result should be
        save_image (bool, optional): Whether to generate image of the output data.. Defaults to True.
    """
    airfoil_name = airfoil_file.split(os.sep)[-1].split(".")[0]
    angle = AoA/180*math.pi
    fsX = math.cos(angle) * velocity
    fsY = math.sin(angle) * velocity
    #print("\tusing {}".format(airfoil_name))
    #print("\tUsing velocity {:.3f} AoA {:.3f} ".format(velocity, AoA))
    #print("\tResulting freestream vel x,y: {},{}".format(fsX, fsY))
    os.makedirs(save_dir+"residuals/", exist_ok=True)
    os.chdir(case_path)
    if genMesh(airfoil_file) != 0:
        print("mesh generation for {} failed, aborting.".format(airfoil_name))
        os.chdir("..")
    else:
        runSim(fsX, fsY)
        os.chdir("..")
        for step in save_steps:
            time_tag = "{:.0f}".format(step)
            outputProcessing(airfoil_name=airfoil_name, velocity=velocity, AoA=AoA, save_dir=save_dir, tag_name=time_tag,
                             save_image=save_image, sample_file="{}postProcessing/internalCloud/{}/cloud.xy".format(case_path,time_tag))
        if copy_residuals:
            os.system("cp {}postProcessing/residuals/0/residuals.dat {}residuals/{}_{:.0f}_{:.0f}.dat".format(case_path,save_dir,airfoil_name,velocity*100,AoA*100))
        #print("\tdone")

