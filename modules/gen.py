import os
import subprocess


def getModelDir(folder):

    rootOfRepo = subprocess.getoutput("git rev-parse --show-toplevel")
    addressFromRoot = "/data/models/"
    address = rootOfRepo + addressFromRoot + folder

    return address


def getLogDir(folder):

    rootOfRepo = subprocess.getoutput("git rev-parse --show-toplevel")
    addressFromRoot = "/data/logs/"
    address = rootOfRepo + addressFromRoot + folder

    return address


def makeModelSaveDir(folder):

    address = getModelDir(folder)

    if not os.path.exists(address):
        os.mkdir(address)
    
    return address


def makeLogSaveDir(folder):

    address = getLogDir(folder)

    if not os.path.exists(address):
        os.mkdir(address)
    
    return address
