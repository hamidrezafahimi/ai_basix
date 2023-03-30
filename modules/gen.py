import os
import subprocess
import sys
from pathlib import Path

def resetDir(dir):
    # loggingPath = 
    if os.path.exists(dir):
        removeDirectory(dir+"/")
    os.mkdir(dir)


def removeDirectory(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item == "." or item == "..":
            pass
        elif item.is_dir():
            removeDirectory(item)
        else:
            item.unlink()
    directory.rmdir()


class TerminalLog(object):
    def __init__(self, file):
        self.orgstdout = sys.stdout
        
        # if os.path.exists(file):
        #     os.remove(file)
        # open(file, "x")
        self.log = open(file, "a")

    def write(self, msg):
        self.orgstdout.write(msg)
        self.log.write(msg)  


def getRootDir():
    return subprocess.getoutput("git rev-parse --show-toplevel")


def getModelDir(folder):

    addressFromRoot = "/data/models/"
    address = getRootDir() + addressFromRoot + folder

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
