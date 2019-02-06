__author__ = 'chechojazz'
import music21 as mus
import xml.etree.ElementTree as ET


def read_xml(pathName,fileName):

    notesStream = mus.converter.parse(pathName + fileName)
    notesStream2 = ET.parse(pathName + fileName)

#    for thisNote in notesStream.pitches


    return notesStream2
