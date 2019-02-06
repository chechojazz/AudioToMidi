# -*- coding: utf-8 -*-
__author__ = 'chechojazz'


import sys
import os
import essentia_extractor_sig
import descriptors_extractor_sig
import csv
import createMidi_gt
import ml_sig
import numpy as np
import Tkinter, tkFileDialog


def main():

    #get imput arguments (use Tkinter to get folder, and console to arguments)
    # root = Tkinter.Tk()
    # folderName = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
    # if len(folderName ) > 0:
    #     print "You chose %s" % folderName

    folderName = os.getcwd()

    #try:
    #    folderName = sys.argv[1]
    #except:
    #    print "usage:", sys.argv[0], "<audio_files_folder>"
    #    sys.exit()

    for fileName in os.listdir(folderName + '/dataIn/performed/wav/'):#Get files name list from performance folder (wavs)
        if fileName[-3:]=="wav":#Read only wav files in the folder
            print 'Parsing ' + fileName  + '...'

            scoreProcess(folderName + '/dataIn/score/', fileName[:-3] + 'xml')
            performanceProcess(folderName + '/dataIn/performed/wav/', fileName)

    print "SUCCESS!!!"

    return

def performanceProcess(folderName, fileName):

    # options

    filter_opt = True
    use_pitch_cont_seg = True  # With adaptative and euristic filters by Giraldo and Bantula
    plot_noise_filter = True
    plot_filters = False

    # initial variables...

    #flag = 1
    #bpm_all = [['File Name','bpm', 'confidence']]
    minFrequency = 100  # E3 = 160Hz guitar second E assuming that usually melody lines will be played above that register
    maxFrequency = 2000  # D6 = 1175Hz guitar highest note
    monophonic = input('Is audio monophonic?')

    # If monophonic audio (use YIN)
    if monophonic:
        f0, pitch_confidence = essentia_extractor_sig.yin(folderName, fileName, minFrequency, maxFrequency)

    else:

    # If poliphonic audio (use Melodia)
        f0, pitch_confidence = essentia_extractor_sig.melody(folderName, fileName, minFrequency, maxFrequency, peakDistributionThreshold, peakFrameThreshold, pitchContinuity, timeContinuity)

    # improove f0 voice and unvoice detection (optional until it works better, may 2015)
    if False:
        melody_pitch, melody_confidence = ml_sig.kmeans_clust(folderName, fileName, f0, pitch_confidence)

    # Get pwr
    if monophonic: #(based on envelope for monophonic signals, nov 2015)
        pwr = essentia_extractor_sig.envelope(folderName, fileName, plot_noise_filter)
    else:
        pwr = pitch_confidence

    # Estimate bpm
    if True:
        bpm = essentia_extractor_sig.beatTrack(folderName, fileName, monophonic)
        #bpm, ticks, confidence, estimates, bpmintervals = essentia_extractor_sig.beatTrack(folderName, fileName, monophonic)[0]

    # Create discrete note events (MIDI) from pitch profile
    if use_pitch_cont_seg:
        # create MIDI from pitch profile using filters (our approach)
        pitch_midi, onset_b, onset_s, dur_b, dur_s, vel = createMidi_gt.f02nmat(folderName, fileName, f0, pwr, bpm, filter_opt, plot_noise_filter, plot_filters, minFrequency, maxFrequency)
    else:
        # create MIDI from pitch profile using essentia (no energy information is obtained here...)
        onset_s, dur_s, pitch_midi = essentia_extractor_sig.pitchContSeg(folderName, fileName, f0)
        onset_b = onset_s * bpm / 60  # get onset in beats
        dur_b = dur_s * bpm / 60  # get duration in beats
        vel = np.ones(len(onset_b) * 70)  # create same velocity for each note (change this to rms val based on note segmentation)

    # write midi file to disk
    createMidi_gt.writeMidi(pitch_midi, onset_b, dur_b, vel, bpm, fileName)

    print "done"

    return


def scoreProcess(folderName, fileName):

    notesStream = descriptors_extractor_sig.read_xml(folderName, fileName)

    return

if __name__ == "__main__":
    main()