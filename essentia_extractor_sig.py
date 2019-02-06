# -*- coding: utf-8 -*-

__author__ = 'chechojazz'


# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License

# version 3 along with this program. If not, see http://www.gnu.org/licenses/

# Import libraries
import csv
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
import numpy
import os as os
import matplotlib.pyplot as plt

# Parameters for all algorithms
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True


def melody(folderName, fileName, minfrequency, maxfrequency, peakDistributionThreshold, peakFrameThreshold, pitchContinuity ,timeContinuity ):

    print "...performing melody extraction",

    #if default values (else use optimized)... to implement
#    minFrequency=1,#(real ∈ [0,∞), default = 0) :the minimum frequency of the range to evaluate [Hz]
#    maxFrequency=20000,#(real ∈ (0,∞), default = 5000) :the maximum frequency of the range to evaluate [Hz]
    peakDistributionThreshold=0.9,# (real ∈ [0,2], default = 0.9) :allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)
    peakFrameThreshold=0.9,# (real ∈ [0,1], default = 0.9) :per-frame salience threshold factor (fraction of the highest peak salience in a frame)
    pitchContinuity=27.5625,# (real ∈ [0,∞), default = 27.5625) :pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]
    timeContinuity=100,# (real ∈ (0,∞), default = 100) :time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]hopSize=hopSize)

    # RUNNING A CHAIN OF ALGORITHMS

    # create our algorithms:
    run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
    run_spectrum = Spectrum(size=frameSize * 4)
    run_spectral_peaks = SpectralPeaks(minFrequency=minfrequency,#(real ∈ [0,∞), default = 0) :the minimum frequency of the range to evaluate [Hz]
                                       maxFrequency=maxfrequency,#(real ∈ (0,∞), default = 5000) :the maximum frequency of the range to evaluate [Hz]
                                       maxPeaks=100,#(integer ∈ [1,∞), default = 100) :the maximum number of returned peaks
                                       sampleRate=sampleRate,#(real ∈ (0,∞), default = 44100) :the sampling rate of the audio signal [Hz]
                                       magnitudeThreshold=0,#(real ∈ [0,∞), default = 0) :peaks below this given threshold are not outputted
                                       orderBy="magnitude")#(string ∈ {frequency,magnitude}, default = frequency) :the ordering type of the outputted peaks (ascending by frequency or descending by magnitude)

    run_pitch_salience_function = PitchSalienceFunction(
                                        binResolution=10,# (real ∈ (0,∞), default = 10) salience function bin resolution [cents]
                                        harmonicWeight=0.8,# (real ∈ (0,1), default = 0.8) :h,armonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)
                                        magnitudeCompression=1,# (real ∈ (0,1], default = 1) :magnitude compression parameter (=0 for maximum compression, =1 for no compression)
                                        magnitudeThreshold=40,# (real ∈ [0,∞), default = 40) :peak magnitude threshold (maximum allowed difference from the highest peak in dBs)
                                        numberHarmonics =20,#(integer ∈ [1,∞), default = 20) :number of considered harmonics
                                        referenceFrequency=55,# (real ∈ (0,∞), default = 55) :the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin
                                                            )
    run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks(
                                        binResolution=10,# (real ∈ (0,∞), default = 10) :salience function bin resolution [cents]
                                        maxFrequency=maxfrequency,# (real ∈ [0,∞), default = 1760) :the maximum frequency to evaluate (ignore peaks above) [Hz]
                                        minFrequency=minfrequency,#(real ∈ [0,∞), default = 55) :the minimum frequency to evaluate (ignore peaks below) [Hz]
                                        referenceFrequency=55,# (real ∈ (0,∞), default = 55) :the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin
                                                                    )
    run_pitch_contours = PitchContours(
                                        binResolution=10,# (real ∈ (0,∞), default = 10) :salience function bin resolution [cents]
                                        hopSize=hopSize,#(integer ∈ (0,∞), default = 128) : the hop size with which the pitch salience function was computed
                                        minDuration=100,# (real ∈ (0,∞), default = 100) :the minimum allowed contour duration [ms]
                                        peakDistributionThreshold=peakDistributionThreshold,# (real ∈ [0,2], default = 0.9) :allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)
                                        peakFrameThreshold=peakFrameThreshold,# (real ∈ [0,1], default = 0.9) :per-frame salience threshold factor (fraction of the highest peak salience in a frame)
                                        pitchContinuity=pitchContinuity,# (real ∈ [0,∞), default = 27.5625) :pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]
                                        sampleRate=sampleRate,# (real ∈ (0,∞), default = 44100) :the sampling rate of the audio signal [Hz]
                                        timeContinuity=timeContinuity,# (real ∈ (0,∞), default = 100) :time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]hopSize=hopSize)
                                        )
    run_pitch_contours_melody = PitchContoursMelody(
                                        binResolution=10,# (real ∈ (0,∞), default = 10) :salience function bin resolution [cents]
                                        filterIterations=3,# (integer ∈ [1,∞), default = 3) :number of interations for the octave errors / pitch outlier filtering process
                                        guessUnvoiced=guessUnvoiced,#(bool ∈ {false,true}, default = false) :Estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame
                                        hopSize=hopSize,# (integer ∈ (0,∞), default = 128) :the hop size with which the pitch salience function was computed
                                        maxFrequency=20000,# (real ∈ [0,∞), default = 20000) :the minimum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]
                                        minFrequency=80,# (real ∈ [0,∞), default = 80) :the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]
                                        referenceFrequency=55,# (real ∈ (0,∞), default = 55) :the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin
                                        sampleRate=sampleRate,# (real ∈ (0,∞), default = 44100) :the sampling rate of the audio signal (Hz)
                                        #voiceVibrato=false,# (bool ∈ {true,false}, default = false) :detect voice vibrato
                                        voicingTolerance=0.2,# (real ∈ [-1.0,1.4], default = 0.2) :allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)
                                                    )
    run_equal_loudness = EqualLoudness(sampleRate = sampleRate)
    pool = Pool();


    # load audio
    audio = MonoLoader(filename = folderName + fileName)()
    audio = run_equal_loudness(audio)
    i=1
    # per-frame processing: computing peaks of the salience function
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
        salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)

        sec_idx=i*((hopSize*1.000000)/(sampleRate*1.000000))

        pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
        pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)
        pool.add('secs',sec_idx)
        pool.add('frame',i)

        i=i+1

    # post-processing: contour tracking and melody detection
    contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
            pool['allframes_salience_peaks_bins'],
            pool['allframes_salience_peaks_saliences'])
    pitch, confidence = run_pitch_contours_melody(contours_bins,
                                                  contours_saliences,
                                                  contours_start_times,
                                                  duration)


    #add contours and pitch to pool
    for line in contours_bins:
        pool.add('contours_bins', line)
    for line in contours_saliences:
        pool.add('contours_saliences', line)

    #pool.add('melody_pitch',pitch)
    #pool.add('melody_confidence',confidence)

    #write_yaml = YamlOutput(filename='melodia_output_all.sig');
    #write_yaml(pool)



    n_frames = len(pitch)
    print "number of frames:", n_frames

    #separate output in different lists
    allframes_salience_peaks_bins = y['allframes_salience_peaks_bins']
    allframes_salience_peaks_saliences = y['allframes_salience_peaks_saliences']
    contours_bins = y['contours_bins']
    contours_saliences = y['contours_saliences']
    secs = y['secs']
    frames_idx = y['frame']

    #create csvs for: salience, contours, and pitch
    if False: #write_data
        print "Writing csv of pitch contour..."
        with open(os.getcwd() + '/csv/' + fileName[:-4] + '_melody.csv','wb') as csv_file:
            f_writer=csv.writer(csv_file, delimiter = ',')
            f_writer.writerow(pitch)
            f_writer.writerow(confidence)
            f_writer.writerow(allframes_salience_peaks_bins)
            f_writer.writerow(allframes_salience_peaks_saliences)
            f_writer.writerow(secs)
            f_writer.writerow(frames_idx)


    print "...done!"
    return pitch, confidence



    # visualize output pitch
    #fig = plt.figure()
    #plot(range(n_frames), pitch, 'b')
    #n_ticks = 10
    #xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
    #xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
    #xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
    #plt.xticks(xtick_locs, xtick_lbls)
    #ax = fig.add_subplot(111)
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Pitch (Hz)')
    #suptitle("Predominant melody pitch")
    #
    ## visualize output pitch confidence
    #fig = plt.figure()
    #plot(range(n_frames), confidence, 'b')
    #n_ticks = 10
    #xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
    #xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
    #xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
    #plt.xticks(xtick_locs, xtick_lbls)
    #ax = fig.add_subplot(111)
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Confidence')
    #suptitle("Predominant melody pitch confidence")
    #
    #show()

def beatTrack(folderName, fileName, monophonic):
    print "...extracting beat information",

    if False: #input ('perform beat traking?'):
        run_Rhythm_Extractor = RhythmExtractor2013() # create algorithm

        if monophonic:  # nov 2015 do this for backing track in case of monophonic!
            audio = MonoLoader(filename = os.getcwd() + '/dataIn/bt/' + fileName)()
        else:  # if polyphonic
            audio = MonoLoader(filename = folderName + fileName)()

        bpm, ticks, confidence, estimates, bpmintervals = run_Rhythm_Extractor(audio)

        # nov 2015 copy beat tracking scheme from matlab with manual corection
        # to do...

        if False: #input ('write_bpm data?'):
            print "...writing csv of beat tracking",
            # nov 2015 data should be saved in dataOut/csv
            with open(os.getcwd() +'/dataOut/csv/'+ fileName[:-4] + '_beatTrack.csv','wb') as csv_file:
                f_writer=csv.writer(csv_file, delimiter=',')
                f_writer.writerow(ticks)
                f_writer.writerow(estimates)
                f_writer.writerow(bpmintervals)
        print "...done!"
        return bpm
        #bpm_all.append([fileName, bpm, confidence])
    else:
        print "...getting beat information from annotated data",
        # nov 2015 open bpm_all_manual.csv file---> we do this because for now we are not sure if bpm are detected
        # correcly. We will do bpm detection manual
        ifile  = open(os.getcwd() + '/dataIn/csv/' + 'bpm_all_manual.csv', "rb")
        reader = csv.reader(ifile)
        flg = 1
        for row in reader:
            if row[0] == fileName: #read the bpm marked for current file
                bpm = float(row[1])
                flg = 0

        if flg:
            raise ValueError('bpm information for file ' + fileName + ' was not found in /dataIn/csv/bpm_all_manual.csv')

        print "...done!"

        return bpm



def MFCCs(fileName):

    # RUNNING A CHAIN OF ALGORITHMS

    # create our algorithms:
    run_windowing = Windowing(type='hann', zeroPadding=3 * frameSize)  # Hann window with x4 zero padding
    run_spectrum = Spectrum(size=frameSize * 4)
    run_mfcc = MFCC()

    mfcc_all = [];

    # load audio
    audio = MonoLoader(filename = fileName)()
    # per-frame processing: computing peaks of the salience function
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        mfcc_frame = run_mfcc(spectrum)
        mfcc_all.append(mfcc_frame)

    return mfcc_all


def yin(folderName, fileName, minFrequency, maxFrequency):

    # create algorithms
    run_windowing = Windowing(type='hann') # Hann window
    run_equal_loudness = EqualLoudness(sampleRate = sampleRate)
    run_yin = PitchYin(frameSize = frameSize, # (integer ∈ [2, ∞), default = 2048) :number of samples in the input frame
                       interpolate = True, #(bool ∈ {true, false}, default = true) :enable interpolation
                       maxFrequency = maxFrequency, #(real ∈ (0, ∞), default = 22050) :the maximum allowed frequency [Hz]
                       minFrequency = minFrequency, #(real ∈ (0, ∞), default = 20) :the minimum allowed frequency [Hz]
                       sampleRate = sampleRate, #(real ∈ (0, ∞), default = 44100) :sampling rate of the input spectrum [Hz]
                       tolerance = 0.15 # (real ∈ [0, 1], default = 0.15) :tolerance for peak detection
                       )

    # create output
    pitch = []
    pitchConfidence = []


    # load audio
    audio = MonoLoader(filename = folderName + fileName)()
    audio = run_equal_loudness(audio)
    n = len(audio)/hopSize
    i = 0
    # per-frame processing: computing pitch per frame
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        pitch_frame, pitchConficence_frame = run_yin(frame)
        pitch.append(pitch_frame)
        pitchConfidence.append(pitchConficence_frame)
        print "\r...performing pitch detection: {0:.2f}%".format((float(i) / n) * 100),
        i += 1

    pitchArray = numpy.array(pitch)
    pitchConfidenceArray = numpy.array(pitchConfidence)

    print "...done!"
    return pitchArray, pitchConfidenceArray

def pitchContSeg(folderName, fileName, pitch_profile):
    print ('...creating MIDI data from pitch profile using pitch contour segmentation...'),
    # create algorithms
    run_pitchContourSegmentation = PitchContourSegmentation(hopSize = hopSize, #(integer ∈ (0, ∞), default = 128) :hop size of the extracted pitch
                                                            minDur = 0.1, #(real ∈ (0, ∞), default = 0.1) :minimum note duration [s]
                                                            pitchDistanceThreshold = 60, # (integer ∈ (0, ∞), default = 60) :pitch threshold for note segmentation [cents]
                                                            rmsThreshold = -2, # (integer ∈ (-∞, 0), default = -2) :zscore threshold for note segmentation
                                                            sampleRate = sampleRate, # (integer ∈ (0, ∞), default = 44100) :sample rate of the audio signal
                                                            tuningFreq = 440, # (integer ∈ (0, 22000), default = 440) :tuning reference frequency [Hz]
                                                            )


    # load audio
    audio = MonoLoader(filename = folderName + fileName)()

    # get midi data

    onset_s, dur_s, pitch_midi = run_pitchContourSegmentation(essentia.array(pitch_profile), audio)

    print '...done'
    return onset_s, dur_s, pitch_midi

def envelope (folderName, fileName, plot_noise_filter):

    print ('...calculating signal envelope...'),
    # Create algorithm
    run_envelope = Envelope(applyRectification=True, # (bool ∈ {true, false}, default = true) :whether to apply rectification (envelope based on the absolute value of signal)
                            attackTime=3, # (real ∈ [0, ∞), default = 10) :the attack time of the first order lowpass in the attack phase [ms]
                            releaseTime=3, # (real ∈ [0, ∞), default = 1500) :the release time of the first order lowpass in the release phase [ms]
                            sampleRate=sampleRate,  # (real ∈ (0, ∞), default = 44100) :the audio sampling rate [Hz]
                            )
    # load audio
    audio = MonoLoader(filename = folderName + fileName)()

    # get envelope

    envl = run_envelope(audio)
    if plot_noise_filter:
        plt.figure(2)
        plt.subplot(411)
        plt.plot(np.arange(0, len(audio), dtype=np.float) / sampleRate, audio)
    #    plt.xlabel('time(s)')
        plt.ylabel('dB')
        plt.title('(a) Audio wave.', fontsize=12)
        # plt.plot(envl)
#        plt.axes([])

    if True:  # down sample the envelope signal to the same hopsize and frame size (this is filtering, so we
    # decided to use a median filter approach. Use median instead of mean
        envl_2 = []
        for frame in FrameGenerator(envl, frameSize=frameSize, hopSize=hopSize):
            envl_2_frame = np.sort(frame)[int(frameSize/3)]
            envl_2.append(envl_2_frame)

        envl = np.array(envl_2)

    print '...done'
    return envl

