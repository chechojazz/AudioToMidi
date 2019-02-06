
# -*- coding: utf-8 -*-
#Python code based on create_nmat_guitar.mat by Helena Bantula and Sergio Giraldo.

__author__ = 'Sergio Giraldo. MTG 2015'

import numpy as np
import matplotlib.pyplot as plt
import copy
import music21 as mus
from midiutil.MidiFile import MIDIFile # use this?.... better use music 21 writer...?
import os as os

hopSize = 128
frameSize = 2048
sampleRate = 44100
lenPowThre = 0.1
minNoteLen = 0.03  # minimum note length defined in 30 ms


def f02nmat(folderName, fileName, f0, pwr, bpm, filter_opt, plot_noise_filter, plot_filters, minFrequency, maxFrequency):
    print('...creating MIDI data from pitch profile...'),

    # initial variables
    fs = 44100.0  # Sample rate

    # convert to Hz
    f0_Hz = copy.copy(f0)

    # guitar tessitura
    if filter_opt:
        # minFrequency = 160  # E3 guitar second E assuming that usually melody lines will be played above that register
        # maxFrequency = 1175  # D6 guitar highest note
        f0_Hz[f0_Hz < minFrequency] = 0 # nescesary? already filtered in pitch extraction.
        f0_Hz[f0_Hz > maxFrequency] = 0

    # Delete nans
    if np.size(np.where(np.isnan(f0_Hz))) > 0:#change for isempty!!!
        f0_Hz[np.where(np.isnan(f0_Hz))] = 0.0

    # Calculate energy...
    # pwr = np.absolute(pwr)
    pwr[np.where(pwr == 0)] = 0.000001
    energy = pwr  # energy = 10*np.log10(pwr)

    # plotAgain=1
    # while plotAgain:

    # Create noise gate...
    f0_noise = copy.copy(f0_Hz)
    if filter_opt:
        print "   ...applying noise adaptative filter."
        ro = 0.3
        #thres = np.mean(energy[np.where(energy > -0.00001)]) - ro*np.std(energy[np.where(energy > -0.000001)])# where energy is greater than zero
        thres = adapthres(pwr)
        noise = np.where(energy < thres)
        f0_noise[noise] = 0

    if plot_noise_filter:
        # axisShow = input('energy axis?:')
        # plot figures to adjust threshold value
        # timeVector = etc....
        plt.subplot(412)
        plt.plot(np.arange(0, len(f0), dtype=np.float) * hopSize / sampleRate, f0)
        #plt.xlabel('time(s)')
        plt.ylabel('Hz')
        plt.title('(b) Pitch profile.', fontsize=12)
        plt.subplot(413)
        # x_axis=np.arange(0,len(energy),1)
        plt.plot(np.arange(0, len(energy), dtype=np.float) * hopSize / sampleRate, energy)
        #plt.xlabel('time(s)')
        plt.ylabel('dB')
        plt.title('(c) Audio wave envelope (blue) and adaptative threshold (green).', fontsize=12)
        #plt.axhline(y=thres, color='r')
        plt.plot(np.arange(0, len(thres), dtype=np.float) * hopSize / sampleRate, thres)
        plt.subplot(414)
        plt.plot(np.arange(0, len(f0_noise), dtype=np.float) * hopSize / sampleRate, f0_noise)
        plt.xlabel('time(s)')
        plt.ylabel('Hz')
        plt.title('(d) Pitch profile filtered.', fontsize=12)
        plt.tight_layout()
        plt.show()

    # Hz to midi number
    tuning = 440  # standard tuning, usually 440hz but can variate +/- 4hz depending on the tuning of the instrument.
    f0_midi = f0_2_midi_sig(f0_noise, tuning)
    # Delete inf
    f0_midi[np.where(np.isinf(f0_midi))] = 0

    # Quantize (frec.)
    f0_midi = np.round(f0_midi)


    # first filter

    if filter_opt:
        print
        print "   ...applying filter 1..."
        frames_per_sec = fs / hopSize
        w = np.round(frames_per_sec * minNoteLen)
        f0_midi_fil_1 = filter_1(f0_midi, w)
    else:
        f0_midi_fil_1 = copy.copy(f0_midi)

    ## this can be done by essentia...
    # Get onset offsets (peaks diff(f0))
    df = np.append(0, np.diff(f0_midi_fil_1))
    onsets = np.zeros(len(df))
    offsets = np.zeros(len(df))
    off_flag = 0
    index = np.nonzero(df)[0]

    for i in range(len(index)):
        if index[i]==len(f0_midi_fil_1)-1:  #If there's a peak in the last frame,
            f0_midi_fil_1[-1] = f0_midi_fil_1[-2] #erase peak
        else :
            prev_f0 = f0_midi_fil_1[index[i]-1]
            next_f0 = f0_midi_fil_1[index[i]+1]
            actual_f0 = f0_midi_fil_1[index[i]]

            if prev_f0 > 0:
                offsets[index[i]] = 45
                if all([next_f0 == 0, df[index[i]+1] != 0]): #if next note is zero is a one length note going to a zero, so is an onset too
                    onsets[index[i]] = 40
            if next_f0 > 0:
                onsets[index[i]] = 40

            # in cases of one frame notes (or silences) comming or going to zero (conditions fail)
            if all([prev_f0 == 0, next_f0 == 0]):# if is a one frame note ..:¨:..
                onsets[index[i]] = 40
            if all([actual_f0 == 0, prev_f0 > 0, next_f0 > 0 ]):# if it is a one frame silence  ¨¨:.:¨¨
                onsets[index[i]] = 0 # previous wrongly labeled as onset
            asd=1

    # correct first and last onset/offset
    if f0_midi_fil_1[0] > 1:
        onsets[0] = 40
    if f0_midi_fil_1[-1] > 0:
        offsets[-1] = 45

    off_idx = np.where(offsets == 45)[0]
    on_idx = np.where(onsets == 40)[0]

    # calculate velocity based on energy
    power = np.zeros(len(on_idx))
    for i in range(len(on_idx)):
        noteLen = energy[on_idx[i]:min(off_idx[i],len(energy))]
        power[i] = np.mean(noteLen)

    # create midi type array
    pitch_midi = f0_midi_fil_1[on_idx]#nmat4
    chann = np.ones(len(pitch_midi))#nmat3
    vel = np.round(linmap(power,[60,100]))#nmat5
    onset_s = on_idx * hopSize / fs#nmat6
    dur_s = (off_idx - on_idx) * hopSize / fs #nmat7
    onset_b = onset_s * bpm / 60#nmat1
    dur_b = dur_s * bpm / 60#nmat2

    if filter_opt:
        print('   ...applying filter 2')
        [onset_b, dur_b, chann, pitch_midi, vel, onset_s, dur_s] = filter_2(onset_b, dur_b, chann, pitch_midi, vel, onset_s, dur_s)

    if plot_filters:
        # plot pitch profile based on midi quantization and after first filter
        plt.figure(3)
        # plot filter 1
        plt.subplot(311)
        plt.plot(f0_midi) #f0 before filter1 and after noise filter
        plt.subplot(312)
        plt.plot(f0_midi_fil_1) #f0 after filter 1 before filter 2
        plt.subplot(313)
        plt.plot(f0_midi_fil_1)
        plt.show()

    # create score.....
    if plot_filters:
        stream1 = nmat2score_sig(pitch_midi, onset_b, dur_b, False)
        mus.graph.plotStream(stream1)
    print "   ...done!"
    return pitch_midi, onset_b, onset_s, dur_b, dur_s, vel


def writeMidi(pitch_midi, onset_b, dur_b, vel, bpm, fileName):

    print ('...writing midi file to disk:' + os.getcwd() + '/dataOut/midi/' + fileName[:-4]+'.mid...'),
    # create midi file from parsed data
    # Create the MIDIFile Object with 1 track
    MyMIDI = MIDIFile(1)

    # Tracks are numbered from zero. Times are measured in beats.
    track = 0
    time = 0

    # Add track name and tempo.
    MyMIDI.addTrackName(track, time, fileName[:-4])
    MyMIDI.addTempo(track, time, bpm)

    # Add notes
    # addNote expects the following information:
    track = 0
    channel = 0
    # volume = 100
    for i in range(len(pitch_midi)):
        pitch_m = pitch_midi[i]
        time = onset_b[i]
        duration = dur_b[i]
        volume = vel[i]
        # Channel = chann[i] # channel values must be integers...
        # Now add the note.
        MyMIDI.addNote(track, channel, pitch_m, time, duration, volume)

    # And write it to disk.
    binfile = open(os.getcwd() + '/dataOut/midi/' + fileName[:-4]+'.mid', 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()
    print "   ...done!"


def adapthres(signal):
    M = 100  # input('input w (even):')
    ro = 0.05  #input('input ro:')
    lmbda = 0.8  # input('input lambda:')
    algo = 'lpf'  # raw_input('algorithm (median/lpf):')

    signal = np.append(np.zeros(M), signal) # append initial zeros to calculate first window
    signal = np.append(signal, np.zeros(M))

    t = np.zeros(len(signal))

    if algo == 'median':
        # median filter
        for n in range(M+1, len(signal)-M):
            aux = signal[n-M:n+M+1]
            t[n] = ro + lmbda*np.median(aux)
    else:
        # Low pass filter with hann window
        for n in range(M+1, len(signal)-M):
            aux = signal[n-M:n+M+1]
            t[n] = ro + lmbda * np.sum(np.multiply(aux, np.hanning(len(aux))))/len(aux)
        t = t[M:-M]
    return t  # take away initial zeros

def nmat2score_sig(pitch_midi,onset_b,dur_b, show):
    #function assumes durations of min 1/64

    t = 0 # time position at the end of each note

    stream1 = mus.stream.Stream()

    for i in range(len(pitch_midi)):
        if onset_b[i] - t > 0:  # If note onset minus previous note offset is higher than zero, then we have a rest
            r1 = mus.note.Rest()
            r1.duration.quarterLength = onset_b[i] - t #rest duration is from the end of previous note to onset of current note
            stream1.append(r1)
            t = onset_b[i]
        n1 = mus.note.Note() # Now we can create the current note
        n1.pitch.ps = pitch_midi[i]
        n1.duration.quarterLength = dur_b[i]
        stream1.append(n1)
        t = t + dur_b[i] # Update cursor position to the end of the current note

    if show:
        stream1.show('text')

    return stream1

def filter_1(P,w):
    # Python code based on filter_pitch.mat by Sergio Giraldo.
    # A function that quantices pitch according to the midi number notes. The input P is the midi number with decimal factors calculated by hz2midi function. It filters notes below 55hz (33 midi number), and notes shorter than w frames)

    import numpy as np
    import matplotlib.pyplot as plt

    gap =1 #silence gap equal to one frame The idea is that notes and silences must be treated differently: their respective minimun duration is different
    Q=P

    if w!=0:
        #filter short notes =< w frames
        Qdiff=np.append(np.diff(Q),0)#differentiate to find onsets (diff>0) and offsets (diff<0)
        on_off_idx=np.nonzero(Qdiff)[0]#find onset offset index for each note
        note_len=np.diff(on_off_idx)#find the length of each note

        for i in range(len(note_len)):#search notes shorter than w
            cond1=all([Q[on_off_idx[i]+1]>0 , note_len[i]<=w])#if is a note and length is less than w .....OR
            cond2=all([Q[on_off_idx[i]+1]<=0, note_len[i]<=gap])# if is a silence and length is less than gap
            if any([cond1,cond2]):

                prev_int=Q[on_off_idx[i]+1] - Q[on_off_idx[i]]#actual minus previous inteval

                # calculate interval to the next note
                next_note_len=note_len[i]#actual note length
                j=i#next note length
                next_int=Q[on_off_idx[i]+next_note_len+1] - Q[on_off_idx[i]+note_len[i]]

                ## CASE 1
                #print i
                if prev_int==-next_int:
                    #the note its a mistake, so the note is the previous(equal to next)
                    Q[on_off_idx[i]+1:on_off_idx[i]+1+note_len[i]]=Q[on_off_idx[i]]
                else:##Calcualte new next interval based on next note length
                    if j!=len(note_len)-1:
                        while note_len[j+1]<w and Q[on_off_idx[j+1]+1]>0:#if next note is too short and is not a rest
                            next_note_len=next_note_len+note_len[j+1]#add next note length to current note
                            j=j+1

                            if len(note_len)-1==j:#if j has reached the end of note length array
                                break

                    next_int=Q[on_off_idx[i]+next_note_len+1]-Q[on_off_idx[i]+note_len[i]]

                    ## CASE 2
                    #if the note is right in the middle betwen two notes half of the note is asigne to the previous note and half to the next note(ceil and floor are used in case the length of the note is odd
                    if prev_int==next_int:
                        whb=np.floor(next_note_len/2)#half part that goes backward
                        Q[on_off_idx[i]+1:on_off_idx[i]+whb]=Q[on_off_idx[i]]
                        whf=np.floor(next_note_len/2)#half part that goes foward
                        Q[on_off_idx[i]+whb+1:on_off_idx[i]+whb+whf]=Q[on_off_idx[i]+whf+whb+1]
                    else:# the note is assigned to the closest note

                        ## CASE 3
                        if min(abs(prev_int),abs(next_int))==abs(prev_int):#if the shorter interval is with the previous note
                            Q[on_off_idx[i]+1:on_off_idx[i]+1+note_len[i]]=Q[on_off_idx[i]]#assign the prevoious note
                        else:#assign the next note (no if required)
                            Q[on_off_idx[i]+1:on_off_idx[i]+1+note_len[i]]=Q[on_off_idx[i]+next_note_len+1]#assign the next note


    return Q[:]

def filter_2(onset_b, dur_b, chann, pitch_midi, vel, onset_s, dur_s):
    # second filter
    # Post filter: we multiply duration by energy, and define a thershold
    # for this value. Notes below threshold are omited. The idea is that notes
    # which are too short (but longer than 30 ms) and with low energy are
    # prone to be errors.
    i = 1
    while i <= len(pitch_midi) - 1:

        if dur_b[i] < 1 / 128:  # filter notes shorter than 1/128
            pitch_midi.pop(i)  # nmat4
            chann.pop(i)  # nmat3
            vel.pop(i)  # nmat5
            onset_s.pop(i)  # nmat6
            dur_s.pop(i)  # nmat7
            onset_b.pop(i)  # nmat1
            dur_b.pop(i)  # nmat2
            i = i - 1
        elif dur_s[i] < 0.1:  # apply only for notes shorter than 100ms
            idx_in = i - 2
            idx_out = i + 2  # we set a window of 5 notes, so we calculate mean of two previous notes and two next notes.

            # we calculate mean of vel*dur of two
            # surrounding notes and multyply it by a reducing factor (thre) of
            # 10%
            adap_th = np.mean(vel[max(idx_in, 0):min(idx_out, len(vel) - 1)]) * np.mean(
                dur_s[max(idx_in, 0):min(idx_out, len(dur_s) - 1)]) * lenPowThre
            vel_dur_fac = vel[i] * dur_s[i]  # Calculate vel*dur of current note

            if adap_th > vel_dur_fac:  # if vel*dur factor is smaller than a thresohold (10 %) equal to the mean of surrounding notes
                # nmat(i,:) = [];%the note is considered noise

                pitch_midi = np.delete(pitch_midi, i)  # nmat4
                chann = np.delete(chann, i)  # nmat3
                vel = np.delete(vel, i)  # nmat5
                onset_s = np.delete(onset_s, i)  # nmat6
                dur_s = np.delete(dur_s, i)  # nmat7
                onset_b = np.delete(onset_b, i)  # nmat1
                dur_b = np.delete(dur_b, i)  # nmat2
                i = i - 1
        i = i + 1
    return onset_b, dur_b, chann, pitch_midi, vel, onset_s, dur_s


def linmap(vin, rout):
    # function for linear mapping between two ranges
    # inputs:
    # vin: the input vector you want to map, range [min(vin),max(vin)]
    # rout: the range of the resulting vector
    # output:
    # the resulting vector in range rout
    # usage:
    # >>> v1 = np.linspace(-2,9,100);
    # >>> rin = np.array([-3,5])
    # >>> v2 = linmap(v1,rin);
    # *** (this function uses numpy module)
    a = np.amin(vin)
    b = np.amax(vin)
    c = rout[0]
    d = rout[1]
    e =((c+d) + (d-c)*((2*vin - (a+b))/(b-a)))/2

    return e


def f0_2_midi_sig(f0, tuning):
    # tuning: standard tuning, usually 440hz but can variate +/- 4hz depending on the tuning of the instrument.
    f0_midi = 12 * np.log2(f0 / tuning) + 69
    return f0_midi
