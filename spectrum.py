# Code adapted from https://github.com/sikora507/elgen/blob/master/src/audio%20analysis.ipynb into a command line tool.
# Thank you Tomasz. No licence present in original work, but the repo was public. Use at your own risk.

# Use: spectrum.py topng/towav input.wav/input.png rate
# rate only needed when converting back to wav.
#import tensorflow as tf

import os
cwd = os.getcwd()
import sys
import scipy.io.wavfile

if len(sys.argv) <  2 :
	sys.exit("Not enough arguments: Use: spectrum.py topng/towav input.wav/input.png")


input_file = sys.argv[2]

action = sys.argv[1]

if len(sys.argv) > 3 :

	rate = int(sys.argv[3])

if action == "topng" :

	rate,audData=scipy.io.wavfile.read(input_file)
	
	count = 0
	for sigh in audData.shape:
		count = count + 1

	print(count)
	print(audData.dtype)
	
	if count == 2:
		channel1=audData[:,0] #left
		#channel2=audData[:,1] #right - but we don't care
		signal_fragment = channel1
	else:
		signal_fragment = audData # mono

import numpy as np
import math
from PIL import Image
import time



FFT_LENGTH = 1024
WINDOW_LENGTH = 512
WINDOW_STEP = int(WINDOW_LENGTH / 2)
magnitudeMin = float("inf")
magnitudeMax = float("-inf")
phaseMin = float("inf")
phaseMax = float("-inf")



def amplifyMagnitudeByLog(d):
    return 188.301 * math.log10(d + 1)

def weakenAmplifiedMagnitude(d):
    return math.pow(10, d/188.301)-1

def generateLinearScale(magnitudePixels, phasePixels, magnitudeMin, magnitudeMax, phaseMin, phaseMax):
    height = magnitudePixels.shape[0] - 1 # scarrrryyyyy..
    width = magnitudePixels.shape[1]
    magnitudeRange = magnitudeMax - magnitudeMin
    phaseRange = phaseMax - phaseMin
    rgbArray = np.zeros((height, width, 3), 'uint8')
    
    for w in range(width):
        for h in range(height):
            magnitudePixels[h,w] = (magnitudePixels[h,w] - magnitudeMin) / (magnitudeRange) * 255 * 2
            magnitudePixels[h,w] = amplifyMagnitudeByLog(magnitudePixels[h,w])
            phasePixels[h,w] = (phasePixels[h,w] - phaseMin) / (phaseRange) * 255
            red = 255 if magnitudePixels[h,w] > 255 else magnitudePixels[h,w]
            green = (magnitudePixels[h,w] - 255) if magnitudePixels[h,w] > 255 else 0
            blue = phasePixels[h,w]
            rgbArray[h,w,0] = int(red)
            rgbArray[h,w,1] = int(green)
            rgbArray[h,w,2] = int(blue)
    return rgbArray

def recoverLinearScale(rgbArray, magnitudeMin, magnitudeMax, phaseMin, phaseMax):
    width = rgbArray.shape[1]
    height = rgbArray.shape[0]
    magnitudeVals = rgbArray[:,:,0].astype(float) + rgbArray[:,:,1].astype(float)
    phaseVals = rgbArray[:,:,2].astype(float)
    phaseRange = phaseMax - phaseMin
    magnitudeRange = magnitudeMax - magnitudeMin
    
    print(phaseRange)
    print(magnitudeRange)
    
    for w in range(width):
        for h in range(height):
            phaseVals[h,w] = (phaseVals[h,w] / 255 * phaseRange) + phaseMin
            magnitudeVals[h,w] = weakenAmplifiedMagnitude(magnitudeVals[h,w])
            magnitudeVals[h,w] = (magnitudeVals[h,w] / (255*2) * magnitudeRange) + magnitudeMin
    return magnitudeVals, phaseVals



def generateSpectrogramForWave(signal):
    start_time = time.time()
    global magnitudeMin
    global magnitudeMax
    global phaseMin
    global phaseMax
    buffer = np.zeros(int(signal.size + WINDOW_STEP - (signal.size % WINDOW_STEP)))
    buffer[0:len(signal)] = signal
    height = int(FFT_LENGTH / 2 + 1)
    width = int(len(buffer) / (WINDOW_STEP) - 1)
    magnitudePixels = np.zeros((height, width))
    phasePixels = np.zeros((height, width))

    for w in range(width):
        buff = np.zeros(FFT_LENGTH)
        stepBuff = buffer[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH]
        # apply hanning window
        stepBuff = stepBuff * np.hanning(WINDOW_LENGTH)
        buff[0:len(stepBuff)] = stepBuff
        #buff now contains windowed signal with step length and padded with zeroes to the end
        fft = np.fft.rfft(buff)
        for h in range(len(fft)):
            #print(buff.shape)
            #return
            magnitude = math.sqrt(fft[h].real**2 + fft[h].imag**2)
            if magnitude > magnitudeMax:
                magnitudeMax = magnitude 
            if magnitude < magnitudeMin:
                magnitudeMin = magnitude 

            phase = math.atan2(fft[h].imag, fft[h].real)
            if phase > phaseMax:
                phaseMax = phase
            if phase < phaseMin:
                phaseMin = phase
            magnitudePixels[height-h-1,w] = magnitude
            phasePixels[height-h-1,w] = phase
    rgbArray = generateLinearScale(magnitudePixels, phasePixels, magnitudeMin, magnitudeMax, phaseMin, phaseMax)
    elapsed_time = time.time() - start_time
    print('%.2f' % elapsed_time, 's', sep='')
    img = Image.fromarray(rgbArray, 'RGB')
    return img

if action == "topng" :

	scipy.io.wavfile.write("before.wav", rate, signal_fragment)
	img = generateSpectrogramForWave(signal_fragment)
	outfilename = input_file
	outfilename += "spectrogram.png"

	img.save(outfilename,"PNG")
	print(rate)
	print(phaseMin, phaseMax)
	print(magnitudeMin, magnitudeMax)

#generateSpectrogramForWave(signal_fragment)
	sys.exit(0)


def recoverSignalFromSpectrogram(filePath):
    img = Image.open(filePath)
    data = np.array( img, dtype='uint8' )
    width = data.shape[1]
    height = data.shape[0]
# need to locate min and max values from np array data.. bugger.

    magnitudeVals, phaseVals = recoverLinearScale(data, 0, 999999, -3.14158542844022, 3.141592653589793)
    
    recovered = np.zeros(WINDOW_LENGTH * width // 2 + WINDOW_STEP, dtype=np.int16)
    for w in range(width):
        toInverse = np.zeros(height, dtype=np.complex_)
        for h in range(height):
            magnitude = magnitudeVals[height-h-1,w]
            phase = phaseVals[height-h-1,w]
            toInverse[h] = magnitude * math.cos(phase) + (1j * magnitude * math.sin(phase))
        signal = np.fft.irfft(toInverse)
        recovered[w*WINDOW_STEP:w*WINDOW_STEP + WINDOW_LENGTH] += signal[:WINDOW_LENGTH].astype(np.int16)
    outfilename = input_file
    outfilename += "rec.wav"
    scipy.io.wavfile.write( outfilename, rate, recovered)




if action == "towav" :
	
	recoverSignalFromSpectrogram(input_file)


