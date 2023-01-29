import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from scipy import signal
import math
from scipy.io.wavfile import read
from scipy.signal import spectrogram, lfilter, tf2zpk, freqz


def dft(x):
    val = 0
    dftcko = np.ndarray((1024), dtype=complex)
    for k in range(1024):
        for n in range(1024):
            val += x[n] * np.exp(-1j*((2*math.pi)/1024) * k * n)
        dftcko[k] = val
        val = 0
    return dftcko

#Ukol1
fs, s  = read('../audio/xbacka01.wav')

delkaSignaluVzorky = s.size
delkaSignaluSekundy = delkaSignaluVzorky / fs
minHodnota = min(s)
maxHodnota = max(s)

print("Delka signalu vo vzorkach: ", delkaSignaluVzorky, "\nDelka signalu v sekundach: ", delkaSignaluSekundy, "\nMin. hodnota signalu: ", minHodnota, "\nMax. hodnota signalu: ", maxHodnota)

plotSekundy = np.arange(delkaSignaluVzorky) / fs
plt.figure(figsize=(6,3))
plt.plot(plotSekundy, s)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Signál')
plt.tight_layout()
plt.show()
#plt.savefig('baseSignal.png')
plt.clf()


#Ukol2
dataw = s + 0.0 #premena na float
dataw -= np.mean(dataw) #ustrednenie
dataw /= np.abs(s).max() #-1 az 1 range

plt.figure(figsize=(6,3))
plt.plot(plotSekundy, dataw)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Ustredneny a normalizovany signál')
plt.tight_layout()
#plt.show()
#plt.savefig('normalizSignal.png')
plt.clf()


ramceMatrix = np.ndarray((93, 1024))



for i in range(93):
    positionStart = int((768)*i)
    positionEnd = int(positionStart + 1024)
    if positionEnd <= len(dataw) :
        frameI = np.array(dataw[positionStart : positionEnd])
        ramceMatrix[i] = np.array(frameI)

plotFrameSekundy = np.arange(1024) / fs
plotFrameSekundy += 1024/fs*768*47/1000
plt.figure(figsize=(6,3))
plt.plot(plotFrameSekundy, ramceMatrix[46])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Jeden Rámec č.47')
plt.tight_layout()
plt.show()
#plt.savefig('normalizSignalRamec47.png')
plt.clf()

#Ukol3
plotFrameSekundyhalf = np.arange(512) / fs
plotFrameSekundyhalf += 512/fs*768*49/1000

dftcko = dft(ramceMatrix[49])
fftcko = np.fft.fft(ramceMatrix[49])

#G = 10 * np.log10(1/1024 * np.abs(fftcko)**2)


_, ax = plt.subplots(2, 1)

ax[0].plot(plotFrameSekundyhalf, fftcko[:512])
ax[0].set_xlabel('$f[Hz]$')
ax[0].set_title('FFT')
ax[0].grid(alpha=0.5, linestyle='--')


ax[1].plot(plotFrameSekundyhalf, dftcko[:512])
ax[1].set_xlabel('$f[Hz]$')
ax[1].set_title('DFT')
ax[1].grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()


#uloha 4
f, t, sgr = spectrogram(dataw, fs)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20)



plt.figure(figsize=(10, 5))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

#Uloha 6

# Parameters
rate = 16000    # samples per second
                # sample duration (seconds)
f = 693       # sound frequency (Hz)# Compute waveform samples
t1 = np.arange(48026)
x1= np.cos(2 * np.pi * f * t1 /fs)/160# Write the samples to a file


x2= np.cos(2 * np.pi * f*2 * t1 /fs)/160# Write the samples to a file


x3 = np.cos(2 * np.pi * f*3 * t1 /fs)/160# Write the samples to a file


x4 = np.cos(2 * np.pi * f*4 * t1 /fs)/160# Write the samples to a file


x = x1 + x2 + x3 + x4

sf.write('../audio/4cos.wav', x, fs)

f, t, sgr = spectrogram(x, fs)
sgr_log = 10 * np.log10(sgr+1e-20)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu rusenia [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

#Uloha 7


samp_freq = 16000
freq1 = 693.0
freq2 = 693.0 *2.0
freq3 = 693.0 *3.00
freq4 = 693.0 *4.0
quality_factor = 30.0
b1, a1 = signal.iirnotch(freq1, quality_factor, samp_freq)
b2, a2 = signal.iirnotch(freq2, quality_factor, samp_freq)
b3, a3 = signal.iirnotch(freq3, quality_factor, samp_freq)
b4, a4 = signal.iirnotch(freq4, quality_factor, samp_freq)
freq1, h1 = signal.freqz(b1, a1, fs = samp_freq)
freq2, h2 = signal.freqz(b2, a2, fs = samp_freq)
freq3, h3 = signal.freqz(b3, a3, fs = samp_freq)
freq4, h4 = signal.freqz(b4, a4, fs = samp_freq)





y1 = signal.filtfilt(b1, a1, dataw)
y2 = signal.filtfilt(b2, a2, y1)
y3 = signal.filtfilt(b3, a3, y2)
y4 = signal.filtfilt(b4, a4, y3)

f, t, sgr = spectrogram(y4, fs)
sgr_log = 10 * np.log10(sgr+1e-20)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.gca().set_title('Spektogram vyfiltrovaného signálu')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu signalu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()
###############################################Ukol 10#################################################
sf.write('../audio/clean_bandstop.wav', y4, fs) #uzastne vyfiltrovane :)

#dataw = data + 0.0
#dataw -= np.mean(dataw) #ustrednenie
#dataw /= np.abs(data).max() #-1 az 1 range

plt.figure(figsize=(12,6))
plt.plot(plotSekundy, y4)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Vyfiltrovaný signál')
plt.tight_layout()
plt.show()
#plt.savefig('normalizSignal.png')
plt.clf()

#######################################################################################################
print("Koeficienty 1. Filtru a:", a1, "b:", b1)
print("Koeficienty 2. Filtru a:", a2, "b:", b2)
print("Koeficienty 3. Filtru a:", a3, "b:", b3)
print("Koeficienty 4. Filtru a:", a4, "b:", b4)

N_imp = 50
imp = [1, *np.zeros(N_imp-1)] # jednotkovy impuls

h1 = lfilter(b1, a1, imp)

plt.figure(figsize=(7,5))
plt.stem(np.arange(N_imp), h1, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva prvého filtru(693Hz) $h[n]$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

h2 = lfilter(b2, a2, imp)

plt.figure(figsize=(7,5))
plt.stem(np.arange(N_imp), h2, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva druhého filtru(693*2Hz) $h[n]$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

h3 = lfilter(b3, a3, imp)

plt.figure(figsize=(7,5))
plt.stem(np.arange(N_imp), h3, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva tretieho filtru(693*3Hz) $h[n]$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

h4 = lfilter(b4, a4, imp)

plt.figure(figsize=(7,5))
plt.stem(np.arange(N_imp), h4, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva štvrtého filtru(693*4Hz) $h[n]$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.show()

#Uloha8
# nuly, poly
z1, p1, k1 = tf2zpk(b1, a1)
# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z1), np.imag(z1), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p1), np.imag(p1), marker='x', color='g', label='póly')

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
plt.gca().set_title('Nulové body a póly prvého filtru')
plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# nuly, poly
z2, p2, k2 = tf2zpk(b2, a2)
# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z2), np.imag(z2), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p2), np.imag(p2), marker='x', color='g', label='póly')
plt.gca().set_title('Nulové body a póly druhého filtru')
plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# nuly, poly
z3, p3, k3 = tf2zpk(b3, a3)
# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z3), np.imag(z3), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p3), np.imag(p3), marker='x', color='g', label='póly')
plt.gca().set_title('Nulové body a póly tretieho filtru')
plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# nuly, poly
z4, p4, k4 = tf2zpk(b4, a4)
# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z4), np.imag(z4), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p4), np.imag(p4), marker='x', color='g', label='póly')
plt.gca().set_title('Nulové body a póly štvrtého filtru')
plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("Prvy filter: Poly: ", p1," Nuly: ", z1  )
print("Druhy filter: Poly: ", p2," Nuly: ", z2  )
print("Treti filter: Poly: ", p3," Nuly: ", z3  )
print("Štvrtý filter: Poly: ", p4," Nuly: ", z4  )

#Ukol 9
# frekvencni charakteristika
w1, H1 = freqz(b1, a1)

_, ax = plt.subplots(1, 2, figsize=(8,3))

ax[0].plot(w1 / 2 / np.pi * fs, np.abs(H1))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvenční charakteristiky prvého filtra $|H(e^{j\omega})|$')

ax[1].plot(w1 / 2 / np.pi * fs, np.angle(H1))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenční charakteristiky prvého filtra $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()
plt.clf()


w2, H2 = freqz(b2, a2)

_, ax = plt.subplots(1, 2, figsize=(8,3))

ax[0].plot(w2 / 2 / np.pi * fs, np.abs(H2))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvenční charakteristiky druhého filtra $|H(e^{j\omega})|$')

ax[1].plot(w2 / 2 / np.pi * fs, np.angle(H2))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenční charakteristiky druhého filtra $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()
plt.clf()


w3, H3 = freqz(b3, a3)

_, ax = plt.subplots(1, 2, figsize=(8,3))

ax[0].plot(w3 / 2 / np.pi * fs, np.abs(H3))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvenční charakteristiky tretieho filtra $|H(e^{j\omega})|$')

ax[1].plot(w3 / 2 / np.pi * fs, np.angle(H3))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenční charakteristiky tretieho filtra $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()
plt.clf()


w4, H4 = freqz(b4, a4)

_, ax = plt.subplots(1, 2, figsize=(8,3))

ax[0].plot(w4 / 2 / np.pi * fs, np.abs(H4))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvenční charakteristiky tretieho filtra $|H(e^{j\omega})|$')

ax[1].plot(w4 / 2 / np.pi * fs, np.angle(H4))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title('Argument frekvenční charakteristiky tretieho filtra $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()
plt.clf()