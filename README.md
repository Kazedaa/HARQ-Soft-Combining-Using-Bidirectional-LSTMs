# HARQ_Soft_Combining_Using_BiLSTMs

## Abstract
Hybrid Automatic Repeat reQuest is used in modern wireless data communica-
tion to integrate both Automatic Repeat reQuest and high-rate Forward Error Correction
mechanisms to enhance the reliability of data transmission. Unlike traditional ARQ, where an
error-ridden frame is discarded upon reception, HARQ temporarily stores the erroneous frame
in a buffer. When the re-transmission of the same frame occurs, these two frames are com-
bined to generate a new frame, trying to minimize errors. This is HARQ with Soft Combining.
Existing methods like Chase Combining (Type-I) and Incremental Redundancy (Type-II and
Type-III) implement Log-Likelihood Ratio and Maximum Ratio Combining to combine two
erroneous frames. This paper uses a Bidirectional Long Short-Term Memory model to com-
bine two frames with high channel noise errors. This paper introduces the BiLSTM model,
which aims to reduce Bit Error Rate and provides an approach for integrating this model
into the existing HARQ structure.

![image](https://github.com/user-attachments/assets/0039d881-38eb-4ee5-9a0b-7743219d0af8)

## Usage
### Set up the environment
```bash
conda create -n harq_soft_combining_using_bilstms -f environment.yaml
conda activate harq_soft_combining_using_bilstms
```
### Run Main
#### Modulation Schemes <br>
Quadrature Amplitude Modulation(QAM) - "qam" <br>
Pulse amplitude modulation(PAM) - "pam" <br>
Phase-shift keying(PSK) - "psk" <br>
Amplitude shift keying(ASK) - "ask" <br>

```bash
python3 main.py -m "<modulation-scheme>" 
```
The plots are saved in their respective folders.

### Run Demo Notebook
Change the "MOD" macro to any of the modulation techniques used.

## References
