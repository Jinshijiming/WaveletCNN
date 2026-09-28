
#CNN1D MCLDNN   FEA_T SigFormer IQFormer   ALWNN ULCNN WaveletCNN
python run.py model=IQFormer version=RML128_STFT__version_1 general.mode=Test
python run.py model=IQFormer version=RML128b_STFT__version_1 general.mode=Test
python run.py model=IQFormer version=RML1024_STFT__version_5 general.mode=Test
python run.py model=IQFormer version=TechRec_STFT__version_3 general.mode=Test

#----------------------
# RML128
#----------------------
python run.py model=FixedWaveletCNN dataset=RML128 dataset.norm=unit-power

nohup python run.py suffix=flops dataset=RML128  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML128  model=CNN1D dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128  model=MCLDNN dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops  dataset=RML128  model=FEA_T dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128  model=SigFormer dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128_STFT  model=IQFormer dataset.epochs=100 dataset.use_transform=False dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML128  model=ALWNN model.num_levels=1 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128  model=ULCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128  model=AMC_Net dataset.epochs=100 dataset.norm=unit-power  &
#----------------------
#RML128b
#----------------------
nohup python run.py suffix=flops dataset=RML128b  model=WaveletCNN dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML128b  model=CNN1D dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128b  model=MCLDNN dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML128b  model=FEA_T dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128b  model=SigFormer dataset.epochs=100 dataset.norm=unit-power  &
# nohup python run.py suffix=flops dataset=RML128b_STFT  model=IQFormer dataset.epochs=100 dataset.use_transform=False dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML128b  model=ALWNN model.num_levels=1 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128b  model=ULCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML128b  model=AMC_Net dataset.epochs=100 dataset.norm=unit-power  &

#----------------------
#TechRec
#----------------------
nohup python run.py dataset=TechRec model=FixedWaveletCNN dataset.norm=unit-power &

nohup python run.py suffix=flops dataset=TechRec  model=WaveletCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=TechRec  model=WaveletCNN model.kernel_size=17 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=TechRec  model=WaveletCNN model.kernel_size=21 dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=TechRec  model=CNN1D dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=TechRec  model=MCLDNN dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=TechRec  model=FEA_T dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=TechRec  model=SigFormer dataset.epochs=100 dataset.norm=unit-power  &
# nohup python run.py dataset=TechRec_STFT  model=IQFormer dataset.epochs=10000 dataset.use_transform=False dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=TechRec  model=ALWNN model.num_levels=4 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=TechRec  model=ULCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=TechRec  model=AMC_Net dataset.epochs=100 dataset.norm=unit-power  &


#----------------------
#RML1024
#----------------------
nohup python run.py suffix=flops dataset=RML1024  model=WaveletCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=RML1024  model=WaveletCNN model.kernel_size=13 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=RML1024  model=WaveletCNN model.kernel_size=15 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=RML1024  model=WaveletCNN model.kernel_size=17 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py dataset=RML1024  model=WaveletCNN model.kernel_size=21 dataset.epochs=100 dataset.norm=unit-power  &


nohup python run.py suffix=flops dataset=RML1024  model=CNN1D dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML1024  model=MCLDNN dataset.epochs=100 dataset.norm=mean-std  &

nohup python run.py suffix=flops dataset=RML1024  model=FEA_T dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML1024  model=SigFormer dataset.epochs=100 dataset.norm=max-min  &
nohup python run.py suffix=flops dataset=RML1024  model=SigFormer model.seg_lens=[2,2,7,7] dataset.epochs=100 dataset.norm=max-min  &
# nohup python run.py dataset=RML1024_STFT  model=IQFormer dataset.epochs=100 dataset.use_transform=False dataset.norm=unit-power  &

nohup python run.py suffix=flops dataset=RML1024  model=ALWNN model.num_levels=4 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML1024  model=ULCNN dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=flops dataset=RML1024  model=AMC_Net dataset.epochs=100 dataset.norm=unit-power  &



#-------------------
#超参数敏感性 TechRec RML128
#-------------------
nohup python run.py suffix=rate1 dataset=RML1024  model=WaveletCNN  model.rate=1 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate4 dataset=RML1024  model=WaveletCNN  model.rate=4 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate16 dataset=RML1024  model=WaveletCNN  model.rate=16 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate32 dataset=RML1024  model=WaveletCNN  model.rate=32 dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=rate1 dataset=TechRec  model=WaveletCNN  model.rate=1 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate4 dataset=TechRec  model=WaveletCNN  model.rate=4 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate16 dataset=TechRec  model=WaveletCNN  model.rate=16 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=rate32 dataset=TechRec  model=WaveletCNN  model.rate=32 dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=kernel7 dataset=RML1024  model=WaveletCNN  model.kernel_size=7 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel11 dataset=RML1024  model=WaveletCNN  model.kernel_size=11 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel15 dataset=RML1024  model=WaveletCNN  model.kernel_size=15 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel19 dataset=RML1024  model=WaveletCNN  model.kernel_size=19 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel21 dataset=RML1024  model=WaveletCNN  model.kernel_size=21 dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=kernel7 dataset=TechRec  model=WaveletCNN  model.kernel_size=7 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel11 dataset=TechRec  model=WaveletCNN  model.kernel_size=11 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel15 dataset=TechRec  model=WaveletCNN  model.kernel_size=15 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel19 dataset=TechRec  model=WaveletCNN  model.kernel_size=19 dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=kernel21 dataset=TechRec  model=WaveletCNN  model.kernel_size=21 dataset.epochs=100 dataset.norm=unit-power  &


#-------------------
#消融实验 TechRec RML128
#-------------------
nohup python run.py suffix=wofilter dataset=RML128  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=wofilter dataset=TechRec  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power  &


nohup python run.py suffix=woWavelet dataset=RML128  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power  &
nohup python run.py suffix=woWavelet dataset=TechRec  model=WaveletCNN  dataset.epochs=100 dataset.norm=unit-power  &

nohup python run.py suffix=woWavelet dataset=RML128  model=IQGCA  dataset.epochs=100 dataset.norm=unit-power  &