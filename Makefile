.PHONY: data

clean :
	#rm -r data/set
	#rm -r data/stat
	rm -r data/lf0
	rm -r data/mgc
	rm -r data/lab/state
	
#DNN : mkc mkset mkstat train_acoustic_DNN train_duration_DNN generate_acoustic_DNN generate_duration_DNN synthesis_acoustic_DNN synthesis_duration_DNN
DNN : mkdab mkc mkset mkstat train_acoustic_DNN train_duration_DNN generate_acoustic_DNN generate_duration_DNN synthesis_acoustic_DNN synthesis_duration_DNN
LSTM : mkdab mkc mkset mkstat train_acoustic_LSTM train_duration_LSTM generate_acoustic_LSTM generate_duration_LSTM synthesis_acoustic_LSTM synthesis_duration_LSTM
DNN_train : train_acoustic_DNN train_duration_DNN generate_acoustic_DNN generate_duration_DNN synthesis_acoustic_DNN synthesis_duration_DNN


data : 
	python scripts/preprocessing.py
mkdab :
	python scripts/mkdab.py

mkc :
	python scripts/mkc.py

mkset :
	python scripts/mkset.py
	
mkstat :
	python scripts/mkstat.py
	
train_acoustic_DNN :
	python scripts/train_v2.py --train_model acoustic --train_type DNN
train_acoustic_LSTM :
	python scripts/train_v2.py --train_model acoustic --train_type LSTM
train_duration_DNN :
	python scripts/train_v2.py --train_model duration --train_type DNN
train_duration_LSTM :
	python scripts/train_v2.py --train_model duration --train_type LSTM
	
generate_acoustic_DNN :
	python scripts/gen.py --train_model acoustic --train_type DNN
generate_acoustic_LSTM :
	python scripts/gen.py --train_model acoustic --train_type LSTM
generate_duration_DNN : 
	python scripts/gen.py --train_model duration --train_type DNN
generate_duration_LSTM :
	python scripts/gen.py --train_model duration --train_type LSTM

synthesis_acoustic_DNN :
	python scripts/synthesis_mlpg.py --train_model acoustic --train_type DNN
synthesis_acoustic_LSTM :
	python scripts/synthesis_mlpg.py --train_model acoustic --train_type LSTM
synthesis_duration_DNN : 
	python scripts/synthesis_mlpg.py --train_model duration --train_type DNN
synthesis_duration_LSTM :
	python scripts/synthesis_mlpg.py --train_model duration --train_type LSTM
	
synthesis :
	python scripts/synthesis_mlpg.py
verify :
	python scripts/verify_data.py ; python scripts/verify_data_r.py

train_am :
	python scripts/train.py --am
	
train_dur :
	python scripts/train.py --dur
	
train_am_lstm :
	python scripts/train.py --am_lstm
	
train_dur_lstm :
	python scripts/train.py --dur_lstm

gen_hdf5_from_b :
	python scripts/gen_hdf5_from_b.py --source
	
mvf_hdf5 :
	cp -r /home/adamliu/Desktop/project/dnn_tts/gen/hdf5_DNN/tr_slt /home/adamliu/PytorchWaveNetVocoder/egs/arctic/mine/hdf5/
	cp -r /home/adamliu/Desktop/project/dnn_tts/gen/hdf5_DNN/ev_slt /home/adamliu/PytorchWaveNetVocoder/egs/arctic/mine/hdf5/
	
check : 
	python scripts/check_data.py

