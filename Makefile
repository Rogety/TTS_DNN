all: mkdir mkdab mkc mkset mkstat trn_source_am trn_source_dur gen_source_from_a syn_source_from_a gen_source_from_b syn_source_from_b
source: mkdir mkdab mkc mkset mkstat trn_source_am trn_source_dur
source_gen: gen_source_from_a syn_source_from_a gen_source_from_b syn_source_from_b

mkdir:
	ruby scripts/mkdir.rb

mkdab:
	ruby scripts/mkdab.rb --source  --gen

mkc:
	ruby scripts/mkc.rb --source

mkset:
	ruby scripts/mkset.rb --source

mkstat:
	python scripts/mkstat.py --source

trn_source_am:
	python scripts/trn_am.py --source

trn_source_dur:
	python scripts/trn_dur.py --source

gen_source_from_a:
	python scripts/gen_from_a.py --source

gen_source_from_b:
	python scripts/gen_from_b.py --source

syn_source_from_a:
	ruby scripts/synthesis_mlpg.rb --source --from_a

syn_source_from_b:
	ruby scripts/synthesis_mlpg.rb --source --from_b

gen_hdf5_from_b :
	python scripts/gen_hdf5_from_b.py --source
	

clean :
	rm -r data/source/set 
	rm -r data/source/stat 
	rm -r data/gen/set
	
clean-lab : 
	rm -r data/source/lab/state 
	rm -r data/gen/lab/state
	
clean-lf0 : 
	rm -r data/source/lf0 
	rm -r data/source/mgc
	
