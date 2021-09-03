# TTS_DNN

![image](https://user-images.githubusercontent.com/37763987/130567151-4e43dc5c-d4f0-420d-9dfc-bac4073f0d67.png)

## directory 

- TTS_DNN
    - data
        - source 
            - lab
                - state
                    - xxx.lab ~ xxx.lab   
            - lf0
                - xxx.lf0 ~ xxx.lf0
            - mgc
                - xxx.mgc ~ xxx.mgc
            - set
            - stat
        - gen
            - lab
                - state
            - set
        - questions
            - questions_qst001.conf
    - model 
        - source 
            - am.state_dict
            - dur.state_dict
    - configs 
        - IOConfigs.json
        - Configs.json
    - gen 
        - source 
            - from_a
            - from_b
            - output_hdf5
    - scripts
        - tts.py
        - mkdir.rb
        - mkdab.rb
        - mkc.rb
        - mkset.rb
        - mkstat.rb
        - trn_am.py
        - trn_dur_v2.py
        - gen_from_a.py
        - gen_from_b.py
        - synthesis_mlpg.rb
        - synthesis_mlpg.rb
        - gen_hdf5_from_b.py
    - Makefile

## Note
1. lab 檔案必須由HTS系統所切過的 state lab
2. lf0 和 mgc 檔案由WORLD演算法求得
3. questionset 由 LabelGenerator 所產出


# references: 
- H. Zen and A. Senior, “Deep mixture density networks for acoustic modeling in statistical parametric speech,” in ICASSP, pp. 3844–3848, 2014. 


