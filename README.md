# Neural Lexicon Reader: Reduce Pronunciation Errors in End-to-end TTS by Leveraging External Textual Knowledge
This is an implementation of the [paper](https://arxiv.org/abs/2110.09698), along with the pipeline and pretrained model using an open dataset. Audio 
samples of the paper is available [here](https://mutiann.github.io/papers/nlr).

# Recipe
This open pipeline uses the Databaker dataset. Please refer to [our previous pipeline](https://github.com/mutiann/few-shot-transformer-tts) for dataset preprocessing, 
while only the Databaker dataset is used. Besides, you need to run `lexicon/build_databaker.py` to build the 
vocabulary, download the lexicon from zdic.net, and encode them with XLM-R. Feel free to change the target directory to 
save the data, which is specified in `build_databaker.py` and `lexicon_utils.py`.

Below are the commands to train and evaluate. Default target directories specified in the preprocessing scripts are 
used, so please substitute them with your own.
The evaluation script can be run simultaneously with the training script.
You may also use the evaluation script to synthesize samples from pretrained models.
Please refer to the help of the arguments for their meanings.

`python -m torch.distributed.launch --nproc_per_node=NGPU  --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=D:\free_corpus\packed\ --training_languages=zh-cn 
--eval_languages=zh-cn --training_speakers=databaker --eval_steps=100000:150000 
--hparams="input_method=char,multi_speaker=True,use_knowledge_attention=True,remove_space=True,data_format=nlti"
--external_embed=D:\free_corpus\packed\embed.zip --vocab=D:\free_corpus\packed\db_vocab.json` 

`python eval.py --model-dir=MODEL_DIR --log-dir=LOG_DIR 
--data-dir=D:\free_corpus\packed\ --eval_languages=zh-cn --eval_meta=D:\free_corpus\packed\metadata.eval.txt
--hparams="input_method=char,multi_speaker=True,use_knowledge_attention=True,remove_space=True,data_format=nlti"  
--start_step=100000 --vocab=D:\free_corpus\packed\db_vocab.json 
--external_embed=D:\free_corpus\packed\embed.zip --eval_speakers=databaker`

Besides, to report CER, you need to create `azure_key.json` with your own Azure STT subscription, with content of
`{"subscription": "YOUR_KEY", "region": "YOUR_REGION"}`, see `utils/transcribe.py`.
Due to significant differences of the datasets used, the implementation is for demonstration only and could not fully 
reproduce the results in the paper.

## Pretrained Model
The pretrained models on Databaker are available at [OneDrive Link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/mhear_connect_ust_hk/EnGa-PhdOn1GnJ7_qH2y8nwBW-75jAfQiXVir65ut-7f6w?e=T9y4hq),
which reaches a CER of 4.19%. Relevant files necessary for generation of speeches including lexicon texts, lexicon embeddings, 
the vocabulary file, and evaluation scripts are also included to aid fast reproduction.
