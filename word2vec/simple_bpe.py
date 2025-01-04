# from tokenizers import ByteLevelBPETokenizer
# tokenizer=ByteLevelBPETokenizer()
# tokenizer.train(["C:\\Users\\Administrator\\Desktop\\Task3\\datasetexample.txt"],vocab_size=900)
# tokenizer.save('charbpe_007.json')

from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers,processors
tokenizer=Tokenizer(models.BPE())
tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder=decoders.ByteLevel()
tokenizer.post_processor=processors.ByteLevel(trim_offsets=True)
trainer=trainers.BpeTrainer(vocab_size=2000)
tokenizer.train(["C:\\Users\\Administrator\\Desktop\\Task3\\datasetexample.txt"],trainer=trainer)
tokenizer.save("charbpe_007.json",pretty=True)
