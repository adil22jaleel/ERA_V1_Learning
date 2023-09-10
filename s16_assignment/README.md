# Custom Transformer-Based Machine Translation: English to French

# Final Loss: 1.684
# Total Epochs Ran: 50
# Epoch in which loss became under 1.8: 26th Epoch

This repository contains the code for training a transformer-based machine translation model using PyTorch. The project is structured into several files:

[model.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s16_assignment/model.py): Defines the transformer architecture for machine translation.

[config.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s16_assignment/config.py): Contains configuration parameters for the training process.

[dataset.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s16_assignment/dataset.py): Implements the dataset loader and preprocessing for machine translation data.

[train.py](https://github.com/adil22jaleel/era-v1-assignments/blob/main/s16_assignment/train.py): Contains the training loop and logic for training the machine translation model.


## Model Architecture (model.py)
The model.py file defines the transformer architecture used for machine translation. It includes components such as:

- Positional encoding
- Multi-head attention
- Feedforward layers
- Encoder and decoder blocks
- Layer normalization


## Configuration (config.py)
The config.py file stores configuration parameters for the training process. You can adjust these parameters to control aspects like batch size, learning rate, model size, and training epochs. Additionally, you can specify whether to preload a pre-trained model checkpoint.
- "batch_size": The batch size used for training and evaluation.

- "num_epochs": The number of training epochs or iterations over the entire dataset.

- "lr": The learning rate, determining the step size at which the model's weights are updated during training.

- "seq_len": The maximum sequence length for input data.

- "d_model": The dimensionality of the model's hidden representations.

- "d_ff": The dimensionality of the feed-forward layer in the model.

- "lang_src": The source language identifier or code.

- "lang_tgt": The target language identifier or code.

- "model_folder": The folder path where model weights and checkpoint files will be saved.

- "model_basename": The base name for model weight and checkpoint files.

- "preload": A boolean flag indicating whether to preload a pretrained model.

- "tokenizer_file": The file name template for saving and retrieving tokenizers.

- "experiment_name": The name of the experiment or run, typically used for logging or organizing experiment results.

- "ds_mode": The data loading mode, which could be "not_disk" or another mode indicating how data is loaded.

- "ds_path": The path to the dataset directory or source.

- "save_ds_to_disk": A boolean flag indicating whether to save the dataset to disk.

## Dataset Loading (dataset.py)
The dataset.py file handles dataset loading and preprocessing. It defines a custom dataset class, BilingualDataset, that reads and preprocesses bilingual text data. This class also tokenizes the input sentences and generates masks for padding.


## Training Loop (train.py)
The train.py file contains the main training loop for training the machine translation model. You can run this script to train your model using the specified configuration.
Train.py contains the following functions
- greedy_decode: The greedy_decode function performs greedy decoding to generate a translation of the source sequence. Greedy decoding is a simple decoding strategy where, at each step, the model selects the token with the highest probability as the next token in the sequence.

- run_validation: run_validation function is designed for evaluating a sequence-to-sequence model on a validation dataset and computing various evaluation metrics such as Character Error Rate (CER), Word Error Rate (WER), and BLEU score. The function also prints examples of source, target, and predicted translations for a specified number of examples.

-  get_all_sentences: Simple helper function designed to extract all sentences from a dataset in a specified language.

- get_or_build_tokenizer: Designed to either retrieve an existing tokenizer for a specified language or build a new tokenizer if it doesn't exist. The function is based on the Hugging Face tokenizers library and uses a word-level tokenizer with special tokens for unknown words ([UNK]), padding ([PAD]), start-of-sequence ([SOS]), and end-of-sequence ([EOS]).

- get_ds: Designed for data preprocessing and dataset creation. The function loads a dataset, performs tokenization, divides it into training and validation sets, and prepares data loaders for training a neural network.

- collate: Typically used as a custom collation function for creating batches of data during training or evaluation. Dynamic paddings is done on this function. 
Dynamic padding is important for optimizing training speed and memory usage 
when working with variable-length sequences, such as in machine translation tasks. 
It allows you to batch sequences of different lengths without wasting computation 
on unnecessary padding tokens

- train_model: Designed to train a Transformer model for a sequence-to-sequence (seq2seq) task, likely machine translation. The function encompasses various steps of the training process, including data loading, model creation, optimization, and checkpointing.

   - Device Selection:The function selects the computing device (CPU or GPU) based on the availability of a CUDA-compatible GPU. The selected device is stored in the device variable.

   - Creating Model Folder:
It creates a folder for saving model weights and other training-related files based on the provided configuration.

   - Data Loading:
The function uses the get_ds function to load and preprocess the training and validation datasets. It also retrieves the source and target language tokenizers and prepares data loaders for training.

   - Model Initialization:
The Transformer model is initialized using the get_model function, with the configuration, source language vocab size, and target language vocab size passed as arguments. The model is moved to the selected device (device).

   - Tensorboard Setup:
Tensorboard logging is set up using the SummaryWriter to track training progress, losses, and metrics.

   - Optimizer and Learning Rate Scheduler:
An Adam optimizer is initialized for training the model, with the specified learning rate (lr) and other hyperparameters.A learning rate scheduler, OneCycleLR, is defined to adjust the learning rate during training. It includes parameters like max_lr, steps_per_epoch, and epochs.

   - Loading Pretrained Model (Optional):
If the preload flag is set in the configuration, a pretrained model is loaded from a specified file path. This is helpful for fine-tuning or continuing training from a previous checkpoint.

   - Loss Function and Gradient Scaling:
The loss function is defined as CrossEntropyLoss with special handling for padding tokens ([PAD]) and label smoothing.Gradient scaling is enabled using the torch.cuda.amp.GradScaler to handle mixed-precision training.

   - Training Loop:The main training loop iterates over epochs, where each epoch processes batches of data.
The function uses gradient accumulation to accumulate gradients over multiple steps before performing the optimizer step. This helps improve training stability and efficiency.
Learning rate scheduling is applied, and the current learning rate is logged during training.
Losses and metrics are logged using Tensorboard.

   - Validation:
After each epoch, the function calls the run_validation function to perform validation on the validation dataset. It calculates metrics like Character Error Rate (CER), Word Error Rate (WER), and BLEU score and logs them using Tensorboard.
   
   - Model Checkpointing:At the end of each epoch, the model's weights, optimizer state, and other training-related information are saved as a checkpoint file. The file includes the epoch number and can be used for model resumption or evaluation.  


## Training Log
```
Max length of source sentence: 471
Max length of target sentence: 482
Training on Epoch 00: 100%|██████████| 834/834 [03:27<00:00,  4.01it/s, Loss_Acc=6.133, Loss=5.214, Sqn_L=61]
--------------------------------------------------------------------------------
    SOURCE: "Well, do calculate, my boy."
    TARGET: --Calcule, mon garçon.
 PREDICTED: -- Eh bien , mon cher , mon oncle .
--------------------------------------------------------------------------------
    SOURCE: "True God!" muttered Phoebus, "targes, big−blanks, little blanks, mailles,* every two worth one of Tournay, farthings of Paris, real eagle liards!
    TARGET: « Vrai Dieu ! grommela Phœbus, des targes, des grands-blancs, des petits-blancs, des mailles d’un tournois les deux, des deniers parisis, de vrais liards-à-l’aigle !
 PREDICTED: -- C ' est ce que M . de Rênal , , , , , , , , , , , !
--------------------------------------------------------------------------------
/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `CharErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `CharErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `WordErrorRate` from `torchmetrics` was deprecated and will be removed in 2.0. Import `WordErrorRate` from `torchmetrics.text` instead.
  _future_warning(
/usr/local/lib/python3.10/dist-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `BLEUScore` from `torchmetrics` was deprecated and will be removed in 2.0. Import `BLEUScore` from `torchmetrics.text` instead.
  _future_warning(
Training on Epoch 01: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=4.593, Loss=4.544, Sqn_L=64]
--------------------------------------------------------------------------------
    SOURCE: I had now brought my state of life to be much easier in itself than it was at first, and much easier to my mind, as well as to my body.
    TARGET: J'avais alors amené mon état de vie à être en soi beaucoup plus heureux qu'il ne l'avait été premièrement, et beaucoup plus heureux pour mon esprit et pour mon corps.
 PREDICTED: J ' avais donc pris mon état de vie pour être plus qu ' il était à la première , et bien que mon esprit , comme mon corps , comme mon corps .
--------------------------------------------------------------------------------
    SOURCE: "Yes, yes," said he; "you disturb, you agitate the people who live in the castle."
    TARGET: «Oui, oui, répondit-il, oui: vous troublez, vous agitez les gens qui habitent ce château.»
 PREDICTED: -- Oui , oui , dit - il , vous , vous les gens qui dans le château .
--------------------------------------------------------------------------------
Training on Epoch 02: 100%|██████████| 834/834 [03:26<00:00,  4.04it/s, Loss_Acc=3.859, Loss=3.623, Sqn_L=136]
--------------------------------------------------------------------------------
    SOURCE: "There are many men in town, such as Lord St. Vincent, Lord Hood, and others, who move in the most respectable circles, although they have nothing but their services in the Navy to recommend them."
    TARGET: Il y a à la ville des hommes, tels que Lord Saint-Vincent, Lord Hood, qui font figure dans les sociétés les plus respectables, bien qu'ils n'aient pour toute recommandation que leurs services dans la marine.
 PREDICTED: -- Il y a des hommes dans la ville , comme Lord Saint - , Lord , et autres , qui se dans les plus , bien qu ' ils n ' ont que leurs services à leur .
--------------------------------------------------------------------------------
    SOURCE: It was apparent that his strength was gradually diminishing.
    TARGET: Il était visible que le capitaine s'éteignait peu à peu.
 PREDICTED: C ' était une violence que sa force était peu .
--------------------------------------------------------------------------------
Training on Epoch 03: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=3.470, Loss=3.152, Sqn_L=48]
--------------------------------------------------------------------------------
    SOURCE: Under these dark waters there stretched the bank of shellfish, an inexhaustible field of pearls more than twenty miles long.
    TARGET: Là, sous les sombres eaux, s'étendait le banc de pintadines, inépuisable champ de perles dont la longueur dépasse vingt milles.
 PREDICTED: Sous ces eaux sombres , il y avait la rive de la , un champ de perles plus de vingt milles .
--------------------------------------------------------------------------------
    SOURCE: He took to his heels when he spotted me.
    TARGET: Il a détalé en m’apercevant.
 PREDICTED: Il me prit à ses talons , lorsqu ’ il m ’ aperçut .
--------------------------------------------------------------------------------
Training on Epoch 04: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=3.237, Loss=3.196, Sqn_L=68]
--------------------------------------------------------------------------------
    SOURCE: In the next place, I told her that her being alive was the only support of the discovery, and that while she owned me for her child, and saw reason to be satisfied that I was so, nobody else would doubt it; but that if she should die before the discovery, I should be taken for an impudent creature that had forged such a thing to go away from my husband, or should be counted crazed and distracted.
    TARGET: En second lieu je lui dis que ce n'était que parce qu'elle était en vie qu'il y avait quelque support à la découverte, et que tant qu'elle me reconnaîtrait pour sa fille, avec raison d'en être persuadée, personne d'autre n'en douterait; mais que si elle mourait avant la découverte, on me prendrait pour une créature imprudente qui avait forgé ce mensonge afin d'abandonner mon mari, ou on me considérerait comme folle et égarée.
 PREDICTED: Au lieu suivant , je lui dis que son être vivait était vivant , et que , pendant qu ' elle m ' avait pour son enfant , et qu ' elle me vit bien que j ' étais si bien sûr que je ne le , mais que si elle mourrait devant la découverte , je serais pris pour une créature pour aller sortir de mon mari , ou de s ' aventurer à mon mari , ou de folie .
--------------------------------------------------------------------------------
    SOURCE: Par Riscara, le prince reçut un avis anonyme qui l’avertissait qu’une expédition de la sentence de Fabrice avait été adressée officiellement au gouverneur de la citadelle.
    TARGET: From Riscara the Prince received an anonymous message informing him that a copy of Fabrizio's sentence had been officially addressed to the governor of the citadel.
 PREDICTED: For Riscara , the Prince received a kind which a of Fabrizio had been addressed to the governor of the citadel .
--------------------------------------------------------------------------------
Training on Epoch 05: 100%|██████████| 834/834 [03:23<00:00,  4.09it/s, Loss_Acc=3.022, Loss=2.940, Sqn_L=49]
--------------------------------------------------------------------------------
    SOURCE: Elle ajouta en se levant :
    TARGET: She added as she rose to her feet:
 PREDICTED: She added to herself :
--------------------------------------------------------------------------------
    SOURCE: With a little more heart, he might have been contented with this new conquest; but the principal features of his character were ambition and pride.
    TARGET: Aussi, avec un peu de coeur, se serait-il contenté de cette nouvelle conquête; mais d'Artagnan n'avait que de l'ambition et de l'orgueil.
 PREDICTED: Avec un peu de cœur , il eût été satisfait de cette nouvelle conquête ; mais les traits principaux de son caractère étaient et de son orgueil .
--------------------------------------------------------------------------------
Training on Epoch 06: 100%|██████████| 834/834 [03:23<00:00,  4.09it/s, Loss_Acc=2.776, Loss=2.785, Sqn_L=74]
--------------------------------------------------------------------------------
    SOURCE: En face de la porte, sur le coin d’une cheminée prétentieuse en imitation de marbre blanc, on remarquait le reste d’une bougie de cire rouge a moitié consumée.
    TARGET: Opposite the door was a showy fireplace, surmounted by a mantelpiece of imitation white marble. On one corner of this was stuck the stump of a red wax candle.
 PREDICTED: In the face of the door , on the platform of a in marble , was noticed the rest of a candle which hung a half of half .
--------------------------------------------------------------------------------
    SOURCE: – Ecoute, lui dit-elle, si tu peux donner une centaine de francs, je mettrai un double napoléon sur chacun des yeux du caporal qui va venir relever la garde pendant la nuit.
    TARGET: "Listen," she said to him, "if you can put down a hundred francs, I will place a double napoleon on each eye of the corporal who comes to change the guard during the night.
 PREDICTED: " ," she said to her , " if you may give a hundred francs , I shall go a double napoleon on each other eyes of the corporal who will come to the night for the night ."
--------------------------------------------------------------------------------
Training on Epoch 07: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=2.561, Loss=2.672, Sqn_L=63]
--------------------------------------------------------------------------------
    SOURCE: Is this the first time you have left your parents to come to school, my little girl?"
    TARGET: Est-ce la première fois que vous quittez vos parents pour venir en pension, mon enfant?»
 PREDICTED: Est - ce la première fois que vous avez laissé vos parents pour venir en pension , ma petite fille ?
--------------------------------------------------------------------------------
    SOURCE: About eleven I came unawares upon Booby keeping a lookout in a field close to the chapel.
    TARGET: J’ai surpris Ganache à onze heures en train de guetter dans un champ auprès de la chapelle.
 PREDICTED: À onze heures je m ’ approchai avec Ganache , en un champ à la chapelle .
--------------------------------------------------------------------------------
Training on Epoch 08: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=2.370, Loss=2.429, Sqn_L=93]
--------------------------------------------------------------------------------
    SOURCE: 'If the arrogance of the Marquise, or the mischievous pranks of her son, make the house definitely insupportable to you, I advise you to finish your studies in some Seminary thirty leagues from Paris, and in the North, rather than in the South.
    TARGET: Si les hauteurs de la marquise, ou les mauvaises plaisanteries de son fils, vous rendent cette maison décidément insupportable, je vous conseille de finir vos études dans quelque séminaire à trente lieues de Paris, et plutôt au nord qu’au midi.
 PREDICTED: Si l ’ arrogance de la marquise , ou les de son fils , vous faites une maison bien insupportable à vous , je vous conseille d ’ achever vos études de trente lieues de Paris , et au nord plutôt que au Sud .
--------------------------------------------------------------------------------
    SOURCE: "Did Rivers spend much time with the ladies of his family?"
    TARGET: -- Rivers passait-il beaucoup de temps auprès de vous et de ses soeurs?
 PREDICTED: -- Est - ce que M . Rivers passait beaucoup de temps avec les dames de sa famille ?
--------------------------------------------------------------------------------
Training on Epoch 09: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=2.206, Loss=2.300, Sqn_L=78]
--------------------------------------------------------------------------------
    SOURCE: "By Jove," said Spilett, "our friend Pencroft has become very particular about the proprieties all at once!"
    TARGET: -- Diable! fit Gédéon Spilett, notre ami Pencroff est à cheval sur les convenances!
 PREDICTED: -- Bon , dit Gédéon Spilett , notre ami Pencroff est devenu très particulièrement à tous les !
--------------------------------------------------------------------------------
    SOURCE: Why should it not, at a certain depth, attain an impassable limit, instead of rising to such a point as to fuse the most infusible metals?"
    TARGET: Pourquoi, à une certaine profondeur, n'atteindrait-elle pas une limite infranchissable, au lieu de s'élever jusqu'au degré de fusion des minéraux les plus réfractaires?»
 PREDICTED: Pourquoi n ' y serait - il pas , à une certaine profondeur , atteindre une limite infranchissable , jusqu ' à se lever pour amener à la mèche la plus de métaux ?
--------------------------------------------------------------------------------
Training on Epoch 10: 100%|██████████| 834/834 [03:23<00:00,  4.10it/s, Loss_Acc=2.106, Loss=2.146, Sqn_L=90]
--------------------------------------------------------------------------------
    SOURCE: She ventured to question him as to the portrait in which he took such an interest; Julien swore to her that it was that of a man.
    TARGET: Elle osa l’interroger sur le portrait auquel il mettait tant d’intérêt ; Julien lui jura que c’était celui d’un homme.
 PREDICTED: Elle osa le questionner avec ce portrait dans lequel il prenait tant d ’ intérêt ; Julien lui jura que c ’ était celui d ’ un homme .
--------------------------------------------------------------------------------
    SOURCE: "My dear Miss Elizabeth, I have the highest opinion in the world in your excellent judgement in all matters within the scope of your understanding; but permit me to say, that there must be a wide difference between the established forms of ceremony amongst the laity, and those which regulate the clergy; for, give me leave to observe that I consider the clerical office as equal in point of dignity with the highest rank in the kingdom--provided that a proper humility of behaviour is at the same time maintained.
    TARGET: – Ma chere miss Elizabeth, dit-il, j’ai la plus haute opinion de votre excellent jugement pour toutes les matieres qui sont de votre compétence. Mais permettez-moi de vous faire observer qu’a l’égard de l’étiquette les gens du monde et le clergé ne sont pas astreints aux memes regles.
 PREDICTED: – Mon cher miss Elizabeth , j ’ ai la plus grande opinion du monde dans votre esprit de contradiction avec votre intelligence et la diligence qui est la plus grande que vous ne pensez pas , mais permettez - moi de vous dire qu ’ il y a une grande différence entre les formes de la cérémonie du et ceux qui le clergé ; car , me présentant à l ’ honneur que je considere comme une dignité de dignité dans le royaume , pourvu qu ’ une conduite est de temps en même temps .
--------------------------------------------------------------------------------
Training on Epoch 11: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=2.071, Loss=2.100, Sqn_L=108]
--------------------------------------------------------------------------------
    SOURCE: He appealed to his son.
    TARGET: Il se tourna vers son fils.
 PREDICTED: Il fit un appel à son fils .
--------------------------------------------------------------------------------
    SOURCE: Cette petite surprise chassa l’ennui : « Voilà un gaillard, se dit-il, pour lequel on va me demander Dieu sait quelles faveurs, toutes celles dont je puis disposer.
    TARGET: This little surprise dispelled his boredom: "Here is a fellow," he said to himself, "for whom they will be asking me heaven knows what favours, everything that I have to bestow.
 PREDICTED: That little surprise drove the boredom : " There is a ," he said to himself , " to which one is to ask for me how God knows , all those which I can promise .
--------------------------------------------------------------------------------
Training on Epoch 12: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=2.041, Loss=2.077, Sqn_L=129]
--------------------------------------------------------------------------------
    SOURCE: As for Elizabeth, her thoughts were at Pemberley this evening more than the last; and the evening, though as it passed it seemed long, was not long enough to determine her feelings towards _one_ in that mansion; and she lay awake two whole hours endeavouring to make them out.
    TARGET: Quant a Elizabeth, ses pensées étaient a Pemberley ce soir-la plus encore que la veille. La fin de la journée lui parut longue mais ne le fut pas encore assez pour lui permettre de déterminer la nature exacte des sentiments qu’elle éprouvait a l’égard d’un des habitants du château, et elle resta éveillée deux bonnes heures, cherchant a voir clair dans son esprit.
 PREDICTED: Quant a Elizabeth , elle avait a Pemberley plus que la fin . La soirée , bien que le temps se fut bientôt passée , ne fut pas assez longue pour se faire sentir dans le château en personne , elle se réveilla en deux heures .
--------------------------------------------------------------------------------
    SOURCE: Am I more happy than she is? We have her cash, I have no need to constrain myself."
    TARGET: Est-ce que je suis plus heureux qu'elle, moi?… Nous avons son argent, je n'ai pas besoin de me gêner.
 PREDICTED: Est - ce que je suis plus heureuse que sa bourse ? nous avons sa fortune , je n ’ ai pas besoin de m ’ y rendre .
--------------------------------------------------------------------------------
Training on Epoch 13: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=2.014, Loss=2.063, Sqn_L=71]
--------------------------------------------------------------------------------
    SOURCE: "The miserable villain! He had foreseen all. His breast was covered with a coat-of-mail; the knife was bent against it.
    TARGET: «Le misérable! il avait tout prévu: sa poitrine était couverte d'une cotte de mailles; le couteau s'émoussa.
 PREDICTED: -- Le misérable ! il avait tout prévu , la poitrine se couvrit d ' un habit de ; le couteau s ' y .
--------------------------------------------------------------------------------
    SOURCE: A small picture presented the interior of an immensely long and rectangular vault or tunnel, with low walls, smooth, white, and without interruption or device.
    TARGET: C'était un petit tableau représentant l'intérieur d'une cave ou d'un souterrain immensément long, rectangulaire, avec des murs bas, polis, blancs, sans aucun ornement, sans aucune interruption.
 PREDICTED: Un petit tableau présentait l ' intérieur d ' une voûte assez considérable , de cette voûte ou de , aux murs bas blancs , blancs et sans rupture .
--------------------------------------------------------------------------------
Training on Epoch 14: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=1.989, Loss=1.866, Sqn_L=57]
--------------------------------------------------------------------------------
    SOURCE: "Knock some of the soot off him, Lord Frederick!" they shouted.
    TARGET: -- Secouez-lui un peu sa suie, Lord Frederick, criaient-ils.
 PREDICTED: « de la suie , de l ' eau , Lord Frederick ! cria - t - on .
--------------------------------------------------------------------------------
    SOURCE: Le pauvre homme a ajouté quelques supplications assez mal tournées et assez inopportunes après le mot “adieu” prononcé par moi.
    TARGET: The poor man added various supplications, by no means well expressed and distinctly inopportune after the word Good-bye had been uttered by me.
 PREDICTED: The poor man added a few gay times , and quite , after the words , uttered by me .
--------------------------------------------------------------------------------
Training on Epoch 15: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.967, Loss=2.113, Sqn_L=44]
--------------------------------------------------------------------------------
    SOURCE: She sank down on the low chair, trembling, with her forehead between her hands.
    TARGET: Elle retomba sur la chaise basse, frémissante, le front entre les mains.
 PREDICTED: Elle tomba sur la chaise basse , tremblante , le front entre ses mains .
--------------------------------------------------------------------------------
    SOURCE: "Yes, sire."
    TARGET: – Oui, Sire.
 PREDICTED: -- Oui , Sire .
--------------------------------------------------------------------------------
Training on Epoch 16: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.946, Loss=2.023, Sqn_L=73]
--------------------------------------------------------------------------------
    SOURCE: But it is never too late to be wise; and I cannot but advise all considering men, whose lives are attended with such extraordinary incidents as mine, or even though not so extraordinary, not to slight such secret intimations of Providence, let them come from what invisible intelligence they will.
    TARGET: Mais il n'est jamais trop tard pour être sage, et je ne puis que conseiller à tout homme judicieux dont la vie est exposée à des événements extraordinaires comme le fut la mienne, ou même à de moindres événements, de ne jamais mépriser de pareils avertissements intimes de la Providence, ou de n'importe quelle intelligence invisible il voudra.
 PREDICTED: Mais il n ' est pas trop tard pour être sage , et je ne saurais donner de conseils à des hommes dont la vie est accompagnée de circonstances extraordinaires , ou bien même elle ne doit pas avoir le temps d ' une extrême de la Providence , laissez - les venir de son intelligence invisible .
--------------------------------------------------------------------------------
    SOURCE: "I'm right sorry to be so late, Sir Charles," he cried.
    TARGET: -- Je suis bien fâché d'arriver aussi tard, Sir Charles.
 PREDICTED: -- Je suis fâché d ’ être si tard , Sir Charles , s ’ écria - t - il .
--------------------------------------------------------------------------------
Training on Epoch 17: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.927, Loss=1.873, Sqn_L=62]
--------------------------------------------------------------------------------
    SOURCE: Why has Brigham Young, our chief, been imprisoned, in contempt of all justice?
    TARGET: Céderons-nous à la force ?
 PREDICTED: Pourquoi a - t - il de tels que nous , notre chef , était emprisonné dans le mépris de toute justice ?
--------------------------------------------------------------------------------
    SOURCE: Various duties awaited me on my arrival. I had to sit with the girls during their hour of study; then it was my turn to read prayers; to see them to bed: afterwards I supped with the other teachers.
    TARGET: Différents devoirs m'attendaient à mon arrivée: il fallait rester avec les enfants pendant l'heure de l'étude; c'était à moi de lire les prières, d'assister au coucher des élèves; ensuite vint le souper avec les maîtresses; enfin, lorsque nous nous retirâmes, l'inévitable Mlle Gryee partagea encore ma chambre.
 PREDICTED: J ' étais prêt à rester avec les jeunes filles pendant l ' heure de l ' étude ; je me mis à lire ; j ' avais des prières pour les lire et les voir se coucher ; j ' en ai soupé plus tard avec les autres maîtresses .
--------------------------------------------------------------------------------
Training on Epoch 18: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=1.909, Loss=1.866, Sqn_L=60]
--------------------------------------------------------------------------------
    SOURCE: Cacambo set out the same day. This Cacambo was a very honest fellow.
    TARGET: Cacambo partit dès le jour même: c'était un très bon homme que ce Cacambo.
 PREDICTED: Cacambo aujourd ' hui même , et Cacambo était un garçon très honnête .
--------------------------------------------------------------------------------
    SOURCE: I mean he has better luck than you."
    TARGET: Je veux dire qu'il a plus de chance que toi.
 PREDICTED: Je veux dire qu ' il vaut mieux que toi .
--------------------------------------------------------------------------------
Training on Epoch 19: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.893, Loss=1.927, Sqn_L=58]
--------------------------------------------------------------------------------
    SOURCE: This was said with a careless, abstracted indifference, which showed that my solicitude was, at least in his opinion, wholly superfluous.
    TARGET: Il dit ces mots avec indifférence et d'un air absorbé, ce qui me prouva qu'à ses yeux ma sollicitude était au moins superflue.
 PREDICTED: Ceci fut dit avec une indifférence sans et sans indifférence et que j ’ aperçus que ma sollicitude était , du moins son opinion sur ses sentiments .
--------------------------------------------------------------------------------
    SOURCE: You know what Servius saith: '~Nullus enim locus sine genio est~,−−for there is no place that hath not its spirit.'"
    TARGET: Vous savez ce que dit Servius : Nullus enim locus sine genio est. »
 PREDICTED: Vous savez ce que dit : les y sont sans place qui n ’ a pas le ! »
--------------------------------------------------------------------------------
Training on Epoch 20: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.877, Loss=1.902, Sqn_L=76]
--------------------------------------------------------------------------------
    SOURCE: She did not seem to trouble herself in the least about Laurent. She barely looked at him, rarely exchanged a word with him, treating him with perfect indifference.
    TARGET: Il lui semblait que Laurent n'existât pas pour elle; elle le regardait à peine, lui adressait de rares paroles, le traitait avec une indifférence parfaite.
 PREDICTED: Elle ne paraissait pas se gêner du moins autour de Laurent ; à peine elle le regardait , bien que rarement , avec lui , le traitait avec une indifférence parfaite .
--------------------------------------------------------------------------------
    SOURCE: « Je crois en effet, répondit-il, que nous avons fait absolument tout ce qu’il y avait a faire ; toutefois c’est un cas fort curieux et, comme je connais votre gout pour ce qui est extraordinaire….
    TARGET: "I think we have done all that can be done," he answered; "it's a queer case though, and I knew your taste for such things."
 PREDICTED: " I think that we have done ," he answered , " that we have absolutely all the that there was to do . This is a very curious case , and , as I know your goutte to a common .
--------------------------------------------------------------------------------
Training on Epoch 21: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.863, Loss=1.831, Sqn_L=82]
--------------------------------------------------------------------------------
    SOURCE: Both resumed their way.
    TARGET: Et tous deux reprirent leur course.
 PREDICTED: Tous deux reprirent leur chemin .
--------------------------------------------------------------------------------
    SOURCE: D’Artagnan, hearing the Musketeer swear, wished to escape from the cloak, which blinded him, and sought to find his way from under the folds of it.
    TARGET: D'Artagnan, entendant jurer le mousquetaire, voulut sortir de dessous le manteau qui l'aveuglait, et chercha son chemin dans le pli.
 PREDICTED: D ' Artagnan , entendant le mousquetaire , voulut fuir du manteau qui l ' impassible , et voulut trouver sa route sous les plis de la rue .
--------------------------------------------------------------------------------
Training on Epoch 22: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.850, Loss=1.875, Sqn_L=132]
--------------------------------------------------------------------------------
    SOURCE: Julien, locked and double-locked in his room, was a prey to the most violent despair.
    TARGET: Julien, enfermé à double tour dans sa chambre, était en proie au plus violent désespoir.
 PREDICTED: Julien , fermée à double tour dans sa chambre , était en proie au plus fort désespoir .
--------------------------------------------------------------------------------
    SOURCE: "Come, let us go back to supper.
    TARGET: Ah! bah! allons nous remettre à table.
 PREDICTED: -- Allons , allons souper .
--------------------------------------------------------------------------------
Training on Epoch 23: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.838, Loss=1.801, Sqn_L=95]
--------------------------------------------------------------------------------
    SOURCE: A smile passed over the servant's face.
    TARGET: Un sourire passa sur la figure du domestique.
 PREDICTED: Un sourire passa sur la figure de la bonne .
--------------------------------------------------------------------------------
    SOURCE: St. John smiled.
    TARGET: Saint-John sourit.
 PREDICTED: Saint - John sourit .
--------------------------------------------------------------------------------
Training on Epoch 24: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.825, Loss=1.822, Sqn_L=87]
--------------------------------------------------------------------------------
    SOURCE: He had seen her, accompanied by her goat, take to the Rue de la Coutellerie; he took the Rue de la Coutellerie.
    TARGET: Il lui avait vu prendre, avec sa chèvre, la rue de la Coutellerie ; il avait pris la rue de la Coutellerie.
 PREDICTED: Il l ’ avait vue , accompagné de sa chèvre , prendre la rue de la Coutellerie , il prit la rue de la saut .
--------------------------------------------------------------------------------
    SOURCE: "What do you call him?"
    TARGET: – Comment le nommez-vous ?
 PREDICTED: -- Qu ' appelez - vous ?
--------------------------------------------------------------------------------
Training on Epoch 25: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.814, Loss=1.781, Sqn_L=106]
--------------------------------------------------------------------------------
    SOURCE: "Now, let me try."
    TARGET: --Maintenant laissez-moi essayer.
 PREDICTED: -- Maintenant , laissez - moi faire .
--------------------------------------------------------------------------------
    SOURCE: At last he mastered her arms; Grace Poole gave him a cord, and he pinioned them behind her: with more rope, which was at hand, he bound her to a chair.
    TARGET: Enfin il s'empara des bras de la folle, il les lui attacha derrière le dos avec une corde que lui donna Grace; avec une autre corde, il la lia à une chaise.
 PREDICTED: Enfin il se rendit à ses bras ; Grace Poole lui donna une corde , et les bras liés ; il était à une corde plus que lui , il la lui fallait sur une chaise .
--------------------------------------------------------------------------------
Training on Epoch 26: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.804, Loss=1.887, Sqn_L=63]
--------------------------------------------------------------------------------
    SOURCE: Charles's conversation was commonplace as a street pavement, and everyone's ideas trooped through it in their everyday garb, without exciting emotion, laughter, or thought.
    TARGET: La conversation de Charles était plate comme un trottoir de rue, et les idées de tout le monde y défilaient dans leur costume ordinaire, sans exciter d’émotion, de rire ou de rêverie.
 PREDICTED: La conversation de Charles était assez commune comme une rue , et les idées de tous , le , se trouvant en leurs vêtements de bonne et sans armes , sans grand air , ou songea .
--------------------------------------------------------------------------------
    SOURCE: And I do not want a stranger--unsympathising, alien, different from me; I want my kindred: those with whom I have full fellow- feeling.
    TARGET: Je ne veux pas d'un étranger qui serait différent de moi, et avec lequel je ne pourrais pas sympathiser.
 PREDICTED: -- Moi , monsieur , je ne veux pas qu ' un étranger .
--------------------------------------------------------------------------------
Training on Epoch 27: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=1.794, Loss=1.773, Sqn_L=54]
--------------------------------------------------------------------------------
    SOURCE: And do you know I've only one regret, that we didn't let the old man strangle the Piolaine girl.
    TARGET: Et, tu ne sais pas? je n'ai qu'un regret, c'est de n'avoir pas laissé le vieux étrangler la fille de la Piolaine…
 PREDICTED: Et savez - vous que je n ' ai plus qu ' un regret , que nous ne pas le vieux a étrangler la Piolaine .
--------------------------------------------------------------------------------
    SOURCE: "Sooner." said Aramis.
    TARGET: -- Tôt, dit Aramis.
 PREDICTED: -- tôt , dit Aramis .
--------------------------------------------------------------------------------
Training on Epoch 28: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.785, Loss=1.782, Sqn_L=46]
--------------------------------------------------------------------------------
    SOURCE: The grass stretches right up to the very base of the wall, and my feet made little noise until I reached the crumbling arch where the old gate used to be.
    TARGET: L'herbe monte jusqu'au bas même du mur, et mes pieds ne firent que peu de bruit jusqu'au moment où j'arrivai à l'arc coulant où se trouvait jadis l'entrée.
 PREDICTED: Les herbes se sont exacts jusqu ’ à la base du mur et mes pieds firent le bruit jusqu ’ à l ’ intérieur où se trouvait la vieille grille .
--------------------------------------------------------------------------------
    SOURCE: The Canadian looked carefully.
    TARGET: Le Canadien regarda attentivement.
 PREDICTED: Le Canadien regarda avec attention .
--------------------------------------------------------------------------------
Training on Epoch 29: 100%|██████████| 834/834 [03:27<00:00,  4.02it/s, Loss_Acc=1.776, Loss=1.783, Sqn_L=79]
--------------------------------------------------------------------------------
    SOURCE: During her absence, a neighbor had seen two gypsies ascend up to it with a bundle in their arms, then descend again, after closing the door.
    TARGET: Pendant son absence, une voisine avait vu deux égyptiennes y monter en cachette avec un paquet dans leurs bras, puis redescendre après avoir refermé la porte, et s’enfuir en hâte.
 PREDICTED: Pendant son absence , un voisin avait connu deux égyptiennes en y monter avec un paquet de leurs bras ; puis redescendre la porte , en fermant la porte .
--------------------------------------------------------------------------------
    SOURCE: Lastly, this liquid being partly evaporated, crystals of sulphate of iron were deposited, and the not evaporated liquid, which contained the sulphate of alumina, was thrown away.
    TARGET: Enfin, ce liquide s'étant vaporisé en partie, des cristaux de sulfate de fer se déposèrent, et les eaux-mères, c'est-à-dire le liquide non vaporisé, qui contenait du sulfate d'alumine, furent abandonnées.
 PREDICTED: Enfin , ce liquide se à demi , des cristaux de sulfate de fer fut déposés , et le long cours d ' alumine , qui contenaient l ' sulfate d ' alumine , s ' .
--------------------------------------------------------------------------------
Training on Epoch 30: 100%|██████████| 834/834 [03:24<00:00,  4.08it/s, Loss_Acc=1.768, Loss=1.815, Sqn_L=62]
--------------------------------------------------------------------------------
    SOURCE: Now there only remained the Piolaine people.
    TARGET: Maintenant, il ne lui restait que les bourgeois de la Piolaine.
 PREDICTED: Maintenant , il ne restait plus que les bourgeois de la Piolaine .
--------------------------------------------------------------------------------
    SOURCE: Je fais, en passant, un profond salut à la citadelle, que le courage de monsignore et l’esprit de Madame viennent de déshonorer ; je prends un sentier dans la campagne, de moi bien connu, et je fais mon entrée à la Ricciarda.
    TARGET: I make, as I pass it, a profound bow to the citadel, which Monsignore's courage and the Signora's spirit have succeeded in disgracing; I take a path across country, which I know well, and I make my entry into La Ricciarda."
 PREDICTED: I am breaking , on traversa and a profound to the citadel , that the courage of Monsignore and the spirit of the Signora have been able to ; I take a path through the country , and I am making my entry into La Ricciarda .
--------------------------------------------------------------------------------
Training on Epoch 31: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.760, Loss=1.754, Sqn_L=67]
--------------------------------------------------------------------------------
    SOURCE: "Your music subscription is out; am I to renew it?"
    TARGET: -- Votre abonnement de musique est terminé, dois-je le reprendre?
 PREDICTED: Votre affaire est en train de louer ; je le lui renouveler ?
--------------------------------------------------------------------------------
    SOURCE: He never met the eyes either of his mother or his brother; to avoid hisgaze theirs had become surprisingly alert, with the cunning of foes whofear to cross each other.
    TARGET: Jamais il ne rencontrait plus le regard de sa mère ou le regard de sonfrère. Leurs yeux pour s'éviter avaient pris une mobilité surprenanteet des ruses d'ennemis qui redoutent de se croiser.
 PREDICTED: Il ne rencontrait jamais les yeux ni de sa mère ni de son frère ; pour éviter les étaient et les de leur façon à montrer les plus tendres .
--------------------------------------------------------------------------------
Training on Epoch 32: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.752, Loss=1.716, Sqn_L=69]
--------------------------------------------------------------------------------
    SOURCE: She dared not even mention that gentleman.
    TARGET: Darcy. Elle n’osait meme pas nommer ce dernier.
 PREDICTED: Elle n ’ osa pas même parler de ce gentilhomme .
--------------------------------------------------------------------------------
    SOURCE: At first Étienne thought she was speaking of the low noise of the ever-rising water.
    TARGET: D'abord, Étienne crut qu'elle parlait du petit bruit de l'eau montant toujours.
 PREDICTED: D ' abord , Étienne crut parler au bruit bas du fleuve qui montait .
--------------------------------------------------------------------------------
Training on Epoch 33: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.746, Loss=1.781, Sqn_L=132]
--------------------------------------------------------------------------------
    SOURCE: You had to go down a rather steep slope paved here and there; then, taking two or three turns amongst weavers' back yards and empty stables, you came to a wide blind alley closed up by a farmyard long since deserted.
    TARGET: On descendait d’abord une pente assez raide, dallée de place en place, puis après avoir tourné deux ou trois fois, entre des petites cours de tisserands ou des écuries vides, on arrivait dans une large impasse fermée par une cour de ferme depuis longtemps abandonnée.
 PREDICTED: Il fallut descendre un peu de pente raide , vers le sud , puis prenant deux ou trois tours de tisserands autour de l ’ écurie vides et vide , vous êtes tombé dans une grande allée déserte depuis une longue cour déserte .
--------------------------------------------------------------------------------
    SOURCE: On his wrists and ankles could be seen great bruises.
    TARGET: À ses poignets et à ses chevilles se voyaient de larges meurtrissures.
 PREDICTED: On voyait sur ses poignets , aux grands contusions .
--------------------------------------------------------------------------------
Training on Epoch 34: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=1.739, Loss=1.710, Sqn_L=56]
--------------------------------------------------------------------------------
    SOURCE: But my husband, having so dexterously got out of the bailiff's house by letting himself down in a most desperate manner from almost the top of the house to the top of another building, and leaping from thence, which was almost two storeys, and which was enough indeed to have broken his neck, he came home and got away his goods before the creditors could come to seize; that is to say, before they could get out the commission, and be ready to send their officers to take possession.
    TARGET: Mais mon mari s'étant désespérément échappé de chez le baillif, en se laissant tomber presque du haut de la maison sur le haut d'un autre bâtiment d'où il avait sauté et qui avait presque deux étages, en quoi il manqua de bien peu se casser le cou, il rentra et emmena ses marchandises avant que les créanciers pussent venir saisir, c'est-à-dire, avant qu'ils eussent obtenu la commission à temps pour envoyer les officiers prendre possession.
 PREDICTED: Mais mon mari , ayant si adroitement été aussi adroitement hors de la maison du bailli , en se laissant à son cou le plus désespéré de la maison d ’ en haut en bas d ’ un autre logis et de là - dessus deux curieux , qui avait assez de faiblesse pour se rompre le cou , il revint chez lui ses marchandises que les créanciers n ’ eussent pu tenir , c ’ est - à - dire avant qu ’ on leur eût pu sortir les et prêts à prendre leur garde .
--------------------------------------------------------------------------------
    SOURCE: "Have you formed any plan, Cyrus?" asked the reporter.
    TARGET: -- Avez-vous un projet, Cyrus? demanda le reporter.
 PREDICTED: -- Avez - vous pris quelque plan , Cyrus ? demanda le reporter .
--------------------------------------------------------------------------------
Training on Epoch 35: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=1.733, Loss=1.763, Sqn_L=59]
--------------------------------------------------------------------------------
    SOURCE: After this, I made a great heavy pestle or beater of the wood called the iron-wood; and this I prepared and laid by against I had my next crop of corn, which I proposed to myself to grind, or rather pound into meal to make bread.
    TARGET: Je fis enfin une hie ou grand pilon avec de ce bois appelé _bois de fer_, et je mis de côté ces instruments en attendant ma prochaine récolte, après laquelle je me proposai de moudre mon grain, au plutôt de l'égruger, pour faire du pain.
 PREDICTED: Après cela , je fis de grands ou du bois qu ' on appelait le , et je en la récolte de blé , que je me à , ou plutôt de livre pour faire du pain .
--------------------------------------------------------------------------------
    SOURCE: I say again, all the gentlemen that do so ought to be used in the same manner, and then they would be cautious of themselves.
    TARGET: Je le répète encore, tous les gentilshommes qui agissent ainsi devraient être traités de la même manière, et cela les porterait à veiller sur leurs actions.
 PREDICTED: Je le répète , tous les gentilshommes qui devaient être aussi , et puis ils se d ’ eux - memes .
--------------------------------------------------------------------------------
Training on Epoch 36: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.728, Loss=1.687, Sqn_L=86]
--------------------------------------------------------------------------------
    SOURCE: Here, being detained by bad weather for some time, the captain, who continued the same kind, good-humoured man as at first, took us two on shore with him again.
    TARGET: Là, ayant été retenus par le mauvais temps, le capitaine qui continuait de montrer la même humeur charmante, nous emmena de nouveau tous deux à terre.
 PREDICTED: Là , le mauvais temps le capitaine , qui , pendant quelque temps , continuait le même homme d ' esprit , de bonne humeur qu ' au premier , nous emmena deux sur le rivage .
--------------------------------------------------------------------------------
    SOURCE: She raised her dry, red eyes to heaven, to the sun, to the silvery clouds, cut here and there by a blue trapezium or triangle; then she lowered them to objects around her, to the earth, the throng, the houses; all at once, while the yellow man was binding her elbows, she uttered a terrible cry, a cry of joy.
    TARGET: Elle leva ses yeux rouges et secs vers le ciel, vers le soleil, vers les nuages d’argent coupés çà et là de trapèzes et de triangles bleus, puis elle les abaissa autour d’elle, sur la terre, sur la foule, sur les maisons… Tout à coup, tandis que l’homme jaune lui liait les coudes, elle poussa un cri terrible, un cri de joie.
 PREDICTED: Elle la souleva , les yeux rouges au ciel , au soleil d ’ argent , frappa çà et là et là par un triangle bleu ou un triangle bleu ; puis elle les baissa pour se ranger autour d ’ elle , à la foule , les maisons ; toutes à la fois , tandis que l ’ homme jaune se penchait les coudes , elle poussa un cri terrible , un cri de joie .
--------------------------------------------------------------------------------
Training on Epoch 37: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.722, Loss=1.791, Sqn_L=47]
--------------------------------------------------------------------------------
    SOURCE: And thus, as a closer and still closer intimacy admitted me more unreservedly into the recesses of his spirit, the more bitterly did I perceive the futility of all attempt at cheering a mind from which darkness, as if an inherent positive quality, poured forth upon all objects of the moral and physical universe in one unceasing radiation of gloom.
    TARGET: Et ainsi, à mesure qu'une intimité de plus en plus étroite m'ouvrait plus familièrement les profondeurs de son âme, je reconnaissais plus amèrement la vanité de tous mes efforts pour ramener un esprit, d'où la nuit, comme une propriété qui lui aurait été inhérente, déversait sur tous les objets de l'univers physique et moral une irradiation incessante de ténèbres.
 PREDICTED: Ainsi , comme une plus profonde et encore plus profonde intimité m ' avait pris en moi un peu de son esprit , plus amèrement encore m ' de toutes les épreuves de sa vie par où s ' ils avaient pour moi une qualité plus positive , une qualité sûre et remplie de choses la morale et l ' univers physique sans relâche .
--------------------------------------------------------------------------------
    SOURCE: The pointed arch is found between the two.
    TARGET: L’ogive est entre deux.
 PREDICTED: L ' ogive est entre les deux .
--------------------------------------------------------------------------------
Training on Epoch 38: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.717, Loss=1.721, Sqn_L=63]
--------------------------------------------------------------------------------
    SOURCE: The hasty departure from London soon after the robbery; the large sum carried by Mr. Fogg; his eagerness to reach distant countries; the pretext of an eccentric and foolhardy bet--all confirmed Fix in his theory.
    TARGET: Ce départ précipité de Londres, peu de temps après le vol, cette grosse somme emportée, cette hâte d'arriver en des pays lointains, ce prétexte d'un pari excentrique, tout confirmait et devait confirmer Fix dans ses idées.
 PREDICTED: Le départ de Londres avant l ' vol , la somme énorme portait Mr . Fogg , son empressement à gagner des contrées éloignées ; le prétexte d ' une folie singulière et pari tout ce que lui avait fait l ' inspecteur de police .
--------------------------------------------------------------------------------
    SOURCE: But before he left, he again gazed at the canvases and said to Laurent:
    TARGET: Avant de partir, il regarda encore les toiles et dit à Laurent:
 PREDICTED: Mais , avant de partir , il regarda encore les toiles et dit à Laurent :
--------------------------------------------------------------------------------
Training on Epoch 39: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.712, Loss=1.685, Sqn_L=60]
--------------------------------------------------------------------------------
    SOURCE: The blows which he received increased greatly his esteem and friendship for Monsieur de Treville.
    TARGET: Les coups qu'il en reçut lui donnèrent beaucoup d'estime et d'amitié pour M. de Tréville.
 PREDICTED: Les coups qu ' il recevait fort son estime et son amitié pour M . de Tréville .
--------------------------------------------------------------------------------
    SOURCE: "The great cardinal!"
    TARGET: -- Le grand cardinal!
 PREDICTED: -- Le grand cardinal !
--------------------------------------------------------------------------------
Training on Epoch 40: 100%|██████████| 834/834 [03:25<00:00,  4.07it/s, Loss_Acc=1.708, Loss=1.742, Sqn_L=82]
--------------------------------------------------------------------------------
    SOURCE: But incredulity and indifference were evidently my strongest cards.
    TARGET: L’incrédulité et l’indifférence demeuraient mes atouts majeurs.
 PREDICTED: Mais l ' incrédulité et de ne pas croire aux plus sérieuses mes forces .
--------------------------------------------------------------------------------
    SOURCE: Indeed, the Committee of the Thames Angler's Association did recommend its adoption about two years ago, but some of the older members opposed it.
    TARGET: En effet, il y a deux ans, l’Association des pecheurs a la ligne de la Tamise a recommandé son usage, mais quelquesuns de ses membres les plus anciens s’y opposerent.
 PREDICTED: En effet , l ’ Association de la Tamise se dirigea d ’ abord vers son , mais l ’ on s ’ y est opposée un des membres de la vieille .
--------------------------------------------------------------------------------
Training on Epoch 41: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.703, Loss=1.689, Sqn_L=77]
--------------------------------------------------------------------------------
    SOURCE: But he was alone!
    TARGET: Mais il était seul!
 PREDICTED: Mais il était seul !
--------------------------------------------------------------------------------
    SOURCE: I have been a selfish being all my life, in practice, though not in principle. As a child I was taught what was right, but I was not taught to correct my temper.
    TARGET: J’ai vécu jusqu’ici en égoiste : enfant, on m’a enseigné a faire le bien, mais on ne m’a pas appris a corriger mon caractere.
 PREDICTED: J ' ai été un être égoïste toute ma vie , à l ' exercice , mais je n ' ai pas enseigné le bien , mais je n ' ai pas été habituée à me corriger .
--------------------------------------------------------------------------------
Training on Epoch 42: 100%|██████████| 834/834 [03:25<00:00,  4.06it/s, Loss_Acc=1.699, Loss=1.720, Sqn_L=71]
--------------------------------------------------------------------------------
    SOURCE: "No freight.
    TARGET: -- Des cailloux dans le ventre.
 PREDICTED: -- Ne pas .
--------------------------------------------------------------------------------
    SOURCE: Cahusac immediately ran to the Guardsman whom Aramis had killed, seized his rapier, and returned toward d’Artagnan; but on his way he met Athos, who during his relief which d’Artagnan had procured him had recovered his breath, and who, for fear that d’Artagnan would kill his enemy, wished to resume the fight.
    TARGET: Cahusac courut à celui des gardes qu'avait tué Aramis, s'empara de sa rapière, et voulut revenir à d'Artagnan; mais sur son chemin il rencontra Athos, qui, pendant cette pause d'un instant que lui avait procurée d'Artagnan, avait repris haleine, et qui, de crainte que d'Artagnan ne lui tuât son ennemi, voulait recommencer le combat.
 PREDICTED: Cahusac se retourna aussitôt au garde qu ' Aramis avait tué , saisit son empire et revint près de d ' Artagnan ; mais , sur sa route , il rencontra Athos , qui pendant son aide que d ' Artagnan l ' avait faite tout à fait , et qui , par crainte que d ' Artagnan son ennemi , voulut reprendre le combat .
--------------------------------------------------------------------------------
Training on Epoch 43: 100%|██████████| 834/834 [03:26<00:00,  4.04it/s, Loss_Acc=1.696, Loss=1.655, Sqn_L=98]
--------------------------------------------------------------------------------
    SOURCE: 'I must be getting somewhere near the centre of the earth.
    TARGET: « Je dois être bien près du centre de la terre.
 PREDICTED: Il faut que je m ’ quelque part au centre de la terre .
--------------------------------------------------------------------------------
    SOURCE: They were seized with fever and delirium, and this obstacle, in their minds, became material.
    TARGET: C'était comme un obstacle ignoble qui les séparait.
 PREDICTED: Ils avaient eu l ' idée de la fièvre et de l ' ivresse , cet obstacle , l ' esprit .
--------------------------------------------------------------------------------
Training on Epoch 44: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=1.692, Loss=1.728, Sqn_L=78]
--------------------------------------------------------------------------------
    SOURCE: Among the thousands of visages which that light tinged with scarlet, there was one which seemed, even more than all the others, absorbed in contemplation of the dancer.
    TARGET: Parmi les mille visages que cette lueur teignait d’écarlate, il y en avait un qui semblait plus encore que tous les autres absorbé dans la contemplation de la danseuse.
 PREDICTED: Parmi les visages de certains petits tas qui de écarlate , il y avait une qui semblait , plus que tous , absorbé dans la contemplation de la danseuse .
--------------------------------------------------------------------------------
    SOURCE: However, a fire could be made by means of the moss and dry brushwood, which covered certain parts of the plateau.
    TARGET: Cependant, on pouvait obtenir du feu au moyen des mousses et des broussailles sèches qui hérissaient certaines portions du plateau.
 PREDICTED: On pouvait , en effet , un feu , les terrains marin et les taillis sèches , qui couvraient certains parties du plateau .
--------------------------------------------------------------------------------
Training on Epoch 45: 100%|██████████| 834/834 [03:26<00:00,  4.03it/s, Loss_Acc=1.689, Loss=1.698, Sqn_L=69]
--------------------------------------------------------------------------------
    SOURCE: One of the first days after his return he came down to see us, and there for the first time he clapped eyes upon de Lapp.
    TARGET: Dès les premiers jours de son retour, il descendit pour nous rendre visite, et alors ses yeux se portèrent pour la première fois sur de Lapp.
 PREDICTED: Un des premiers jours après son retour , il descendit voir , et là , pour la première fois , il frappa des yeux de Lapp .
--------------------------------------------------------------------------------
    SOURCE: Les adieux furent tristes ; Robert partit avec le comte Pietranera qui suivait les Français dans leur retraite sur Novi.
    TARGET: Their parting was a sad one; Robert set forth with Conte Pietranera, who followed the French in their retirement on Novi.
 PREDICTED: The most were ; Robert went on with Conte Pietranera , who was following the French to retire on their retreat .
--------------------------------------------------------------------------------
Training on Epoch 46: 100%|██████████| 834/834 [03:26<00:00,  4.03it/s, Loss_Acc=1.686, Loss=1.659, Sqn_L=60]
--------------------------------------------------------------------------------
    SOURCE: "You're a deal changed from what you used to be, Jack," said she, looking at me sideways from under her dark lashes.
    TARGET: -- Vous êtes bien changé de ce que vous étiez autrefois, disait- elle en me regardant de côté par-dessous ses cils noirs.
 PREDICTED: -- Vous avez bien changé de ce que vous étiez servi , Jock , me dit - elle en me regardant de côté sous ses sourcils noirs .
--------------------------------------------------------------------------------
    SOURCE: January 8.
    TARGET: "8 janvier.
 PREDICTED: " 8 janvier .
--------------------------------------------------------------------------------
Training on Epoch 47: 100%|██████████| 834/834 [03:25<00:00,  4.05it/s, Loss_Acc=1.683, Loss=1.643, Sqn_L=45]
--------------------------------------------------------------------------------
    SOURCE: "Yes, the document which we found enclosed in a bottle, giving us the exact position of Tabor Island!"
    TARGET: -- Oui, ce document enfermé dans une bouteille que nous avons trouvé, et qui donnait la situation exacte de l'île Tabor!»
 PREDICTED: -- Oui , le document que nous avons trouvé enfermé dans une bouteille , et nous donnait le point de vue exactement de l ' île Tabor !
--------------------------------------------------------------------------------
    SOURCE: XVII
    TARGET: CHAPITRE XVII
 PREDICTED: XVII
--------------------------------------------------------------------------------
Training on Epoch 48: 100%|██████████| 834/834 [03:23<00:00,  4.09it/s, Loss_Acc=1.681, Loss=1.767, Sqn_L=109]
--------------------------------------------------------------------------------
    SOURCE: Entraîné par les événements, nous n’avons pas eu le temps d’esquisser la race comique de courtisans qui pullulent à la cour de Parme et faisaient de drôles de commentaires sur les événements par nous racontés.
    TARGET: Carried away by the train of events, we have not had time to sketch the comic race of courtiers who swarm at the court of Parma and who made fatuous comments on the incidents which we have related.
 PREDICTED: by such events , we have not time to hear the public of courtiers who betrayed at the court of Parma , and for the comments of the events on we have just related .
--------------------------------------------------------------------------------
    SOURCE: One Sunday, Camille, Therese and Laurent left for Saint-Ouen after breakfast, at about eleven o'clock.
    TARGET: Un dimanche, Camille, Thérèse et Laurent partirent pour Saint-Ouen vers onze heures, après le déjeuner.
 PREDICTED: Un dimanche , Camille et Laurent s ' étaient quittés pour la Saint - Ouen , en onze heures environ .
--------------------------------------------------------------------------------
Training on Epoch 49: 100%|██████████| 834/834 [03:24<00:00,  4.07it/s, Loss_Acc=1.678, Loss=1.684, Sqn_L=89]
--------------------------------------------------------------------------------
    SOURCE: They had formed their plans: on rising from table, Paul Négrel was to take the ladies to a mine, Saint-Thomas, which had been luxuriously reinstalled.
    TARGET: C'était toute une partie projetée: en sortant de table, Paul Négrel devait faire visiter a ces dames une fosse, Saint-Thomas, qu'on réinstallait avec luxe.
 PREDICTED: Ils avaient préparé leurs projets ; sur des qu ' ils allaient , Paul Négrel devait prendre les dames chez une mine , Saint - Thomas , qui avait été .
--------------------------------------------------------------------------------
    SOURCE: No one is unaware of the existence of that great warm-water current known by name as the Gulf Stream.
    TARGET: Personne n'ignore l'existence de ce grand courant d'eau chaude connu sous le nom de Gulf Stream.
 PREDICTED: On n ' ignore pas l ' existence de ce grand courant de mer tiede , connu par son nom comme le Gulf - Stream .
```
