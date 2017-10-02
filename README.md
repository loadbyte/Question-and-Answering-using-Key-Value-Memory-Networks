Description:- Implementation of Key-value hashed memory network for facebook bAbI dataset using BoW and GRN.

Requirements:-
1. Python 2.7
2. numpy
3. tensorflow==0.12.1
4. pandas
5. Flask

Dataset:-
----------------------------------------------
1. Download the dataset from Jasen Wetson's(One of the author for dataset creation)page http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
2. Unzip the dataset inside projects home directory in a data directory such that all extracted directories(en, en-10k, hn) appears on data/tasks_1-20_v1-2 path.

Taining and testing is done on both en and en-10k dataset.

Training:-
-----------------------

To train the model we have two files one to train individually all task by(single.py) and training all tasks jointly using (joint.py).
Single training:-
1. Use $python single.py  (It will by default train for task 1 with 1k dataset), we can pass different arguments to work on different task and dataset.
2. $python single.py --task_id <any task ID between 1-20> --data_dir <data/tasks_1-20_v1-2/en-10k for 10k dataset> --reader <bow, simple_gru> to run on different tasks.
3. train_allTask_single.sh can be used to run all task individually at one go.
The models will be generated under models/models-1k/task-<ID> folder for each task for 1k dataset and for 10k dataset it will get stored in models/models-10k/task-<ID> by
making change in line 179 of single.py by giving path as ./models/models-10k/task-{}/model.ckpt".format(FLAGS.task_id).

Joint Training:-
1. $python joint.py  -It will train all the tasks combindely from 1k dataset.
2. $python joint.py --data_dir data/tasks_1-20_v1-2/en-10k   will train on data from 10k dataset.
The models will be generated under models/models-1k/joint folder for each task for 1k dataset and for 10k dataset it will get stored in models/models-10k/joint by
making change in line 179 of single.py by giving path as ./models/models-10k/joint/model.ckpt".

-- All the log files will be generated in logs directory.

The final accuracy score for 1k dataset will be generated in project home directory as "single_scores.csv" file.
The final accuracy score for 10k dataset will be generated in logs directory as csv file.


Webapp:-
-------------------------------------
For demonstration purpose we have created a simple webapp which populates randomly selected stories from dataset and initially a related question. A user can change to other question related to
the given story and predict the output. User can also populate different story by clicking on Get new story button. It evaluates the question on given story based on a pre trained model. 
before running the below code , make shore "./models/joint/" folder has some model populated or train the join model
1. $python webapp.py
2. Go to browser and open 127.0.0.1:5000


Output:-
--------------------
1. Output folder has some outpus generated as part of training.


