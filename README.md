# Key-Value Memory Networks in Facebook bAbI Dataset

This repo contains the implementation of [Key Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126) in Tensorflow. The model is tested on [bAbI](http://arxiv.org/abs/1502.05698).

## Output
 - The story is generated dynamically from the Dataset.
 - Question section will be initially filled with a question relevant to the story.
 - The user can change the question related to the given story and predict the output.
 
<div class="display: inline-block; float: left;">
 <img src="/output/1.png" width="436px" alt="Result 1">
 <img src="/output/r2.png" width="436px" alt="Result 2">
</div>

## Results
After training the model with 1k dataset for 100 epoch by individually and in joint mode, we found that many tasks are performing with less than 90% of accuracy, whereas their performance is better in case of individual training.


<img src="/output/1K.png" alt="Performance of QA model with 1K Dataset">

After training the model with 10k dataset for 200 epoch the performance of many tasks has crossed more than 90%. Only few tasks like "path_finding" and "where_was_object" are failing very badly due to the rigorous dependency on the previous facts(sentences).


<img src="/output/10K.png" alt="Performance of QA model with 10K Dataset" >

Performance Comparison between BoW and GRU (Feature embedding methods)


<img src="/output/result.PNG" alt="Performance Comparison between BoW and GRU" >

## Requirements

1. Python 2.7
2. numpy
3. tensorflow==0.12.1
4. pandas
5. Flask


## Dataset

1. Download the dataset from Jason Wetson's(One of the authors for dataset creation) page http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
2. Unzip the dataset inside c in a data directory such that all extracted directories(en, en-10k, hn) appear on data/tasks_1-20_v1-2 path.

     (Or)

1. Simply extract the data.rar to the project's home directory.

## Training
Training and testing are done on both en(1k) and en-10k dataset.

To train the model, we have two files one to train individually all task by(single.py) and training all tasks jointly using (joint.py).

### Single training

```
# Train the model on a single task (Default - task 1 with 1k dataset)
python single.py
```
Pass the task_id to train the model on a specific task
```
python single.py --task_id <task_id between 1-20>
```
There are several flags with single.py
```
python single.py --task_id <any task ID between 1-20> --data_dir <data/tasks_1-20_v1-2/en-10k for 10k dataset> --reader <bow, simple_gru>
```
To run all the tasks individually at one go.
```
bash train_allTask_single.sh
```
The models will be generated under 
 - models/models-1k/task-<ID> folder for each task for 1k dataset.
 - models/models-10k/task-<ID> folder for each task for 10k dataset

The Model save location can be modified by making change in line 174 of single.py by giving path as ./models/models-10k/task-{}/model.ckpt".format(FLAGS.task_id).

### Joint Training

```
# Train all the tasks from 1k Dataset
python joint.py
```
To train the model from 10k dataset.
```
python joint.py --data_dir data/tasks_1-20_v1-2/en-10k
```
The models will be generated under
 - models/models-1k/joint folder for each task for 1k dataset
 - models/models-10k/joint folder for each task for 10k dataset

The Model save location can be modified by making change in line 216 of joint.py by giving path as "./models/models-10k/joint/model.ckpt".

### Logging

All the log files will be generated in logs directory

 - The final accuracy score for 1k dataset will be generated in project home directory as "single_scores.csv" file.
 - The final accuracy score for 10k dataset will be generated in logs directory as csv file.

## Webapp

For demonstration purpose, we have created a simple webapp which populates randomly selected stories from dataset and initially a related question. A user can change to other question related to the given story and predict the output. User can also populate different story by clicking on Get new story button. It evaluates the question on given story based on a pre trained model. 

Before running the below code , make sure "./models/joint/" folder has some model populated or train the join model.
```
python webapp.py
#Go to browser and open 127.0.0.1:5000
```

## Output
Output folder has some results generated as part of training.
