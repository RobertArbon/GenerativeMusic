#!/bin/bash

# echo add in a line to upload the logs as well. 


udir=`date +%Y-%m-%d_%H-%M`
checkpoint_dir=`find riffusion-guzheng-v2/ -name 'checkpoint*' -type d`
gsutil -m cp -r ${checkpoint_dir} gs://generative_music/${udir}/${checkpoint_dir}
gsutil -m cp riffusion-guzheng-v2/config.json gs://generative_music/${udir}/config.json
gsutil -m cp riffusion-guzheng-v2/model_index.json gs://generative_music/${udir}/model_index.json
gsutil -m cp -r riffusion-guzheng-v2/logs gs://generative_music/${udir}/logs
for wav in `find riffusion-guzheng-v2/ -name '*.wav' -type f`;
do
    gsutil -m cp -r ${wav} gs://generative_music/${udir}/${wav}
done

for jpg in `find riffusion-guzheng-v2/ -name '*.jpg' -type f`;
do
    gsutil -m cp -r ${jpg} gs://generative_music/${udir}/${jpg}
done

