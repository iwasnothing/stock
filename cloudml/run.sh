#!/bin/sh

if [ $1 == "cloud" ] ; then
  TS=`date +%Y%m%d%H%M`
  job_id="stock_training_${TS}"
  gcloud ai-platform jobs submit training $job_id --module-name trainer.task_simple_model --package-path trainer/ \
  --region "us-central1" \
  --python-version 3.7 \
  --runtime-version 2.1 \
  --job-dir "gs://iwasnothing-cloudml-job-dir/stock-train" \
  --config hp-config.yaml
else
  gcloud ai-platform local train --module-name trainer.task --package-path /Users/kahingleung/PycharmProjects/coursera/cloudml/trainer \
  --job-dir local-training-output
fi
