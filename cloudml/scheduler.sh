#!/bin/sh
BRANCH_NAME="master"
gcloud scheduler jobs create http ${PROJECT_ID}-run-trigger \
    --schedule='35 21 * * *' \
    --uri=https://cloudbuild.googleapis.com/v1/projects/${PROJECT_ID}/triggers/${TRIGGER_ID}:run \
    --message-body={"branchName":"master"} \
    --oauth-service-account-email=${PROJECT_ID}@appspot.gserviceaccount.com \
    --oauth-token-scope=https://www.googleapis.com/auth/cloud-platform
