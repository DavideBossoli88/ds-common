## Prerequisites

# Create an IAM role with AmazonElasticContainerRegistryPublicFullAccess policy
# Create a Sagemaker instance using the IAM role above

## Usage

# 0. Open a terminal, move to the root directory of this project
# 1. Run the command "sh build.sh"
# 2. Run on command "sh push.sh ds-common-0 default"
# 3. When the container is pushed, we can test it using the test_container.ipynb notebook.

## Notes
# ds-common-0 is the name of the container pushed in ECR. It can be edited in the build.sh file and when running the push.sh command in the command line