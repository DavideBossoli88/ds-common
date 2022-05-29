## Git

# Clone a repository (first time only)
git clone https://<token>@github.com/<user>/ds-common.git

# Create a new feature branch
git checkout -b feature_branch_local

# Setup user (first time only)
git config --global user.email "<email>"
git config --global user.name "<user>"


## Install H2O on Anaconda (runs on anaconda prompt)
conda install -c h2oai h2o